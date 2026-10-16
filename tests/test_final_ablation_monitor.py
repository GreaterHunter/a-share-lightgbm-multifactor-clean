"""Independent monitor tests: temporary artifacts and fake processes only."""
import json
from types import SimpleNamespace

import psutil
import pytest

from scripts import inspect_final_ablation as monitor


class FakeProcess:
    def __init__(self, pid, *, created=990, cpu=10, read=100, write=200, worker=False):
        self.pid = pid
        self.created, self.cpu, self.read, self.write = created, cpu, read, write
        self.worker = worker
        self.child_processes = []
        self.directory = str(monitor.PROJECT_ROOT)
        self.executable = "D:/python/python.exe"

    def create_time(self):
        return self.created

    def cmdline(self):
        return [self.executable, "-m", monitor.MODULE, *(["--worker", "{}"] if self.worker else [])]

    def cwd(self):
        return self.directory

    def exe(self):
        return self.executable

    def status(self):
        return psutil.STATUS_RUNNING

    def cpu_times(self):
        return SimpleNamespace(user=self.cpu, system=0)

    def io_counters(self):
        return SimpleNamespace(read_bytes=self.read, write_bytes=self.write, read_count=1, write_count=1)

    def memory_info(self):
        return SimpleNamespace(rss=1000)

    def children(self, recursive=True):
        return self.child_processes

    def is_running(self):
        return True


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def study(tmp_path, monkeypatch):
    runner = FakeProcess(101)
    child = FakeProcess(102, created=1001, worker=True)
    runner.child_processes = [child]
    processes = {101: runner, 102: child}

    def process(pid):
        if pid not in processes:
            raise psutil.NoSuchProcess(pid)
        return processes[pid]

    monkeypatch.setattr(monitor.psutil, "Process", process)
    monkeypatch.setattr(monitor, "current_code_hash", lambda: "current-code")
    monkeypatch.setattr(monitor.psutil, "virtual_memory", lambda: SimpleNamespace(total=8000, available=4000, percent=50))
    monkeypatch.setattr(monitor.psutil, "disk_usage", lambda _: SimpleNamespace(total=10000, free=6000, percent=40))
    write_json(tmp_path / "runner_process.json", {"pid": 101, "started": 1000,
        "stage": "all", "executable": runner.executable, "code_sha256": "current-code"})
    write_json(tmp_path / "pipeline_status.json", {"stage": "model", "status": "running", "updated": 1002})
    write_json(tmp_path / "current_job.json", {"status": "running", "started": 1001,
        "worker_pid": 102, "command": child.cmdline(), "code_sha256": "current-code",
        "job": {"kind": "anchor", "spec": {"name": "example"}}, "log": "logs/worker.log"})
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/worker.log").write_text("training started\n", encoding="utf-8")
    for name in ("runner_stdout.log", "runner_stderr.log"):
        (tmp_path / name).write_text("", encoding="utf-8")
    write_json(tmp_path / "tasks/current.json", {"status": "complete", "kind": "validation_fold", "code_sha256": "current-code"})
    write_json(tmp_path / "tasks/old.json", {"status": "failed", "kind": "account", "code_sha256": "old-code"})
    return SimpleNamespace(directory=tmp_path, runner=runner, child=child, processes=processes)


def test_first_snapshot_unknown_and_default_read_only(study):
    before = {path.relative_to(study.directory): path.read_bytes() for path in study.directory.rglob("*") if path.is_file()}
    snapshot = monitor.inspect(study.directory, now=5000)
    after = {path.relative_to(study.directory): path.read_bytes() for path in study.directory.rglob("*") if path.is_file()}
    assert snapshot["status"] == "unknown"
    assert snapshot["runner"]["status"] == "verified"
    assert snapshot["progress"]["no_progress_intervals"] == 0
    assert snapshot["tasks"]["counts"] == {"complete": 1}
    assert snapshot["tasks"]["kind_status_counts"] == {"validation_fold:complete": 1}
    assert snapshot["tasks"]["other_code_count"] == 1
    assert snapshot["system"]["memory_available_bytes"] == 4000
    assert snapshot["system"]["d_disk"]["free_bytes"] == 6000
    assert before == after


def test_dead_pid_is_reported_without_restart(study):
    study.processes.pop(101)
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["status"] == "runner_missing"
    assert snapshot["runner"]["status"] == "missing"
    assert not (study.directory / "monitor").exists()


def test_reused_pid_with_matching_command_is_rejected(study):
    study.runner.created = 4900
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["status"] == "identity_mismatch"
    assert snapshot["runner"]["identity_checks"]["command"] is True
    assert snapshot["runner"]["identity_checks"]["create_time"] is False


@pytest.mark.parametrize("field,value", [("directory", "D:/wrong-project"), ("executable", "D:/other/python.exe")])
def test_wrong_working_directory_or_executable_is_rejected(study, field, value):
    setattr(study.runner, field, value)
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["runner"]["status"] == "identity_mismatch"


def test_access_denied_is_unknown(study, monkeypatch):
    def denied():
        raise psutil.AccessDenied(101)
    monkeypatch.setattr(study.runner, "cmdline", denied)
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["status"] == "unknown"
    assert snapshot["runner"]["status"] == "access_denied"


def test_child_cpu_advances_even_when_log_does_not(study):
    first = monitor.inspect(study.directory, now=5000)
    study.child.cpu += 100
    second = monitor.inspect(study.directory, now=6800, previous=first)
    assert first["logs"] == second["logs"]
    assert second["status"] == "running"
    assert second["progress"]["status"] == "progress"
    assert second["progress"]["counter_deltas"]["cpu_seconds"] == 100
    assert second["progress"]["no_progress_intervals"] == 0


def test_io_advances_without_cpu_or_log(study):
    first = monitor.inspect(study.directory, now=5000)
    study.child.write += 500
    second = monitor.inspect(study.directory, now=6800, previous=first)
    assert second["progress"]["status"] == "progress"
    assert second["progress"]["counter_deltas"]["io_write_bytes"] == 500


def test_parent_heartbeat_and_helper_activity_do_not_hide_worker_stall(study):
    helper = FakeProcess(103, created=1001, cpu=0)
    helper.cmdline = lambda: ["conhost.exe"]
    study.runner.child_processes.append(helper)
    first = monitor.inspect(study.directory, now=5000)
    study.runner.cpu += 1
    study.runner.write += 10000
    helper.cpu += 100
    second = monitor.inspect(study.directory, now=6800, previous=first)
    study.runner.cpu += 1
    study.runner.write += 10000
    helper.cpu += 100
    third = monitor.inspect(study.directory, now=8600, previous=second)
    assert second["progress"]["status"] == "no_progress"
    assert third["status"] == "stalled_suspected"
    assert third["progress"]["process_scope"] == "research_workers"
    assert third["progress"]["counter_deltas"]["cpu_seconds"] == 0


def test_short_intervals_do_not_count_or_discard_full_baseline(study):
    first = monitor.inspect(study.directory, now=5000)
    short = monitor.inspect(study.directory, now=5060, previous=first)
    assert short["progress"]["status"] == "interval_too_short"
    assert short["progress"]["no_progress_intervals"] == 0
    assert short["progress"]["baseline"]["observed_at"] == 5000
    first_interval = monitor.inspect(study.directory, now=6800, previous=short)
    assert first_interval["status"] == "no_progress"
    short_again = monitor.inspect(study.directory, now=6810, previous=first_interval)
    assert short_again["progress"]["no_progress_intervals"] == 1
    assert short_again["status"] != "stalled_suspected"


def test_two_full_unchanged_intervals_suspect_stall_then_cpu_clears_it(study):
    first = monitor.inspect(study.directory, now=5000)
    second = monitor.inspect(study.directory, now=6800, previous=first)
    third = monitor.inspect(study.directory, now=8600, previous=second)
    assert second["status"] == "no_progress"
    assert second["progress"]["no_progress_intervals"] == 1
    assert third["status"] == "stalled_suspected"
    assert third["progress"]["no_progress_intervals"] == 2
    study.child.cpu += 1
    fourth = monitor.inspect(study.directory, now=8660, previous=third)
    assert fourth["status"] == "running"
    assert fourth["progress"]["no_progress_intervals"] == 0


def test_inaccessible_child_counters_cannot_suspect_stall(study, monkeypatch):
    def denied():
        raise psutil.AccessDenied(102)
    monkeypatch.setattr(study.child, "cpu_times", denied)
    first = monitor.inspect(study.directory, now=5000)
    second = monitor.inspect(study.directory, now=6800, previous=first)
    third = monitor.inspect(study.directory, now=8600, previous=second)
    assert third["status"] == "unknown"
    assert third["progress"]["no_progress_intervals"] == 0


@pytest.mark.parametrize("source", ["pipeline_status", "current_job"])
def test_explicit_training_failure_overrides_first_observation(study, source):
    path = study.directory / f"{source}.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(status="failed", error="worker failed", updated=1100)
    write_json(path, state)
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["status"] == "failed"


def test_record_keeps_history_and_reads_latest_baseline(study):
    first = monitor.record_snapshot(monitor.inspect(study.directory, now=5000))
    second = monitor.record_snapshot(monitor.inspect(study.directory, now=6800))
    third = monitor.record_snapshot(monitor.inspect(study.directory, now=8600))
    histories = list((study.directory / "monitor").glob("snapshot_*.json"))
    latest = json.loads((study.directory / "monitor/latest.json").read_text(encoding="utf-8"))
    assert len(histories) == 3
    assert first["recorded_snapshot"] != second["recorded_snapshot"]
    assert latest["status"] == third["status"] == "stalled_suspected"
    assert not list((study.directory / "monitor").glob("*.tmp"))


def test_cli_explicit_experiment_dir_is_read_only(study, capsys):
    assert monitor.main(["--experiment-dir", str(study.directory)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["experiment_dir"] == str(study.directory)
    assert result["status"] == "unknown"
    assert not (study.directory / "monitor").exists()


def test_unavailable_disk_api_is_reported_without_crashing(study, monkeypatch):
    def unavailable(_):
        raise SystemError("incompatible psutil binary")
    monkeypatch.setattr(monitor.psutil, "disk_usage", unavailable)
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["system"]["d_disk"]["status"] == "unavailable"
    assert "SystemError" in snapshot["system"]["d_disk"]["error"]


def test_recovery_running_requires_current_code_and_verified_worker(study):
    study.processes.pop(101)
    write_json(study.directory / "runner_process.json", {"pid": 101, "started": 900,
        "stage": "all", "executable": study.runner.executable, "code_sha256": "old-code"})
    write_json(study.directory / "pipeline_status.json", {"stage": "model", "status": "failed", "updated": 950})
    write_json(study.directory / "recovery_acceptance.json", {"status": "running", "started": 1000, "code_sha256": "current-code"})
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["status"] == "recovery_running"
    assert snapshot["current_worker"]["status"] == "verified"
    assert snapshot["monitor_target"] == "current_worker"
    assert snapshot["status_record_applies"]["pipeline"] is False
    study.processes.pop(102)
    stale = monitor.inspect(study.directory, now=6800, previous=snapshot)
    assert stale["status"] != "recovery_running"


@pytest.mark.parametrize("status", ["running", "complete", "failed"])
def test_new_main_runner_supersedes_old_recovery(study, status):
    write_json(study.directory / "recovery_acceptance.json", {"status": status, "started": 900, "code_sha256": "current-code"})
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["monitor_target"] == "runner"
    assert snapshot["status"] == "unknown"


def test_completed_recovery_awaits_runner_instead_of_reusing_old_failure(study):
    study.processes.clear()
    write_json(study.directory / "runner_process.json", {"pid": 101, "started": 900,
        "stage": "all", "executable": study.runner.executable, "code_sha256": "old-code"})
    write_json(study.directory / "pipeline_status.json", {"stage": "model", "status": "failed", "updated": 950})
    write_json(study.directory / "recovery_acceptance.json", {"status": "complete", "started": 1000, "code_sha256": "current-code"})
    job = json.loads((study.directory / "current_job.json").read_text(encoding="utf-8"))
    job.update(status="complete", updated=1100)
    write_json(study.directory / "current_job.json", job)
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["status"] == "recovery_complete"
    assert snapshot["runner"]["status"] == "missing"
    assert snapshot["status_record_applies"]["pipeline"] is False
    assert not (study.directory / "monitor").exists()


def test_running_recovery_record_without_live_worker_is_not_trusted(study):
    study.processes.clear()
    write_json(study.directory / "runner_process.json", {"pid": 101, "started": 900,
        "stage": "all", "executable": study.runner.executable, "code_sha256": "old-code"})
    write_json(study.directory / "pipeline_status.json", {"stage": "model", "status": "failed", "updated": 950})
    write_json(study.directory / "recovery_acceptance.json", {"status": "running", "started": 1000, "code_sha256": "current-code"})
    snapshot = monitor.inspect(study.directory, now=5000)
    assert snapshot["status"] == "unknown"
    assert snapshot["status_record_applies"]["pipeline"] is False


def test_resumed_runner_and_original_workers_are_verified(study):
    study.runner.cmdline = lambda: [study.runner.executable,"-u","-m",monitor.RESUME_MODULE,"--stage","all"]
    snapshot = monitor.inspect(study.directory,now=5000)
    assert snapshot["runner"]["status"] == "verified"
    assert snapshot["current_worker"]["status"] == "verified"
    study.child.cpu += 2
    later = monitor.inspect(study.directory,now=6800,previous=snapshot)
    assert later["progress"]["counter_deltas"]["cpu_seconds"] == 2


@pytest.mark.parametrize("command", [
    ["python","-m","other_module",monitor.RESUME_MODULE],
    ["python","-c","pass",monitor.RESUME_MODULE],
    ["python","-m",monitor.RESUME_MODULE,"--worker","{}"],
])
def test_resume_name_as_foreign_argument_or_worker_is_rejected(study,command):
    study.runner.cmdline = lambda: command
    assert monitor.inspect(study.directory,now=5000)["runner"]["status"] == "identity_mismatch"


def test_reused_resume_pid_still_requires_creation_time(study):
    study.runner.cmdline = lambda: [study.runner.executable,"-m",monitor.RESUME_MODULE]
    study.runner.created = 4900
    assert monitor.inspect(study.directory,now=5000)["runner"]["status"] == "identity_mismatch"


def test_two_real_training_workers_and_logs_are_observed_without_parent_heartbeat(study):
    study.runner.cmdline=lambda:[study.runner.executable,"-m",monitor.PARALLEL_RESUME_MODULE]
    second=FakeProcess(103,created=1001,worker=True)
    for child in [study.child,second]:
        child.cmdline=lambda child=child:[child.executable,"-m",monitor.TRAINING_WORKER_MODULE,"--worker","{}"]
        study.processes[child.pid]=child
    study.runner.child_processes=[study.child,second]
    workers=[dict(pid=c.pid,started=1001,create_time=c.created,command=c.cmdline(),log=f"logs/train_{c.pid}.log")
             for c in [study.child,second]]
    for item in workers:
        (study.directory/item["log"]).write_text("fit started\n")
    write_json(study.directory/"parallel_training/status.json",dict(controller_pid=101,
        scientific_code_sha256="current-code",status="running",workers=workers))
    first=monitor.inspect(study.directory,now=5000)
    assert [w["status"] for w in first["parallel_workers"]]==["verified","verified"]
    assert "training_worker_102" in first["logs"] and "training_worker_103" in first["logs"]
    study.runner.cpu+=100
    study.runner.write+=10000
    later=monitor.inspect(study.directory,now=6800,previous=first)
    assert later["progress"]["status"]=="no_progress"
    second.cpu+=3
    active=monitor.inspect(study.directory,now=8600,previous=later)
    assert active["progress"]["counter_deltas"]["cpu_seconds"]==3


def test_parallel_worker_requires_exact_recorded_creation_time(study):
    study.child.cmdline=lambda:[study.child.executable,"-m",monitor.TRAINING_WORKER_MODULE,"--worker","{}"]
    record=dict(pid=102,started=1001,create_time=1001.5,executable=study.child.executable)
    assert monitor.inspect_runner(record,worker=True)["status"]=="identity_mismatch"


def test_memory_waiting_scheduler_heartbeat_is_not_fitting_progress(study):
    study.runner.child_processes=[]
    write_json(study.directory/"parallel_training/status.json",dict(controller_pid=101,
        scientific_code_sha256="current-code",status="running",workers=[],pending_streams=1))
    first=monitor.inspect(study.directory,now=5000)
    study.runner.cpu+=10
    study.runner.write+=10000
    later=monitor.inspect(study.directory,now=6800,previous=first)
    assert later["progress"]["status"]=="no_progress"
    assert later["progress"]["process_scope"]=="parallel_scheduler_without_training_worker"


def test_two_account_workers_use_the_same_identity_and_progress_checks(study):
    study.runner.cmdline=lambda:[study.runner.executable,"-m",monitor.PARALLEL_RESUME_MODULE]
    second=FakeProcess(103,created=1001,worker=True)
    for child in [study.child,second]:
        child.cmdline=lambda child=child:[child.executable,"-m",monitor.ACCOUNT_WORKER_MODULE,"--worker","{}"]
        study.processes[child.pid]=child
    study.runner.child_processes=[study.child,second]
    workers=[dict(pid=c.pid,started=1001,create_time=c.created,command=c.cmdline(),log=f"logs/account_{c.pid}.log")
             for c in [study.child,second]]
    for item in workers:
        (study.directory/item["log"]).write_text("account started\n")
    write_json(study.directory/"parallel_training/status.json",dict(controller_pid=101,
        scientific_code_sha256="current-code",status="running",role="account",workers=workers))
    first=monitor.inspect(study.directory,now=5000)
    assert [w["status"] for w in first["parallel_workers"]]==["verified","verified"]
    study.child.cpu+=5
    second.cpu+=7
    later=monitor.inspect(study.directory,now=6800,previous=first)
    assert later["progress"]["counter_deltas"]["cpu_seconds"]==12
