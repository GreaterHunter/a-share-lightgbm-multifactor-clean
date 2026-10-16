"""The parallel resume controller fills slots without overlapping scientific jobs."""
from collections import namedtuple
from types import SimpleNamespace

import pytest

from scripts import resume_overfit_four_parallel as parallel


def test_two_slots_start_before_first_wait_and_own_disjoint_jobs(tmp_path, monkeypatch):
    completed, started, waits = set(), [], []

    class FakeStore:
        def key(self, job):
            return str(job["index"])

        def record(self, job):
            return {"complete": True} if job["index"] in completed else None

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 500+len(started)
            self.indices = [int(x) for x in command[-1].split(",")]
            started.append(self)

        def poll(self):
            if len(started) < 2:
                return None
            completed.update(self.indices)
            return 0

        def wait(self):
            return 0

    class Probe:
        def __init__(self, pid=None):
            self.pid = pid or 100

        def create_time(self):
            return 123.

        def cmdline(self):
            return ["test-worker"]

        def cpu_times(self):
            return (1., 1.)

        def memory_info(self):
            return SimpleNamespace(rss=1000)

        def io_counters(self):
            return namedtuple("IO", "read_bytes write_bytes")(100, 100)

    monkeypatch.setattr(parallel.core, "OUT", tmp_path)
    monkeypatch.setattr(parallel.core, "Store", FakeStore)
    monkeypatch.setattr(parallel, "binding", lambda: {"sha256": "test-scheduler"})
    monkeypatch.setattr(parallel.psutil, "Process", Probe)
    monkeypatch.setattr(parallel.psutil, "virtual_memory", lambda: SimpleNamespace(available=4*2**30))
    monkeypatch.setattr(parallel.subprocess, "Popen", Child)
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: waits.append(len(started)))
    (tmp_path/"logs").mkdir()
    parallel.execute_queue("unit", [dict(kind="fit", index=0), dict(kind="fit", index=1)])
    assert len(started) == 2 and waits[0] == 2
    assert set(started[0].indices).isdisjoint(started[1].indices)
    assert completed == {0, 1}
    state = parallel.core.read(tmp_path/"status.json")
    assert state["completed"] == 2 and state["peak_parallel"] == 2


def test_duplicate_jobs_are_rejected_before_launch(monkeypatch):
    class FakeStore:
        def key(self, job):
            return "same-key"
    monkeypatch.setattr(parallel.core, "Store", FakeStore)
    with pytest.raises(ValueError, match="Duplicate jobs"):
        parallel.execute_queue("unit", [dict(kind="fit"), dict(kind="fit")])


def test_memory_detection_uses_structured_record_despite_mojibake():
    assert parallel.is_memory_failure("RuntimeError: ???? 2.0 GiB 1.97 GiB", [
        dict(error="RuntimeError('拟合需要至少2.0 GiB可用物理内存，目前仅1.97 GiB')")])
    assert parallel.is_memory_failure("runtime_budget.py require_memory(2.0) RuntimeError: ?? GiB")
    assert not parallel.is_memory_failure("ValueError: model mismatch with 4 GiB available")


def test_memory_retry_budget_survives_restart_and_rejects_second_failure(tmp_path, monkeypatch):
    class FakeStore:
        def key(self, job):
            return "memory-job"

        def record(self, job):
            return None

    monkeypatch.setattr(parallel.core, "OUT", tmp_path)
    monkeypatch.setattr(parallel.core, "Store", FakeStore)
    monkeypatch.setattr(parallel, "binding", lambda: {})
    parallel.core.save(tmp_path/"tasks/memory-job.json", dict(status="failed", started=200.,
        error="RuntimeError('拟合需要至少2.0 GiB，目前仅1.97 GiB')"))
    parallel.core.save(tmp_path/"memory_retry_budget.json", {
        "memory-job": dict(original_failure_started=100., attempts_allowed=1)})
    with pytest.raises(RuntimeError, match="Memory retry already exhausted"):
        parallel.execute_queue("unit", [dict(kind="fit")])


def test_controller_releases_only_unused_assembly_buffers(monkeypatch):
    import pyarrow as pa
    calls = []
    monkeypatch.setattr(parallel.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(pa, "default_memory_pool", lambda: SimpleNamespace(
        release_unused=lambda: calls.append("arrow-unused")))
    parallel.release_controller_buffers()
    assert calls == ["gc", "arrow-unused"]
