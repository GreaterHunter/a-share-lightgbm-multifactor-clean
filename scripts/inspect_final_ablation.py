"""Read-only final-ablation health check; --record writes monitor snapshots only.

No research modules are imported, and no process is started, stopped or restarted.
The first observation is unknown. Two unchanged observations at least 30 minutes
apart (after the baseline) are required before suspecting a stall.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "reports/experiments/final_ablation_asof_20261013"
OBSERVATION_SECONDS = 30 * 60
START_TOLERANCE_SECONDS = 120
MODULE = "paper_futu.run_final_ablation"
RESUME_MODULE = "paper_futu.resume_full_factor_ablation"
PARALLEL_RESUME_MODULE = "paper_futu.resume_parallel_full_factor_ablation"
TRAINING_WORKER_MODULE = "paper_futu.parallel_full_factor_worker"
ACCOUNT_WORKER_MODULE = "paper_futu.parallel_account_worker"


def research_command(command, *, worker=False):
    """Recognize an executed module, not a module name passed as another app's argument."""
    try:
        module = command[command.index("-m")+1]
    except (ValueError, IndexError):
        return False
    allowed = (MODULE,TRAINING_WORKER_MODULE,ACCOUNT_WORKER_MODULE) if worker else (MODULE,RESUME_MODULE,PARALLEL_RESUME_MODULE)
    return module in allowed and ("--worker" in command) == worker


def read_json(path, errors):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        return value
    except (OSError, ValueError) as error:
        errors.append({"path": str(path), "error": str(error)})
        return {}


def current_code_hash(project_root=PROJECT_ROOT):
    """Match final_ablation_core.code_hash without importing its dependencies."""
    files = sorted((project_root / "paper_futu").glob("final_ablation_*.py"))
    files += [project_root / relative for relative in (
        "paper_futu/run_final_ablation.py", "src/factor_library.py",
        "src/build_enhanced_dataset.py", "src/compare_models.py",
        "src/portfolio_strategy.py", "src/runtime_budget.py",
    )]
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.relative_to(project_root).as_posix()] = digest.hexdigest()
    canonical = json.dumps(hashes, sort_keys=True, ensure_ascii=False,
                           default=str, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _same_path(actual, expected):
    return bool(actual and expected) and os.path.normcase(os.path.realpath(actual)) == os.path.normcase(os.path.realpath(expected))


def process_sample(process):
    result = {"pid": process.pid, "errors": {}}
    for field, method in (("create_time", "create_time"), ("command", "cmdline"),
                          ("cwd", "cwd"), ("executable", "exe"),
                          ("process_status", "status"), ("cpu", "cpu_times"),
                          ("io", "io_counters"), ("memory", "memory_info")):
        try:
            value = getattr(process, method)()
            if field == "cpu":
                result["cpu_seconds"] = value.user + value.system
            elif field == "io":
                for name in ("read_bytes", "write_bytes", "read_count", "write_count"):
                    result["io_" + name] = getattr(value, name)
            elif field == "memory":
                result["rss_bytes"] = value.rss
            else:
                result[field] = value
        except (psutil.Error, OSError, NotImplementedError) as error:
            result["errors"][field] = type(error).__name__ + ": " + str(error)
    return result


def inspect_runner(record, *, worker=False):
    result = {"record": record, "status": "unknown", "children": [],
              "children_status": "not_checked"}
    try:
        pid = int(record["pid"])
        if pid <= 0:
            raise ValueError("PID must be positive")
        process = psutil.Process(pid)
    except (KeyError, TypeError, ValueError) as error:
        result["reason"] = "运行记录缺少有效 PID: " + str(error)
        return result
    except psutil.NoSuchProcess:
        result.update(status="missing", reason="记录的 PID 已不存在")
        return result
    except psutil.AccessDenied:
        result.update(status="access_denied", reason="无权读取记录的 PID")
        return result
    result.update(process_sample(process))
    errors = result["errors"]
    if any(value.startswith("NoSuchProcess") for value in errors.values()):
        result.update(status="missing", reason="进程在采样过程中退出")
        return result
    try:
        delta = result["create_time"] - float(record["started"])
    except (KeyError, TypeError, ValueError):
        delta = None
    checks = {
        "command": research_command(result["command"], worker=worker) if "command" in result else None,
        "create_time": (abs(result["create_time"]-float(record["create_time"])) <= .01
                        if record.get("create_time") is not None and "create_time" in result
                        else abs(delta) <= START_TOLERANCE_SECONDS if delta is not None else None),
        "cwd": _same_path(result.get("cwd"), PROJECT_ROOT) if "cwd" in result else None,
        "executable": _same_path(result.get("executable"), record.get("executable")) if record.get("executable") and "executable" in result else None,
    }
    result.update(identity_checks=checks, create_time_delta_seconds=delta,
                  create_time_tolerance_seconds=START_TOLERANCE_SECONDS,
                  expected_cwd=str(PROJECT_ROOT))
    if False in checks.values() or result.get("process_status") == psutil.STATUS_ZOMBIE:
        result.update(status="identity_mismatch", reason="PID、启动时间、命令、工作目录或解释器不匹配；不可当作当前研究进程")
        return result
    if None in checks.values():
        denied = any(value.startswith("AccessDenied") for value in errors.values())
        result.update(status="access_denied" if denied else "unknown", reason="进程身份信息不足，无法确认当前运行")
        return result
    result["status"] = "verified"
    try:
        result["children"] = [process_sample(child) for child in process.children(recursive=True)]
        result["children_status"] = "ok"
        if any(any(value.startswith("NoSuchProcess") for value in child["errors"].values()) for child in result["children"]):
            result["children_status"] = "changed_during_snapshot"
        if not process.is_running():
            result.update(status="missing", reason="进程在采样过程中退出")
    except (psutil.Error, OSError) as error:
        result["children_status"] = type(error).__name__
        result["children_error"] = str(error)
    return result


def task_summary(directory, code):
    result = {"code_sha256": code, "counts": {}, "kind_status_counts": {},
              "total": 0, "other_code_count": 0, "unreadable_count": 0, "errors": []}
    signatures = []
    try:
        paths = sorted((directory / "tasks").glob("*.json"))
    except OSError as error:
        result["errors"].append(str(error))
        paths = []
    for path in paths:
        errors = []
        record = read_json(path, errors)
        if errors:
            result["unreadable_count"] += 1
            result["errors"].extend(errors[:1])
            continue
        if not code or record.get("code_sha256") != code:
            result["other_code_count"] += 1
            continue
        status = str(record.get("status", "unknown"))
        kind_status = str(record.get("kind", "unknown")) + ":" + status
        result["counts"][status] = result["counts"].get(status, 0) + 1
        result["kind_status_counts"][kind_status] = result["kind_status_counts"].get(kind_status, 0) + 1
        result["total"] += 1
        try:
            stat = path.stat()
            signatures.append([path.name, status, stat.st_size, stat.st_mtime_ns])
        except OSError as error:
            result["errors"].append({"path": str(path), "error": str(error)})
    result["fingerprint"] = hashlib.sha256(json.dumps(signatures).encode()).hexdigest()
    return result


def log_sample(value, directory):
    if not value:
        return {"status": "not_recorded"}
    path = Path(value)
    if not path.is_absolute():
        path = directory / path
    try:
        stat = path.stat()
        return {"path": str(path), "status": "ok", "size_bytes": stat.st_size,
                "mtime": stat.st_mtime, "mtime_local": datetime.fromtimestamp(stat.st_mtime, timezone.utc).astimezone().isoformat()}
    except OSError as error:
        return {"path": str(path), "status": "missing" if isinstance(error, FileNotFoundError) else "unavailable", "error": str(error)}


def _baseline(snapshot):
    runner = snapshot[snapshot["monitor_target"]]
    processes = {}
    scope = "research_workers"
    children_status = runner["children_status"]
    if runner["status"] == "verified":
        if snapshot["monitor_target"] == "current_worker":
            observed_processes = [runner]
        else:
            # dispatch rewrites current_job.json every 30 seconds. Those parent
            # CPU/IO heartbeats (and helpers such as conhost) are not training.
            observed_processes = [child for child in runner["children"]
                                  if research_command(child.get("command", []), worker=True)]
            if any("command" not in child for child in runner["children"]):
                children_status = "incomplete"
            if not observed_processes:
                parallel=snapshot.get("parallel_training",{})
                if parallel.get("controller_pid")==runner["record"].get("pid") and parallel.get("status")=="running":
                    # Scheduler status/telemetry writes are not fitting progress even while memory-gated.
                    observed_processes=[]
                    scope="parallel_scheduler_without_training_worker"
                else:
                    observed_processes = [runner]
                    scope = "runner_without_worker"
                    if snapshot["current_job"].get("status") == "running" and _applies_to_runner(snapshot["current_job"], runner["record"]):
                        children_status = "expected_worker_not_observed"
        for process in observed_processes:
            if process is not runner and (not _same_path(process.get("cwd"), PROJECT_ROOT) or
                    not _same_path(process.get("executable"), runner["record"].get("executable"))):
                children_status = "incomplete"
            if process.get("create_time") is not None:
                key = f"{process['pid']}:{process['create_time']}"
                processes[key] = {name: process.get(name) for name in (
                    "cpu_seconds", "io_read_bytes", "io_write_bytes", "io_read_count", "io_write_count")}
            else:
                children_status = "incomplete"
    job = snapshot["current_job"]
    return {"observed_at": snapshot["observed_at"],
            "runner": [runner.get("pid"), runner.get("create_time"), runner["record"].get("started"), snapshot["code"]["current_sha256"]],
            "monitor_target": snapshot["monitor_target"],
            "verified": runner["status"] == "verified", "processes": processes,
            "children_status": children_status, "process_scope": scope,
            "tasks": [snapshot["tasks"]["fingerprint"], snapshot["tasks"]["unreadable_count"]],
            "pipeline": [snapshot["pipeline"].get("stage"), snapshot["pipeline"].get("status")],
            "job": [job.get(name) for name in ("worker_pid", "started", "status", "job")],
            "logs": {key: [value.get("path"), value.get("size_bytes"), value.get("mtime")] for key, value in snapshot["logs"].items()}}


def compare_progress(snapshot, previous):
    current = _baseline(snapshot)
    old_progress = previous.get("progress", {}) if previous else {}
    old = old_progress.get("baseline")
    result = {"status": "unknown", "signals": [], "no_progress_intervals": 0,
              "minimum_interval_seconds": OBSERVATION_SECONDS, "baseline": current,
              "process_scope": current["process_scope"],
              "reason": "首次观察或缺少可比较的历史快照"}
    if not old or not current["verified"] or not old.get("verified") or current["runner"] != old.get("runner") or current["monitor_target"] != old.get("monitor_target"):
        return result
    elapsed = current["observed_at"] - old["observed_at"]
    result["interval_seconds"] = elapsed
    result["since_last_snapshot_seconds"] = current["observed_at"] - previous["observed_at"]
    signals = result["signals"]
    if current["processes"].keys() != old["processes"].keys():
        signals.append("子进程集合发生变化")
    complete_metrics = current["children_status"] == old["children_status"] == "ok"
    deltas = {"cpu_seconds": 0.0, "io_read_bytes": 0, "io_write_bytes": 0,
              "io_read_count": 0, "io_write_count": 0}
    for key in current["processes"].keys() & old["processes"].keys():
        for name in deltas:
            before, after = old["processes"][key][name], current["processes"][key][name]
            if before is None or after is None or after < before:
                complete_metrics = False
            else:
                deltas[name] += after - before
    result["counter_deltas"] = deltas
    if deltas["cpu_seconds"] > 0:
        signals.append("研究 worker（无 worker 时为主进程）CPU 累计时间增加")
    if any(deltas[name] > 0 for name in deltas if name.startswith("io_")):
        signals.append("研究 worker（无 worker 时为主进程）读写计数增加")
    for key, label in (("tasks", "当前代码任务记录变化"), ("pipeline", "阶段状态变化"),
                       ("job", "当前任务变化"), ("logs", "日志大小或修改时间变化")):
        if current[key] != old[key]:
            signals.append(label)
    if signals:
        result.update(status="progress", reason="观察到进展，继续运行观察")
    elif elapsed < 0:
        result["reason"] = "观察时钟倒退，重新建立基线"
    elif not complete_metrics or current["tasks"][1] or old["tasks"][1] or snapshot["source_errors"] or snapshot["code"]["error"]:
        result["reason"] = "采样信息不完整，不能据此判断停滞"
    elif elapsed < OBSERVATION_SECONDS:
        result.update(status="interval_too_short", baseline=old,
                      no_progress_intervals=old_progress.get("no_progress_intervals", 0),
                      reason="观察间隔不足 30 分钟，不累计无进展次数")
    else:
        result.update(status="no_progress", no_progress_intervals=old_progress.get("no_progress_intervals", 0) + 1,
                      reason="完整观察区间内 CPU、读写、任务和日志均无进展")
    return result


def _applies_to_runner(state, runner_record):
    if state.get("code_sha256") and state["code_sha256"] != runner_record.get("code_sha256"):
        return False
    updated = state.get("updated", state.get("started"))
    started = runner_record.get("started")
    return updated is None or started is None or updated >= started


def inspect(experiment_dir=DEFAULT_EXPERIMENT, *, now=None, previous=None):
    directory = Path(experiment_dir).resolve()
    now = time.time() if now is None else now
    errors = []
    pipeline = read_json(directory / "pipeline_status.json", errors)
    record = read_json(directory / "runner_process.json", errors)
    current_job = read_json(directory / "current_job.json", errors)
    recovery_path = directory / "recovery_acceptance.json"
    recovery = read_json(recovery_path, errors) if recovery_path.exists() else {}
    runner = inspect_runner(record)
    code_error = None
    try:
        code = current_code_hash()
    except OSError as error:
        code, code_error = None, str(error)
    command = current_job.get("command") or []
    worker_record = {"pid": current_job.get("worker_pid"), "started": current_job.get("started"),
                     "executable": command[0] if command else record.get("executable"),
                     "code_sha256": current_job.get("code_sha256")}
    current_worker = inspect_runner(worker_record, worker=True) if worker_record["pid"] else {"status": "not_recorded", "record": worker_record, "children": [], "children_status": "not_checked"}
    parallel_path=directory/"parallel_training/status.json"
    parallel=read_json(parallel_path,errors) if parallel_path.exists() else {}
    parallel_workers=[]
    if parallel.get("controller_pid")==record.get("pid") and parallel.get("scientific_code_sha256")==code and parallel.get("status")=="running":
        for item in parallel.get("workers",[]):
            child_record={"pid":item["pid"],"started":item["started"],"create_time":item["create_time"],
                          "executable":item["command"][0],"code_sha256":code}
            parallel_workers.append(inspect_runner(child_record,worker=True))
    recovery_is_current = bool(code and recovery.get("code_sha256") == code and
        recovery.get("started", 0) >= record.get("started", 0))
    recovery_active = recovery_is_current and recovery.get("status") == "running" and current_job.get("status") == "running" and current_job.get("code_sha256") == code and current_worker["status"] == "verified" and current_job.get("started", 0) >= recovery.get("started", 0)
    logs = {"current_job": log_sample(current_job.get("log"), directory),
            "runner_stdout": log_sample(record.get("stdout_log") or "runner_stdout.log", directory),
            "runner_stderr": log_sample(record.get("stderr_log") or "runner_stderr.log", directory)}
    if parallel_workers:
        for item in parallel.get("workers",[]):
            logs[f"training_worker_{item['pid']}"]=log_sample(item.get("log"),directory)
    memory = psutil.virtual_memory()
    system = {"memory_total_bytes": memory.total, "memory_available_bytes": memory.available,
              "memory_percent": memory.percent}
    try:
        disk = psutil.disk_usage("D:\\")
        system["d_disk"] = {"path": "D:\\", "status": "ok", "free_bytes": disk.free,
                            "total_bytes": disk.total, "used_percent": disk.percent}
    except (OSError, SystemError) as error:
        system["d_disk"] = {"path": "D:\\", "status": "unavailable", "error": type(error).__name__ + ": " + str(error)}
    snapshot = {"schema_version": 1, "observed_at": now,
                "observed_at_local": datetime.fromtimestamp(now, timezone.utc).astimezone().isoformat(),
                "experiment_dir": str(directory), "runner": runner, "pipeline": pipeline,
                "current_job": current_job, "current_worker": current_worker,
                "parallel_training":parallel,"parallel_workers":parallel_workers,
                "recovery_acceptance": recovery, "monitor_target": "current_worker" if recovery_active else "runner",
                "logs": logs, "system": system,
                "code": {"current_sha256": code, "runner_sha256": record.get("code_sha256"),
                         "matches_runner": code == record.get("code_sha256") if code else None, "error": code_error},
                "tasks": task_summary(directory, code), "source_errors": errors}
    if previous is None and (directory / "monitor/latest.json").exists():
        history_errors = []
        previous = read_json(directory / "monitor/latest.json", history_errors)
        snapshot["history_errors"] = history_errors
    snapshot["progress"] = compare_progress(snapshot, previous)
    progress = snapshot["progress"]
    active_record = worker_record if recovery_active else recovery if recovery_is_current else record
    pipeline_applies = _applies_to_runner(pipeline, active_record)
    job_applies = _applies_to_runner(current_job, active_record)
    snapshot["status_record_applies"] = {"pipeline": pipeline_applies, "current_job": job_applies}
    if recovery_active:
        status = "stalled_suspected" if progress["no_progress_intervals"] >= 2 else "recovery_running"
        summary = "恢复验收正在运行；当前代码与存活 worker 已核对，旧全量失败记录不属于本次恢复" if status == "recovery_running" else "恢复验收连续两个完整 30 分钟观察区间均无进展，疑似停滞；仅报告，不重启"
    elif recovery_is_current and (recovery.get("status") == "failed" or job_applies and current_job.get("code_sha256") == code and current_job.get("status") == "failed"):
        status, summary = "failed", "本次恢复验收已有明确失败记录；需要查看失败日志和原因"
    elif recovery_is_current and recovery.get("status") == "complete":
        status, summary = "recovery_complete", "当前代码的恢复验收已完成，等待新全量运行进程；旧失败记录属于上一轮，巡检不会自动启动"
    elif recovery_is_current and recovery.get("status") == "running":
        status, summary = "unknown", "恢复验收记录为运行中，但尚未核实到本次存活 worker；继续核对，旧失败记录属于上一轮"
    elif (pipeline_applies and pipeline.get("status") == "failed") or (job_applies and current_job.get("status") == "failed"):
        status, summary = "failed", "研究运行已有明确失败记录；需要查看失败日志和原因"
    elif pipeline_applies and pipeline.get("status") == "complete" and pipeline.get("stage") == record.get("stage"):
        status, summary = "complete", "记录显示所请求研究阶段已完成"
    elif runner["status"] == "missing":
        status, summary = "runner_missing", "研究运行进程已不存在，未自动重启"
    elif runner["status"] == "identity_mismatch":
        status, summary = "identity_mismatch", runner["reason"]
    elif runner["status"] != "verified":
        status, summary = "unknown", runner.get("reason", "无法确认研究运行进程")
    elif progress["no_progress_intervals"] >= 2:
        status, summary = "stalled_suspected", "连续两个完整 30 分钟观察区间均无进展，疑似停滞；仅报告，不重启"
    elif progress["status"] == "unknown":
        status, summary = "unknown", progress["reason"]
    elif progress["status"] == "no_progress":
        status, summary = "no_progress", "一个完整观察区间未见进展，继续观察"
    else:
        status, summary = "running", progress["reason"]
    snapshot.update(status=status, summary=summary,
                    action="只读巡检；记录模式仅写入独立 monitor 目录，绝不重启训练或操作 Futu")
    return snapshot


def _atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def record_snapshot(snapshot):
    directory = Path(snapshot["experiment_dir"]) / "monitor"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(snapshot["observed_at"]).strftime("%Y%m%d_%H%M%S")
    history = directory / f"snapshot_{stamp}_{uuid.uuid4().hex[:8]}.json"
    latest = directory / "latest.json"
    snapshot["recorded_snapshot"] = str(history)
    snapshot["latest_snapshot"] = str(latest)
    _atomic_json(history, snapshot)
    _atomic_json(latest, snapshot)
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT,
                        help="实验目录，默认使用 final_ablation_asof_20261013")
    parser.add_argument("--record", action="store_true", help="保存本次快照与最新观察状态到实验目录下的 monitor；保留历史")
    args = parser.parse_args(argv)
    snapshot = inspect(args.experiment_dir)
    if args.record:
        record_snapshot(snapshot)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
