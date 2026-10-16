"""Run the preregistered four-strategy overfitting diagnosis in bounded processes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from paper_futu import overfit_four_core as core


@contextmanager
def controller_lock():
    core.OUT.mkdir(parents=True, exist_ok=True)
    with (core.OUT / "controller.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"1")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another overfit-four controller is already active") from error
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_job(store, job):
    if job["kind"] == "fold":
        function = lambda key: core.run_fold(job, key)
    elif job["kind"] == "fit":
        function = lambda key: core.run_fit(job, key)
    elif job["kind"] == "account":
        from paper_futu.overfit_four_account import replay
        function = lambda key: replay(job, key)
    elif job["kind"] == "signal_diagnostic":
        from paper_futu.overfit_four_diagnostics import signal_diagnostic
        function = lambda key: signal_diagnostic(job, key, store)
    elif job["kind"] == "account_diagnostic":
        from paper_futu.overfit_four_diagnostics import account_diagnostic
        function = lambda key: account_diagnostic(job, key, store)
    else:
        raise ValueError(f"Unknown job kind: {job['kind']}")
    return store.run(job, function)


def worker(queue, indices):
    store = core.Store()
    document = core.read(queue)
    jobs = document["jobs"]
    if document["implementation_sha256"] != core.scientific_hash():
        raise ValueError("Worker queue code binding changed")
    if document["jobs_sha256"] != core.frozen.identity(jobs):
        raise ValueError("Worker queue configuration changed")
    # A fresh process releases all Qlib quote caches after at most ten accounts.
    if not indices or any(i < 0 or i >= len(jobs) for i in indices) or len(indices) > 10 or len({jobs[i]["kind"] for i in indices}) != 1:
        raise ValueError("Invalid worker batch")
    if jobs[indices[0]]["kind"] == "account":
        core.init_qlib()
    for index in indices:
        if psutil.virtual_memory().available / 2**30 < 2.2:
            # Safe partial batch completion. The parent resumes uncompleted keys.
            break
        run_job(store, jobs[index])


def execute_queue(name, jobs, max_workers=2):
    store = core.Store()
    if not jobs:
        return
    queue = core.OUT / "queues" / f"{name}.json"
    core.save(queue, dict(implementation_sha256=core.scientific_hash(), jobs_sha256=core.frozen.identity(jobs), jobs=jobs))
    pending = [i for i, job in enumerate(jobs) if not store.record(job)]
    completed_at_start = len(jobs)-len(pending)
    children, retries = {}, {}
    started, last_progress = time.time(), time.time()
    no_memory_since = None
    failure = None
    status_path = core.OUT / "status.json"
    core.OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    verified = {i for i in range(len(jobs)) if i not in pending}

    def status(state):
        processes = []
        for child, indices, stdout, stderr in children.values():
            try:
                process = psutil.Process(child.pid)
                processes.append(dict(pid=child.pid, created=process.create_time(), command=process.cmdline(),
                    cpu_seconds=sum(process.cpu_times()[:2]), rss_gib=process.memory_info().rss/2**30,
                    io=process.io_counters()._asdict(), indices=indices))
            except psutil.Error:
                processes.append(dict(pid=child.pid, indices=indices, state="exiting"))
        core.save(status_path, dict(status=state, phase=name, controller_pid=os.getpid(),
            controller_created=psutil.Process().create_time(), started=started, updated=time.time(),
            total=len(jobs), completed=len(verified), completed_at_start=completed_at_start,
            active=processes, memory_available_gib=psutil.virtual_memory().available/2**30,
            disk_free_gib=__import__("shutil").disk_usage(core.ROOT).free/2**30,
            last_progress=last_progress, max_workers=max_workers, min_start_gib=core.MIN_START_GIB, error=failure))

    try:
        while pending or children:
            for pid, (child, indices, stdout, stderr) in list(children.items()):
                code = child.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                del children[pid]
                remaining = []
                for index in indices:
                    if store.record(jobs[index]):
                        verified.add(index)
                        last_progress = time.time()
                    else:
                        remaining.append(index)
                if code != 0:
                    log_meta = core.read(core.OUT / "logs" / f"{name}_{pid}.json")
                    error_text = Path(log_meta["stderr"]).read_text(encoding="utf-8", errors="replace")
                    memory_error = "GiB" in error_text and ("需要至少" in error_text or "MemoryError" in error_text)
                    if memory_error and all(retries.get(i, 0) == 0 for i in remaining):
                        for i in remaining:
                            retries[i] = 1
                        max_workers = 1
                        pending = remaining + pending
                    else:
                        failure = f"Worker {pid} exited {code}: {error_text[-2500:]}"
                else:
                    pending = remaining + pending
            if failure:
                if not children:
                    raise RuntimeError(failure)
                status("draining_after_error")
                time.sleep(5)
                continue
            memory = psutil.virtual_memory().available/2**30
            if not children and pending and memory < core.MIN_START_GIB:
                no_memory_since = no_memory_since or time.time()
                if time.time()-no_memory_since >= 1800:
                    raise MemoryError(f"Available RAM remained below {core.MIN_START_GIB:g} GiB for 30 minutes; release memory before resume")
            else:
                no_memory_since = None
            if pending and len(children) < max_workers and memory >= core.MIN_START_GIB:
                count = 10 if jobs[pending[0]]["kind"] in ("account", "account_diagnostic") else 1
                indices, pending = pending[:count], pending[count:]
                stdout_path = core.OUT / "logs" / f"{name}_{time.time_ns()}.stdout.log"
                stdout = stdout_path.open("w", encoding="utf-8")
                temporary_error = stdout_path.with_suffix(".stderr.log")
                stderr = temporary_error.open("w", encoding="utf-8")
                command = [sys.executable, "-u", "-m", "paper_futu.research_overfit_four", "--stage", "worker",
                           "--queue", str(queue), "--indices", ",".join(map(str, indices))]
                child_env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                             "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"}
                child = subprocess.Popen(command, cwd=core.ROOT, stdout=stdout, stderr=stderr, env=child_env,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                # Stable PID-based paths make a failed worker easy to inspect.
                # Keep the original handle open; Windows cannot rename it until exit.
                children[child.pid] = (child, indices, stdout, stderr)
                core.save(core.OUT / "logs" / f"{name}_{child.pid}.json",
                          dict(stdout=str(stdout_path), stderr=str(temporary_error), command=command,
                               pid=child.pid, created=psutil.Process(child.pid).create_time()))
                status("running")
                time.sleep(3)
            status("running" if children else "waiting_for_memory")
            if children or pending:
                time.sleep(5)
        status("phase_complete")
    except BaseException as error:
        failure = repr(error)
        status("failed")
        raise
    finally:
        for child, _, stdout, stderr in children.values():
            child.wait()
            stdout.close()
            stderr.close()


def stage_train():
    store = core.Store()
    folds = core.fold_jobs(store.protocol)
    execute_queue("validation", folds)
    selections = core.select_all(core.Store())
    fits = core.final_jobs(store.protocol, selections)
    core.save(core.OUT / "fit_jobs.json", fits)
    if len(folds)+len(fits) > 380:
        raise ValueError("Training budget exceeded")
    execute_queue("annual_fit", fits)
    return core.assemble_streams(core.Store(), fits)


def stage_backtest():
    store = core.Store()
    streams = core.read(core.OUT / "streams.json")
    jobs = core.account_jobs(store.protocol, streams)
    core.save(core.OUT / "account_jobs.json", jobs)
    execute_queue("accounts", jobs)


def stage_diagnose():
    from paper_futu.overfit_four_diagnostics import diagnostic_jobs, aggregate
    jobs = diagnostic_jobs()
    core.save(core.OUT / "diagnostic_jobs.json", jobs)
    execute_queue("diagnostics", jobs)
    aggregate(core.Store(), jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["audit", "train", "backtest", "diagnose", "report", "all", "status", "worker"], default="all")
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--indices")
    args = parser.parse_args()
    if args.stage == "status":
        print((core.OUT / "status.json").read_text(encoding="utf-8") if (core.OUT / "status.json").exists() else "Not started")
        return
    if args.stage == "worker":
        if not args.queue or not args.indices:
            parser.error("Worker needs a bound queue and indices")
        worker(args.queue, [int(x) for x in args.indices.split(",")])
        return
    with controller_lock():
        if args.stage in ("audit", "all"):
            core.save(core.OUT / "status.json", dict(status="running", phase="audit", controller_pid=os.getpid(), updated=time.time()))
            if args.stage == "all":
                # Audit loads feature panels and fitted models. Release that memory
                # before the controller waits or schedules its bounded workers.
                environment = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                               "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"}
                subprocess.run([sys.executable, "-u", "-c",
                    "from paper_futu.overfit_four_core import audit; audit()"],
                    cwd=core.ROOT, env=environment, check=True)
            else:
                core.audit()
            if args.stage == "audit":
                core.save(core.OUT / "status.json", dict(status="audit_complete", phase="audit",
                          controller_pid=os.getpid(), updated=time.time()))
        else:
            core.verify_protected(core.Store().protocol)
        if args.stage in ("train", "all"):
            stage_train()
        if args.stage in ("backtest", "all"):
            stage_backtest()
        if args.stage in ("diagnose", "all"):
            stage_diagnose()
        if args.stage in ("report", "all"):
            from paper_futu.overfit_four_diagnostics import report
            report(core.Store())
            core.verify_protected(core.Store().protocol, full_snapshot=True)
            core.save(core.OUT / "completion.json", dict(status="complete", finished=time.time(),
                      accounts=340, selected_champion=False, implementation_sha256=core.scientific_hash()))
            core.save(core.OUT / "status.json", dict(status="complete", updated=time.time(), controller_pid=os.getpid(),
                      report=str(core.OUT / "research_report.html")))


if __name__ == "__main__":
    main()
