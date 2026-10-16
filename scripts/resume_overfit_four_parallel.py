"""Promptly fill two worker slots without changing scientific jobs or cached keys."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from paper_futu import overfit_four_core as core
from paper_futu.research_overfit_four import controller_lock


def binding():
    path = Path(__file__).resolve()
    return dict(path=str(path), sha256=core.frozen.digest(path),
                original_implementation_sha256=core.scientific_hash(),
                max_workers=2, min_start_gib=core.MIN_START_GIB, threads=4,
                compatibility="Scheduling only; original worker, jobs, task keys, model parameters and execution engine unchanged")


def is_memory_failure(error, records=()):
    """Prefer UTF-8 task records; Windows stderr may have a different code page."""
    messages = [error]+[record.get("error", "") for record in records]
    return any("MemoryError" in message or
               ("GiB" in message and any(marker in message for marker in
                ("需要至少", "require_memory(", "runtime_budget.py")))
               for message in messages)


def task_failure(store, job):
    path = core.OUT/"tasks"/(store.key(job)+".json")
    if path.exists():
        record = core.read(path)
        if record.get("status") == "failed":
            return record
    return None


def release_controller_buffers():
    """Return freed Arrow assembly buffers before applying the launch gate."""
    import pyarrow as pa
    gc.collect()
    pa.default_memory_pool().release_unused()


def execute_queue(name, jobs):
    store = core.Store()
    scheduler = binding()
    keys = [store.key(job) for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate jobs would violate exclusive worker ownership")
    queue = core.OUT/"queues"/f"{name}.json"
    core.save(queue, dict(implementation_sha256=core.scientific_hash(), jobs_sha256=core.frozen.identity(jobs), jobs=jobs))
    pending = [i for i, job in enumerate(jobs) if not store.record(job)]
    finished = set(range(len(jobs)))-set(pending)
    initial = len(finished)
    if not pending:
        return
    children = {}
    retry_path = core.OUT/"memory_retry_budget.json"
    retries = core.read(retry_path) if retry_path.exists() else {}
    started = progress = time.time()
    memory_wait = None
    failure = None
    maximum = 2
    peak_parallel = 0

    def reserve_memory_retry(indices, records):
        nonlocal maximum
        for index, record in zip(indices, records):
            key = keys[index]
            prior = retries.get(key)
            if prior and record and record.get("started", 0) > prior["original_failure_started"]:
                raise RuntimeError(f"Memory retry already exhausted for {key}; release RAM before manual recovery")
            if not prior:
                retries[key] = dict(phase=name, attempts_allowed=1, recorded=time.time(),
                    original_failure_started=record.get("started", 0) if record else 0,
                    reason=(record or {}).get("error", "Worker memory failure before task record"))
        core.save(retry_path, retries)
        maximum = 1

    # Recognize the first failed concurrent attempt even after controller restart.
    # Never reset the one-retry budget by restarting the scheduler.
    for index in pending:
        previous = task_failure(store, jobs[index])
        if previous and is_memory_failure("", [previous]):
            reserve_memory_retry([index], [previous])

    def record(state):
        active = []
        for child, indices, _, _, log in children.values():
            try:
                process = psutil.Process(child.pid)
                active.append(dict(pid=child.pid, created=process.create_time(), command=process.cmdline(),
                    indices=indices, cpu_seconds=sum(process.cpu_times()[:2]),
                    rss_gib=process.memory_info().rss/2**30, io=process.io_counters()._asdict(), log=log))
            except psutil.Error:
                active.append(dict(pid=child.pid, indices=indices, state="exiting"))
        core.save(core.OUT/"status.json", dict(status=state, phase=name, controller_pid=os.getpid(),
            controller_created=psutil.Process().create_time(), started=started, updated=time.time(),
            completed=len(finished), completed_at_start=initial, total=len(jobs), active=active,
            memory_available_gib=psutil.virtual_memory().available/2**30,
            disk_free_gib=__import__("shutil").disk_usage(core.ROOT).free/2**30,
            max_workers=maximum, peak_parallel=peak_parallel, min_start_gib=core.MIN_START_GIB,
            last_progress=progress, error=failure, scheduler=scheduler,
            memory_fallback=maximum == 1, memory_retry_keys=[key for key in keys if key in retries]))

    try:
        while pending or children:
            for pid, (child, indices, stdout, stderr, log) in list(children.items()):
                exit_code = child.poll()
                if exit_code is None:
                    continue
                stdout.close()
                stderr.close()
                del children[pid]
                remaining = []
                for index in indices:
                    if store.record(jobs[index]):
                        finished.add(index)
                        progress = time.time()
                    else:
                        remaining.append(index)
                if exit_code:
                    error = Path(log["stderr"]).read_text(encoding="utf-8", errors="replace")
                    records = [task_failure(store, jobs[index]) for index in remaining]
                    is_memory = is_memory_failure(error, [record for record in records if record])
                    if remaining and is_memory and all(keys[index] not in retries for index in remaining):
                        reserve_memory_retry(remaining, records)
                        maximum = 1
                        pending = remaining+pending
                    else:
                        failure = f"Worker {pid} exited {exit_code}: {error[-2500:]}"
                else:
                    pending = remaining+pending
            if failure:
                record("draining_after_error")
                if not children:
                    raise RuntimeError(failure)
                time.sleep(1)
                continue
            if pending and not children and psutil.virtual_memory().available/2**30 < core.MIN_START_GIB:
                memory_wait = memory_wait or time.time()
                if time.time()-memory_wait >= 1800:
                    raise MemoryError(f"Available RAM remained below {core.MIN_START_GIB:g} GiB for 30 minutes")
            else:
                memory_wait = None
            # Fill the free slots before waiting; the old 3+5 second delay was
            # longer than many fits and prevented their lifetimes overlapping.
            while pending and len(children) < maximum and psutil.virtual_memory().available/2**30 >= core.MIN_START_GIB:
                count = 10 if jobs[pending[0]]["kind"] in ("account", "account_diagnostic") else 1
                indices, pending = pending[:count], pending[count:]
                occupied = {i for _, owned, _, _, _ in children.values() for i in owned}
                if occupied.intersection(indices):
                    raise ValueError("Overlapping worker assignments")
                stamp = f"parallel_{name}_{time.time_ns()}"
                stdout_path, stderr_path = core.OUT/"logs"/(stamp+".stdout.log"), core.OUT/"logs"/(stamp+".stderr.log")
                stdout, stderr = stdout_path.open("w",encoding="utf-8"), stderr_path.open("w",encoding="utf-8")
                command = [sys.executable,"-u","-m","paper_futu.research_overfit_four","--stage","worker",
                           "--queue",str(queue),"--indices",",".join(map(str,indices))]
                environment = {**os.environ,"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4",
                               "MKL_NUM_THREADS":"4","NUMEXPR_NUM_THREADS":"4",
                               "PYTHONIOENCODING":"utf-8","PYTHONUTF8":"1"}
                child = subprocess.Popen(command,cwd=core.ROOT,stdout=stdout,stderr=stderr,env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                log = dict(stdout=str(stdout_path),stderr=str(stderr_path),command=command)
                children[child.pid] = (child,indices,stdout,stderr,log)
                core.save(core.OUT/"logs"/f"{name}_{child.pid}.json",dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),**log))
                peak_parallel = max(peak_parallel,len(children))
                if len(children)>2:
                    raise ValueError("Worker budget exceeded")
                record("running")
            record("running" if children else "waiting_for_memory")
            if pending or children:
                time.sleep(.5)
        record("phase_complete")
    except BaseException as error:
        failure=repr(error)
        record("failed")
        raise
    finally:
        for child,_,stdout,stderr,_ in children.values():
            child.wait()
            stdout.close()
            stderr.close()


def main():
    from paper_futu.overfit_four_diagnostics import diagnostic_jobs,aggregate,report
    with controller_lock():
        store=core.Store()
        core.verify_protected(store.protocol)
        prior_binding=core.OUT/"parallel_scheduler_binding.json"
        if prior_binding.exists():
            previous=core.read(prior_binding)
            core.save(core.OUT/"runtime_history"/f"scheduler_{time.time_ns()}.json",previous)
        core.save(core.OUT/"parallel_scheduler_binding.json",dict(created=time.time(),**binding()))
        execute_queue("validation",core.fold_jobs(store.protocol))
        # Recreate only the identical selection manifests; cache identities and
        # fitted models remain under the original verified scientific binding.
        choices=core.select_all(store)
        fits=core.final_jobs(store.protocol,choices)
        existing=core.read(core.OUT/"fit_jobs.json")
        if core.frozen.identity(existing)!=core.frozen.identity(fits):
            raise ValueError("Scheduling change unexpectedly altered fitted-model jobs")
        execute_queue("annual_fit",fits)
        streams=core.assemble_streams(store,fits)
        before=psutil.virtual_memory().available/2**30
        release_controller_buffers()
        core.save(core.OUT/"controller_buffer_release.json",dict(recorded=time.time(),
            before_available_gib=before,after_available_gib=psutil.virtual_memory().available/2**30))
        accounts=core.account_jobs(store.protocol,streams)
        core.save(core.OUT/"account_jobs.json",accounts)
        execute_queue("accounts",accounts)
        diagnostics=diagnostic_jobs()
        core.save(core.OUT/"diagnostic_jobs.json",diagnostics)
        execute_queue("diagnostics",diagnostics)
        aggregate(store,diagnostics)
        report(store)
        core.verify_protected(store.protocol,full_snapshot=True)
        manifest=core.read(core.OUT/"report_manifest.json")
        manifest["scheduler"]=binding()
        core.save(core.OUT/"report_manifest.json",manifest)
        core.save(core.OUT/"completion.json",dict(status="complete",finished=time.time(),accounts=340,
            selected_champion=False,implementation_sha256=core.scientific_hash(),scheduler=binding()))
        core.save(core.OUT/"status.json",dict(status="complete",updated=time.time(),controller_pid=os.getpid(),
            scheduler=binding(),report=str(core.OUT/"research_report.html")))


if __name__=="__main__":
    main()
