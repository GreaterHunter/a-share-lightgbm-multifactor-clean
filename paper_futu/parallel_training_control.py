"""Process-local scheduling and OS locks; scientific functions and task keys are unchanged."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from paper_futu.final_ablation_core import REPORT, SCHEDULES, Store, digest, identity, read_json, save_json
from paper_futu import resume_full_factor_ablation as scope

WORKER_MODULE = "paper_futu.parallel_full_factor_worker"
ACCOUNT_WORKER_MODULE = "paper_futu.parallel_account_worker"
CONTROL_ROOT = REPORT/"parallel_training"
MAX_WORKERS = 2
THREADS = 4
MIN_AVAILABLE_GIB = 2.0
SINGLE_START_GIB = 2.75
DUAL_START_GIB = 3.25
EMERGENCY_GIB = 1.5
MEMORY_EXIT_CODE = 75


@contextmanager
def file_lock(path, timeout=None):
    """Kernel-held byte/flock lock; process death releases it without stale-PID guessing."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    started = time.monotonic()
    acquired = False
    try:
        while not acquired:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                acquired = True
            except OSError as error:
                if error.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK) and getattr(error,"winerror",None) not in (33,36):
                    raise
                if timeout is not None and time.monotonic()-started >= timeout:
                    raise TimeoutError(f"Lock still held: {path}") from error
                time.sleep(.1)
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()


def stream_id(job):
    return identity({k:job[k] for k in ("spec","horizon","policy","months","seed")})[:24]


def unique_jobs(jobs):
    result = {}
    for job in jobs:
        scope.validate_spec(job["spec"])
        if job["horizon"] not in SCHEDULES or job["months"] not in [12,6,3] or job["policy"] not in ["fixed","tuned"] or job["seed"] not in [42,2024,2025]:
            raise ValueError("Training job outside the existing study protocol")
        result.setdefault(stream_id(job),job)
    return list(result.values())


def make_job(spec,horizon,policy="fixed",months=12,seed=42):
    return dict(spec=spec,horizon=horizon,policy=policy,months=months,seed=seed)


def cadence_jobs(candidates):
    jobs = [make_job(scope.A.record(),20,months=m) for m in [12,6,3]]
    for config in candidates:
        jobs += [make_job(config["spec"],config["horizon"],config["policy"],m) for m in [12,6,3]]
    return unique_jobs(jobs)


def seed_jobs(candidates, order):
    jobs = []
    for candidate_id in order[:3]:
        config = next(c for c in candidates if c["candidate_id"] == candidate_id)
        jobs += [make_job(config["spec"],config["horizon"],config["policy"],config["months"],s) for s in [2024,2025]]
    return unique_jobs(jobs)


def stream_cached(job,store):
    from paper_futu.run_final_ablation import prediction_path,spec_from
    path = prediction_path(spec_from(job["spec"]),job["horizon"],job["policy"],job["months"],job["seed"])
    if not path.exists() or not path.with_suffix(".json").exists():
        return False
    meta = read_json(path.with_suffix(".json"))
    if digest(path) != meta["prediction_sha256"]:
        raise ValueError(f"Existing stitched predictions changed: {path}")
    # Old-code manifests are not promoted. Every serving artifact must verify under the original key.
    return all(store.cached("serve",{k:v for k,v in period.items() if k != "path"}) is not None for period in meta["periods"])


def is_memory_failure(error):
    return (isinstance(error,MemoryError) or
            isinstance(error,RuntimeError) and "GiB可用物理内存" in str(error) and "需要至少" in str(error))


def retry_allowed(exit_code,attempt):
    return exit_code == MEMORY_EXIT_CODE and attempt < 1


def launch_allowed(available_gib,active_count,single_only=False):
    if active_count >= (1 if single_only else MAX_WORKERS):
        return False
    return available_gib >= (SINGLE_START_GIB if active_count == 0 else DUAL_START_GIB)


def account_configuration(path,case,prediction_sha=None):
    return dict(prediction_path=str(path),prediction_sha256=prediction_sha or digest(path),
        portfolio=case["portfolio"],rebalance=case["rebalance"],offset=case.get("offset",0),
        friction=case.get("friction",0.),initial_cash=1_000_000,risk_degree=.95,trade_unit=100,
        execution="T+1 open",annualization=252,engine="audited_qlib_v2")


def unique_account_cases(path,cases,store,prediction_sha=None):
    sha=prediction_sha or digest(path)
    selected={}
    for case in cases:
        key=store.key("account",account_configuration(path,case,sha))
        selected.setdefault(key,case)
    return list(selected.values())


def split_account_cases(cases):
    # Alternating cases balance K/turnover costs while keeping each child's local order.
    return [part for part in [cases[::2],cases[1::2]] if part]


def account_batch_cached(job,store):
    from paper_futu.run_final_ablation import prediction_path,spec_from
    path=prediction_path(spec_from(job["spec"]),job["horizon"],job["policy"],job["months"],job["seed"])
    sha=digest(path)
    return all(store.cached("account",account_configuration(path,c,sha)) is not None for c in job["cases"])


def merge_account_rows(job,path,cases,store,require_complete=True):
    sha=digest(path)
    rows=[]
    missing=[]
    for case in unique_account_cases(path,cases,store,sha):
        result=store.cached("account",account_configuration(path,case,sha))
        if result is None:
            missing.append(store.key("account",account_configuration(path,case,sha)))
            continue
        rows += [{**r,"spec":job["spec"]["name"],"engine":job["spec"]["engine"],
                  "policy":job["policy"],"horizon":job["horizon"],"months":job["months"],"seed":job["seed"]}
                 for r in result["rows"]]
    if require_complete and missing:
        raise RuntimeError(f"Cannot mark account batch complete; unfinished keys: {missing}")
    return rows,missing


def work_id(job,role):
    return stream_id(job) if role == "training" else identity({k:v for k,v in job.items() if k != "output"})[:24]


def worker_sample(item):
    p = psutil.Process(item["pid"])
    if abs(p.create_time()-item["create_time"]) > .01:
        raise RuntimeError("Managed training PID was reused")
    cpu = p.cpu_times()
    state_path = CONTROL_ROOT/"workers"/f"{p.pid}.json"
    phase = read_json(state_path).get("phase") if state_path.exists() else "initializing"
    return dict(pid=p.pid,rss_bytes=p.memory_info().rss,cpu_seconds=cpu.user+cpu.system,phase=phase)


def run_work_pool(jobs,label,role="training"):
    """A shared maximum of two hidden children, never two fitting plus two account children."""
    store = Store()
    if role == "training":
        jobs=unique_jobs(jobs)
        cached=stream_cached
    elif role == "account":
        jobs=list({work_id(j,role):j for j in jobs}.values())
        for job in jobs:
            unique_jobs([job])  # Same frozen model/H/policy/month/seed scope.
            if not job.get("cases") or job.get("kind") != "accounts":
                raise ValueError("An account worker must receive a nonempty account-only batch")
        cached=account_batch_cached
    else:
        raise ValueError(f"Unknown worker role {role}")
    pending = [{"job":j,"attempt":0} for j in jobs if not cached(j,store)]
    root = CONTROL_ROOT
    root.mkdir(parents=True,exist_ok=True)
    started = time.time()
    worker_file="parallel_full_factor_worker.py" if role == "training" else "parallel_account_worker.py"
    summary = dict(label=label,role=role,controller_pid=os.getpid(),started=started,status="running",
        scientific_code_sha256=scope.SCIENTIFIC_HASH,worker_sha256=digest(Path(__file__).with_name(worker_file)),
        scheduling_sha256=digest(__file__),threads_per_process=THREADS,max_training_processes=MAX_WORKERS,
        requested_streams=len(jobs),cached_streams=len(jobs)-len(pending),new_streams=len(pending),
        maximum_active_processes=0,dual_fit_samples=0,dual_account_samples=0,minimum_available_gib=None,
        peak_combined_rss_gib=0.,peak_rss_gib_by_pid={},cpu_seconds_by_pid={},
        memory_fallback=False,completed=[],retries=[],workers=[])
    active=[]
    single_only=False
    last_launch=0.
    history = root/f"pool_{label}_{int(started)}.json"
    print(f"POOL {role} {label}: {len(pending)} uncached work items, shared maximum two 4-thread workers",flush=True)
    try:
        while pending or active:
            available = psutil.virtual_memory().available/2**30
            summary["minimum_available_gib"] = available if summary["minimum_available_gib"] is None else min(available,summary["minimum_available_gib"])
            if available < MIN_AVAILABLE_GIB:
                single_only=True
                summary["memory_fallback"]=True
            if available < EMERGENCY_GIB and len(active) > 1:
                # Only our own newest managed training child may be preempted for memory.
                item=active[-1]
                if item["process"].poll() is None:
                    try:
                        worker_sample(item)  # Verify creation time immediately before termination.
                    except psutil.NoSuchProcess:
                        pass
                    else:
                        item["process"].terminate()
                        item["memory_preempted"]=True
            samples=[]
            for item in list(active):
                code=item["process"].poll()
                if code is None:
                    try:
                        samples.append(worker_sample(item))
                    except psutil.NoSuchProcess:
                        pass
                    continue
                item["output"].close()
                active.remove(item)
                if item.get("memory_preempted"):
                    code=MEMORY_EXIT_CODE
                if code == 0:
                    if not cached(item["job"],store):
                        raise RuntimeError(f"{role} worker exited without fully verified task outputs")
                    summary["completed"].append(dict(job=item["job"],pid=item["pid"],elapsed_seconds=time.time()-item["started"]))
                elif retry_allowed(code,item["attempt"]):
                    single_only=True
                    summary["memory_fallback"]=True
                    summary["retries"].append(dict(job=item["job"],pid=item["pid"],reason="explicit_memory_failure",attempt=1))
                    pending.insert(0,dict(job=item["job"],attempt=1))
                else:
                    raise RuntimeError(f"{role} worker {item['pid']} failed (exit {code}); no ordinary-error retry; log={item['log']}")
            for sample in samples:
                pid=str(sample["pid"])
                summary["peak_rss_gib_by_pid"][pid]=max(summary["peak_rss_gib_by_pid"].get(pid,0.),sample["rss_bytes"]/2**30)
                summary["cpu_seconds_by_pid"][pid]=sample["cpu_seconds"]
            summary["peak_combined_rss_gib"]=max(summary["peak_combined_rss_gib"],sum(s["rss_bytes"] for s in samples)/2**30)
            if sum(s["phase"] in ["final_fit","validation_fold"] for s in samples) >= 2:
                summary["dual_fit_samples"]+=1
            if sum(s["phase"] == "account" for s in samples) >= 2:
                summary["dual_account_samples"]+=1
            summary["maximum_active_processes"]=max(summary["maximum_active_processes"],len(active))
            if pending and launch_allowed(available,len(active),single_only) and time.time()-last_launch >= 2:
                item=pending.pop(0)
                job=item["job"]
                key=work_id(job,role)
                log=root/f"{label}_{key}_attempt{item['attempt']}.log"
                output=log.open("a",encoding="utf-8")
                module=WORKER_MODULE if role == "training" else ACCOUNT_WORKER_MODULE
                command=[sys.executable,"-u","-m",module,"--worker",json.dumps(job)]
                process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],stdout=output,stderr=subprocess.STDOUT,
                    env={**os.environ,"PYTHONIOENCODING":"utf-8","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4"},
                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                item.update(process=process,output=output,pid=process.pid,create_time=psutil.Process(process.pid).create_time(),
                    started=time.time(),command=command,log=str(log))
                active.append(item)
                last_launch=time.time()
                print(f"WORK START {role} {process.pid} {job['spec']['name']} H{job['horizon']} {job['policy']} M{job['months']} S{job['seed']}; log={log}",flush=True)
            summary["workers"]=[{k:i[k] for k in ["pid","create_time","started","command","log","job","attempt"]} for i in active]
            summary.update(updated=time.time(),pending_streams=len(pending),single_process_fallback=single_only)
            save_json(root/"status.json",summary)
            save_json(history,summary)
            if pending or active:
                time.sleep(1)
        summary.update(status="complete",elapsed_seconds=time.time()-started,workers=[])
        save_json(root/"status.json",summary)
        save_json(history,summary)
        print(f"POOL COMPLETE {role} {label}: {summary['new_streams']} work items; dual_fit={summary['dual_fit_samples']}; dual_account={summary['dual_account_samples']}",flush=True)
        return summary
    except BaseException as error:
        summary.update(status="failed",error=repr(error),updated=time.time())
        for item in active:
            if item["process"].poll() is None:
                item["process"].terminate()
            item["process"].wait()
            item["output"].close()
        summary["workers"]=[]
        save_json(root/"status.json",summary)
        save_json(history,summary)
        raise


def run_training_pool(jobs,label):
    return run_work_pool(jobs,label,"training")
