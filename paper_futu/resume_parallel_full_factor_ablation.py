"""Resume the unchanged scientific stages with two memory-gated training children."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import pandas as pd
import psutil

from paper_futu import resume_full_factor_ablation as scope
from paper_futu.final_ablation_core import REPORT,code_hash,digest,protected_state,read_json,save_json
from paper_futu.parallel_training_control import (
    CONTROL_ROOT,THREADS,MAX_WORKERS,WORKER_MODULE,ACCOUNT_WORKER_MODULE,cadence_jobs,seed_jobs,stream_id,make_job,
    file_lock,run_training_pool,run_work_pool,account_configuration,unique_account_cases,split_account_cases,
    merge_account_rows,
)

MODULE="paper_futu.resume_parallel_full_factor_ablation"
runner=scope.runner
_ORIGINAL_JOB_STREAM=runner.job_stream
_SEED_PLAN=None


def parallel_metadata(path,value):
    if Path(path).name in {"runner_process.json","protocol.json","reduced_scope.json"}:
        value={**value,"resume_entry":MODULE,"resume_entry_sha256":digest(__file__),
            "base_resume_entry_sha256":digest(scope.__file__),
            "training_worker_sha256":digest(Path(__file__).with_name("parallel_full_factor_worker.py")),
            "training_scheduling_sha256":digest(Path(__file__).with_name("parallel_training_control.py")),
            "retained_model_names":scope.ALLOWED_NAMES,"execution_stages":runner.STAGES,
            "model_and_tune_stages_repeated":False,
            "maximum_training_processes":MAX_WORKERS,"threads_per_training_process":THREADS,
            "account_worker_sha256":digest(Path(__file__).with_name("parallel_account_worker.py")),
            "maximum_total_work_processes":2,"account_execution":"parallel independent accounts; serial dates per account",
            "scientific_code_sha256":scope.SCIENTIFIC_HASH}
    save_json(path,value)


def cadence_stage():
    scope.validate_candidates("cadence_candidates.json")
    run_training_pool(cadence_jobs(read_json(REPORT/"cadence_candidates.json")),"cadence")
    scope.guarded_stage("cadence")()


def job_stream(spec,horizon,policy="fixed",months=12,seed=42):
    global _SEED_PLAN
    if seed in [2024,2025]:
        if _SEED_PLAN is None:
            candidates=read_json(REPORT/"robustness_candidates.json")
            scores=pd.read_csv(REPORT/"robustness_all.csv",float_precision="round_trip")
            _,order=runner.choose_final(candidates,scores)
            _SEED_PLAN=seed_jobs(candidates,order)
            # The source's first seed call must belong to exactly this pretraining plan.
            if stream_id(make_job(spec.record(),horizon,policy,months,seed)) not in {stream_id(j) for j in _SEED_PLAN}:
                raise RuntimeError("Seed pretraining plan disagrees with the original stage's selection")
            run_training_pool(_SEED_PLAN,"seed")
        if stream_id(make_job(spec.record(),horizon,policy,months,seed)) not in {stream_id(j) for j in _SEED_PLAN}:
            raise RuntimeError("A seed stream outside the original top-three plan was requested")
    return _ORIGINAL_JOB_STREAM(spec,horizon,policy,months,seed)


def job_accounts(spec,horizon,policy,cases,output,months=12,seed=42,anchor=False):
    from paper_futu.final_ablation_core import Store,identity
    scope.validate_spec(spec.record())
    if anchor:
        raise RuntimeError("Resumed account pool cannot request a new anchor training stage")
    store=Store()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    path=runner.prediction_path(spec,horizon,policy,months,seed)
    meta=read_json(path.with_suffix(".json"))
    sha=digest(path)
    if sha!=meta["prediction_sha256"]:
        raise ValueError("Stitched predictions changed before account scheduling")
    cases=unique_account_cases(path,cases,store,sha)
    remaining=[c for c in cases if store.cached("account",account_configuration(path,c,sha)) is None]
    job=dict(kind="accounts",spec=spec.record(),horizon=horizon,policy=policy,months=months,seed=seed,
             cases=cases,output=str(output))
    job_key=identity(job)[:16]
    event_log=CONTROL_ROOT/f"accounts_{job_key}_events.log"
    event_log.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    current=dict(job=job,log=str(event_log),started=started,status="running",code_sha256=scope.SCIENTIFIC_HASH,
                 original_case_count=len(cases),uncached_case_count=len(remaining),worker_count_limit=2)
    save_json(REPORT/"current_job.json",current)
    batches=[]
    for index,part in enumerate(split_account_cases(remaining)):
        fragment=CONTROL_ROOT/"fragments"/f"{job_key}_{index}.csv"
        fragment.parent.mkdir(parents=True,exist_ok=True)
        batches.append({**job,"cases":part,"output":str(fragment)})
    with event_log.open("a",encoding="utf-8") as log:
        log.write(f"START {len(cases)} cases; cached={len(cases)-len(remaining)}, remaining={len(remaining)}\n")
    try:
        if batches:
            run_work_pool(batches,f"accounts_{job_key}","account")
        rows,missing=merge_account_rows(job,path,cases,store,require_complete=True)
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(output,index=False,encoding="utf-8-sig")
        current.update(status="complete",updated=time.time(),elapsed_seconds=time.time()-started,exit_code=0)
        with event_log.open("a",encoding="utf-8") as log:
            log.write(f"COMPLETE {len(cases)} verified cases\n")
        save_json(REPORT/"current_job.json",current)
    except BaseException as error:
        try:
            rows,missing=merge_account_rows(job,path,cases,store,require_complete=False)
            if rows:
                pd.DataFrame(rows).to_csv(output,index=False,encoding="utf-8-sig")
        except BaseException as merge_error:
            missing=["partial output verification failed"]
            current["partial_merge_error"]=repr(merge_error)
        current.update(status="failed",error=repr(error),unfinished_keys=missing,
                       updated=time.time(),elapsed_seconds=time.time()-started,exit_code=1)
        save_json(REPORT/"current_job.json",current)
        raise


def configure(start_stage):
    scope.configure_runner()
    runner.STAGES=scope.STAGES[scope.STAGES.index(start_stage):]
    runner.save_json=parallel_metadata
    runner.stage_cadence=cadence_stage
    runner.job_stream=job_stream
    runner.job_accounts=job_accounts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of",default="2026-10-13",choices=["2026-10-13"])
    parser.add_argument("--stage",default="all",choices=scope.STAGES+["all"])
    parser.add_argument("--start-stage",default="portfolio",choices=scope.STAGES)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    configure(args.start_stage)
    sys.argv=[sys.argv[0],"--as-of",args.as_of,"--stage",args.stage]+(["--dry-run"] if args.dry_run else [])
    if args.dry_run:
        runner.main()
        return
    if code_hash()!=scope.SCIENTIFIC_HASH:
        raise RuntimeError("Approved scientific code changed")
    if args.start_stage == "cadence":
        scope.validate_candidates("cadence_candidates.json")
    modules=[MODULE,scope.MODULE,"paper_futu.run_final_ablation",WORKER_MODULE,ACCOUNT_WORKER_MODULE]
    active=[p.info["pid"] for p in psutil.process_iter(["pid","cmdline","cwd"])
            if p.pid!=os.getpid() and p.info["cmdline"] and p.info["cwd"]==str(Path(__file__).resolve().parents[1])
            and any(m in p.info["cmdline"] for m in modules) and "--dry-run" not in p.info["cmdline"]]
    if active:
        raise RuntimeError(f"Other study processes still active: {active}")
    with file_lock(CONTROL_ROOT/"controller.lock",timeout=0):
        before=protected_state()
        record=scope.prepare_resume()
        record.update(status="running",runner_pid=os.getpid(),started=time.time(),
            execution_stages=runner.STAGES,resume_entry=MODULE,resume_entry_sha256=digest(__file__),
            training_worker_sha256=digest(Path(__file__).with_name("parallel_full_factor_worker.py")),
            training_scheduling_sha256=digest(Path(__file__).with_name("parallel_training_control.py")),
            account_worker_sha256=digest(Path(__file__).with_name("parallel_account_worker.py")),
            maximum_total_work_processes=2,maximum_training_processes=2,threads_per_training_process=4,
            account_execution="parallel independent accounts; serial dates per account")
        save_json(REPORT/"parallel_resume.json",record)
        try:
            runner.main()
            record.update(status="complete",updated=time.time())
        except BaseException as error:
            record.update(status="failed",error=repr(error),updated=time.time())
            raise
        finally:
            record["protected_state_unchanged"]=before==protected_state()
            save_json(REPORT/"parallel_resume.json",record)
            if not record["protected_state_unchanged"]:
                raise RuntimeError("Protected files changed during parallel resume")


if __name__ == "__main__":
    main()
