"""Hidden training-only worker with shared-key locks around the original implementation."""
from __future__ import annotations

import argparse
import json
import os
import time

from paper_futu.final_ablation_core import Store,code_hash,digest,identity,save_json
from paper_futu import final_ablation_training as training
from paper_futu import resume_full_factor_ablation as scope
from paper_futu.parallel_training_control import CONTROL_ROOT,MEMORY_EXIT_CODE,file_lock,is_memory_failure,stream_id,unique_jobs


def update_phase(phase,**extra):
    save_json(CONTROL_ROOT/"workers"/f"{os.getpid()}.json",dict(pid=os.getpid(),phase=phase,updated=time.time(),**extra))


def locked_store_run(original,store,kind,config,function,root=CONTROL_ROOT,phase_callback=None):
    key=store.key(kind,config)
    phase_callback=phase_callback or update_phase
    with file_lock(root/"locks"/f"task_{key}.lock"):
        def execute(task_key):
            phase_callback(kind,task_key=task_key)
            try:
                return function(task_key)
            finally:
                phase_callback("idle",finished_task=task_key)
        return original(store,kind,config,execute)


def install_locks():
    original_run=Store.run
    original_selection=training.selection

    def run(store,kind,config,function):
        return locked_store_run(original_run,store,kind,config,function)

    def selection(store,frame,spec,horizon,policy,year):
        # Other horizons read H20's policy/year CSV; serialize those reads and writes together.
        key=identity(dict(spec=spec.record(),policy=policy,year=year))[:24]
        with file_lock(CONTROL_ROOT/"locks"/f"selection_{key}.lock"):
            return original_selection(store,frame,spec,horizon,policy,year)

    Store.run=run
    training.selection=selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",required=True)
    args=parser.parse_args()
    job=unique_jobs([json.loads(args.worker)])[0]
    if code_hash() != scope.SCIENTIFIC_HASH:
        raise RuntimeError("Approved scientific code changed")
    install_locks()
    result_path=CONTROL_ROOT/"results"/f"{stream_id(job)}_{os.getpid()}.json"
    result=dict(pid=os.getpid(),job=job,scientific_code_sha256=scope.SCIENTIFIC_HASH,
                worker_sha256=digest(__file__),started=time.time())
    try:
        with file_lock(CONTROL_ROOT/"locks"/f"stream_{stream_id(job)}.lock"):
            from paper_futu.run_final_ablation import spec_from
            path=training.fit_stream(spec_from(job["spec"]),job["horizon"],job["policy"],job["months"],job["seed"])
        result.update(status="complete",prediction_path=str(path),prediction_sha256=digest(path),elapsed_seconds=time.time()-result["started"])
        save_json(result_path,result)
        update_phase("complete")
    except BaseException as error:
        memory=is_memory_failure(error)
        result.update(status="memory_failed" if memory else "failed",error=repr(error),elapsed_seconds=time.time()-result["started"])
        save_json(result_path,result)
        update_phase(result["status"],error=repr(error))
        if memory:
            raise SystemExit(MEMORY_EXIT_CODE) from error
        raise


if __name__ == "__main__":
    main()
