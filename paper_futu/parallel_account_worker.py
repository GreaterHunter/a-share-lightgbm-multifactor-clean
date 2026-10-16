"""Independent account batch/fragment, using the original serial-per-account replay."""
from __future__ import annotations

import argparse
import json
import os
import time

from paper_futu.final_ablation_core import Store,code_hash,digest,save_json
from paper_futu import resume_full_factor_ablation as scope
from paper_futu.parallel_training_control import CONTROL_ROOT,MEMORY_EXIT_CODE,is_memory_failure,unique_jobs,work_id
from paper_futu.parallel_full_factor_worker import locked_store_run,update_phase
from src.runtime_budget import require_memory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",required=True)
    args=parser.parse_args()
    job=json.loads(args.worker)
    unique_jobs([job])
    if job.get("kind") != "accounts" or not job.get("cases") or code_hash()!=scope.SCIENTIFIC_HASH:
        raise RuntimeError("Invalid account-only job or scientific hash")
    original=Store.run
    def run(store,kind,config,function):
        def guarded(key):
            if kind == "account":
                require_memory(2.,"并行研究账户回测")
            return function(key)
        return locked_store_run(original,store,kind,config,guarded)
    Store.run=run
    result=dict(pid=os.getpid(),job=job,started=time.time(),scientific_code_sha256=scope.SCIENTIFIC_HASH,
                worker_sha256=digest(__file__))
    result_path=CONTROL_ROOT/"results"/f"account_{work_id(job,'account')}_{os.getpid()}.json"
    try:
        from paper_futu.run_final_ablation import worker
        worker(job)  # Original math and per-account date order; only output is a private fragment.
        result.update(status="complete",elapsed_seconds=time.time()-result["started"])
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
