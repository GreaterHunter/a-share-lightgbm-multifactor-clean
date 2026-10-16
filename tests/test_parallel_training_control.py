"""Concurrency regression tests use temporary artifacts and small fake tasks only."""
import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest

from paper_futu import parallel_training_control as control
from paper_futu.parallel_full_factor_worker import locked_store_run
from paper_futu.final_ablation_core import A,Spec,Store,code_hash,identity
from paper_futu import resume_full_factor_ablation as scope


class FakeStore:
    def __init__(self,path):
        self.path=Path(path)

    def key(self,kind,config):
        return identity(dict(kind=kind,config=config))[:24]


def fake_run(store,kind,config,function):
    path=store.path/"result.json"
    if path.exists():
        return json.loads(path.read_text())
    result=function(store.key(kind,config))
    path.write_text(json.dumps(result),encoding="utf-8")
    return result


def shared_task(path,start):
    root=Path(path)
    start.wait()
    def execute(key):
        (root/f"executed_{os.getpid()}.txt").write_text(key)
        counter=root/"executions.txt"
        value=int(counter.read_text()) if counter.exists() else 0
        time.sleep(.15)
        counter.write_text(str(value+1))
        return {"key":key,"result":sum(i*i for i in range(20))}
    result=locked_store_run(fake_run,FakeStore(root),"final_fit",dict(cutoff="2022-07-01",seed=42),
                            execute,root=root,phase_callback=lambda *a,**k:None)
    (root/f"returned_{os.getpid()}.json").write_text(json.dumps(result))


def crash_while_locked(path,ready):
    with control.file_lock(path):
        ready.set()
        os._exit(17)


def hold_lock(path,ready,release):
    with control.file_lock(path):
        ready.set()
        release.wait(15)


def test_shared_key_is_written_once_and_results_match_serial(tmp_path):
    context=multiprocessing.get_context("spawn")
    start=context.Event()
    processes=[context.Process(target=shared_task,args=(str(tmp_path),start)) for _ in range(2)]
    for p in processes:
        p.start()
    start.set()
    for p in processes:
        p.join(25)
        assert not p.is_alive() and p.exitcode == 0
    assert (tmp_path/"executions.txt").read_text() == "1"
    assert len(list(tmp_path.glob("executed_*.txt"))) == 1
    results=[json.loads(p.read_text()) for p in tmp_path.glob("returned_*.json")]
    assert len(results)==2 and results[0]==results[1]
    assert results[0]["result"]==sum(i*i for i in range(20))


def test_crashed_process_releases_os_lock_without_deleting_lock_file(tmp_path):
    context=multiprocessing.get_context("spawn")
    ready=context.Event()
    path=tmp_path/"task.lock"
    p=context.Process(target=crash_while_locked,args=(str(path),ready))
    p.start()
    assert ready.wait(15)
    p.join(15)
    assert p.exitcode==17 and path.exists()
    with control.file_lock(path,timeout=1):
        pass


def test_lock_blocks_conflicting_writer_then_releases(tmp_path):
    context=multiprocessing.get_context("spawn")
    ready,release=context.Event(),context.Event()
    path=tmp_path/"selection.lock"
    p=context.Process(target=hold_lock,args=(str(path),ready,release))
    p.start()
    try:
        assert ready.wait(15)
        with pytest.raises(TimeoutError):
            with control.file_lock(path,timeout=.2):
                pass
    finally:
        release.set()
        p.join(15)
    assert p.exitcode==0
    with control.file_lock(path,timeout=1):
        pass


def test_dedup_and_scope_only_schedule_original_candidate_streams():
    candidate=dict(candidate_id="c0_m6",spec=A.record(),horizon=10,policy="tuned",months=6)
    jobs=control.cadence_jobs([candidate,candidate])
    assert len(jobs)==6  # three existing anchor pilots and three candidate frequencies
    assert {j["months"] for j in jobs}=={12,6,3}
    assert len(control.seed_jobs([candidate],["c0_m6","c0_m6"]))==2
    bad=Spec("without_book_to_price",A.features[:-1]).record()
    with pytest.raises(ValueError):
        control.cadence_jobs([{**candidate,"spec":bad}])


def test_streams_can_differ_but_shared_fit_key_is_unchanged():
    j6=control.make_job(A.record(),10,months=6)
    j3=control.make_job(A.record(),10,months=3)
    assert control.stream_id(j6)!=control.stream_id(j3)
    store=object.__new__(Store)
    store.code,store.data=scope.SCIENTIFIC_HASH,scope.SNAPSHOT
    config=dict(spec=A.record(),horizon=10,seed=42,cutoff="2022-06-30")
    before=store.key("final_fit",config)
    assert store.key("final_fit",config)==before
    assert code_hash()==scope.SCIENTIFIC_HASH


def test_memory_gate_limited_retry_and_no_ordinary_error_retry():
    assert control.launch_allowed(3.85,1)
    assert not control.launch_allowed(3.0,1)
    assert not control.launch_allowed(4.,2)
    assert not control.launch_allowed(4.,1,single_only=True)
    assert control.is_memory_failure(RuntimeError("拟合需要至少2.0 GiB可用物理内存，目前仅1.90 GiB"))
    assert not control.is_memory_failure(RuntimeError("model serialization changed"))
    assert control.retry_allowed(75,0)
    assert not control.retry_allowed(75,1)
    assert not control.retry_allowed(1,0)


def test_seed_pretraining_csv_round_trip_preserves_entire_source_selection(tmp_path):
    import numpy as np
    import pandas as pd
    from paper_futu.run_final_ablation import choose_final

    # Exercise the strict one-percentage-point comparison at adjacent float64 values.
    boundary=.2-.01
    cagr=[.2,np.nextafter(boundary,np.inf),np.nextafter(boundary,-np.inf),.11]
    candidates=[dict(candidate_id=f"c{i}_m6",spec=A.record(),horizon=10,
                     policy="tuned",months=6,rebalance=10) for i in range(4)]
    scores=pd.DataFrame([dict(candidate_id=c["candidate_id"],period="all",friction=f,
        annualized_return=cagr[i] if f==0 else .07,max_drawdown=-.1-i*.01,
        average_daily_turnover=.01234567890123456+i*.001,total_fees_cny=4321.123456789012+i)
        for i,c in enumerate(candidates) for f in [0.,.002] for _ in range(5)])
    original_ranking,original_order=choose_final(candidates,scores)
    output=tmp_path/"robustness_all.csv"
    scores.to_csv(output,index=False,encoding="utf-8-sig")
    restored=pd.read_csv(output,float_precision="round_trip")
    recovered_ranking,recovered_order=choose_final(candidates,restored)
    pd.testing.assert_frame_equal(original_ranking,recovered_ranking)
    assert recovered_order==original_order
    assert control.seed_jobs(candidates,recovered_order)==control.seed_jobs(candidates,original_order)
