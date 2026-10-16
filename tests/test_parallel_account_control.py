"""Original account keys, disjoint fragments, ordered merges, and failure retention."""
from pathlib import Path
import json

import pandas as pd
import pytest

from paper_futu import parallel_training_control as control
from paper_futu import resume_parallel_full_factor_ablation as driver
from paper_futu import final_ablation_core as core
from paper_futu import final_ablation_account as account
from paper_futu.run_final_ablation import COARSE


class FakeAccountStore:
    def __init__(self):
        self.results={}

    def key(self,kind,config):
        return core.identity(dict(kind=kind,config=config))[:24]

    def cached(self,kind,config):
        return self.results.get(self.key(kind,config))


def test_account_identity_is_exactly_the_original_replay_configuration(tmp_path,monkeypatch):
    path=tmp_path/"predictions.parquet"
    path.write_bytes(b"prediction fixture")
    class Spy:
        def run(self,kind,config,execute):
            assert kind=="account"
            return config
    monkeypatch.setattr(account,"Store",Spy)
    case=dict(portfolio=COARSE[0],rebalance=10,offset=2,friction=.002)
    assert control.account_configuration(path,case)==account.replay(path,**case)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    path=tmp_path/"predictions.parquet"
    path.write_bytes(b"prediction fixture")
    core.save_json(path.with_suffix(".json"),dict(prediction_sha256=core.digest(path)))
    store=FakeAccountStore()
    monkeypatch.setattr(core,"Store",lambda:store)
    monkeypatch.setattr(driver,"REPORT",tmp_path)
    monkeypatch.setattr(driver,"CONTROL_ROOT",tmp_path/"control")
    monkeypatch.setattr(driver.runner,"prediction_path",lambda *a,**k:path)
    cases=[dict(portfolio=p,rebalance=10) for p in COARSE]
    def finish(case):
        config=control.account_configuration(path,case)
        key=store.key("account",config)
        store.results[key]=dict(rows=[dict(case_id=key,period="all",total_fees_cny=5.,annualized_return=.1)])
    return tmp_path,path,store,cases,finish


def test_parallel_fragments_are_disjoint_and_parent_merges_original_order(setup,monkeypatch):
    root,path,store,cases,finish=setup
    seen=[]
    def pool(jobs,label,role):
        assert role=="account" and len(jobs)==2
        keys=[{store.key("account",control.account_configuration(path,c)) for c in j["cases"]} for j in jobs]
        assert not keys[0]&keys[1]
        assert jobs[0]["output"]!=jobs[1]["output"]
        for job in reversed(jobs):  # The second child completes first.
            for case in job["cases"]:
                finish(case)
                seen.append(case)
    monkeypatch.setattr(driver,"run_work_pool",pool)
    output=root/"merged.csv"
    driver.job_accounts(core.A,20,"fixed",cases,output)
    expected=[store.key("account",control.account_configuration(path,c)) for c in cases]
    assert list(pd.read_csv(output).case_id)==expected
    assert json.loads((root/"current_job.json").read_text())["status"]=="complete"
    assert seen==list(reversed(cases))


def test_duplicate_case_is_computed_once_and_complete_cache_is_preserved(setup,monkeypatch):
    root,path,store,cases,finish=setup
    finish(cases[0])
    called=[]
    def pool(jobs,label,role):
        assert len(jobs)==1 and jobs[0]["cases"]==[cases[1]]
        called.extend(jobs[0]["cases"])
        finish(cases[1])
    monkeypatch.setattr(driver,"run_work_pool",pool)
    output=root/"merged.csv"
    driver.job_accounts(core.A,20,"fixed",[cases[0],cases[1],cases[0]],output)
    assert called==[cases[1]]
    assert len(pd.read_csv(output))==2


def test_failure_retains_finished_accounts_and_never_marks_batch_complete(setup,monkeypatch):
    root,path,store,cases,finish=setup
    def pool(jobs,label,role):
        finish(cases[1])
        raise RuntimeError("second account failed")
    monkeypatch.setattr(driver,"run_work_pool",pool)
    output=root/"nested"/"partial.csv"
    with pytest.raises(RuntimeError,match="failed"):
        driver.job_accounts(core.A,20,"fixed",cases,output)
    assert len(pd.read_csv(output))==1
    state=json.loads((root/"current_job.json").read_text())
    assert state["status"]=="failed" and len(state["unfinished_keys"])==1
    assert store.cached("account",control.account_configuration(path,cases[1])) is not None
    with pytest.raises(RuntimeError,match="Cannot mark"):
        control.merge_account_rows(state["job"],path,cases,store,require_complete=True)
