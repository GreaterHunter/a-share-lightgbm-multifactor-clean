"""Causality, common account periods and immutable cache checks for S6."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_futu import research_s6_2026 as study
from paper_futu.s6_2026_account import schedule_dates
from paper_futu.s6_2026_report import statistics
from paper_futu import overfit_four_core as core


def test_quarterly_updates_cover_every_execution_year_and_share_june(monkeypatch):
    dates = pd.bdate_range("2021-12-01","2026-10-13")
    # Model the calendar's holiday gap explicitly; dates must not be invented.
    dates = dates[~dates.isin(pd.to_datetime(["2026-10-01","2026-10-02","2026-10-05","2026-10-06","2026-10-07"]))]
    monkeypatch.setattr(study,"calendar",lambda:dates)
    quarter = study.serving_periods(3)
    half = study.serving_periods(6)
    assert len(quarter)==20 and len(half)==10
    assert len(study.fit_cutoffs())==15
    for year in range(2022,2027):
        parts = [p for p in quarter if p["year"]==year]
        assert len(parts)==4
        assert all(p["cutoff"] < p["start"] for p in parts)
        assert all(p["cutoff"]==str(dates[dates < p["start"]][-1].date()) for p in parts)
        assert parts[2]["cutoff"]==next(p for p in half if p["year"]==year and p["start"][5:7]=="07")["cutoff"]
    assert quarter[-1]["start"]=="2026-10-08"


def test_rebalance_offsets_preserve_interval_and_do_not_reset_at_new_year():
    dates = pd.bdate_range("2021-12-01","2026-10-13")
    all_days,_,_ = schedule_dates(dates,"2022-01-04","2026-10-13",0)
    for offset in range(5):
        days,trades,signals = schedule_dates(dates,"2022-01-04","2026-10-13",offset)
        assert days.equals(all_days)
        assert trades.equals(all_days[offset::5])
        assert np.diff(dates.get_indexer(trades)).tolist()==[5]*(len(trades)-1)
        assert np.all(dates.get_indexer(trades)-dates.get_indexer(signals)==1)


@pytest.mark.parametrize("cutoff",["2021-12-31","2022-03-31","2022-06-30","2022-09-30","2025-12-31","2026-06-30"])
def test_training_uses_label_end_not_just_signal_date(cutoff):
    cutoff = pd.Timestamp(cutoff)
    index = pd.MultiIndex.from_arrays([["A"]*3,[cutoff-pd.Timedelta(days=30),cutoff-pd.Timedelta(days=1),cutoff+pd.Timedelta(days=1)]],names=["instrument","datetime"])
    frame = pd.DataFrame(1.,index=index,columns=list(core.frozen.ENHANCED_FEATURE_NAMES))
    frame["label"],frame["target"] = [.1,.2,.3],[.1,.2,.3]
    frame["label_end_date"] = [cutoff,cutoff+pd.Timedelta(days=10),cutoff+pd.Timedelta(days=20)]
    first = core.mature(frame,"2010-01-01",cutoff)
    frame.loc[index[1:], ["label","target"]] = 999
    second = core.mature(frame,"2010-01-01",cutoff)
    assert len(first)==1
    pd.testing.assert_frame_equal(first,second)


def test_drawdown_includes_initial_nav_and_returns_compound():
    dates = pd.bdate_range("2026-01-05",periods=2)
    f = pd.DataFrame(dict(net_return=[-.1,.1],bench=[0.,0.],fees_cash=[5.,5.],turnover=[.2,.2],
        holdings_count=[20,20],bank_weight=[.3,.4],largest_stock_weight=[.1,.1]),index=dates)
    r = statistics(f)
    assert r["total_return"]==pytest.approx(-.01)
    assert r["max_drawdown"]==pytest.approx(.1)
    assert r["fees_cny"]==10


def test_scope_is_bounded_and_cadence_never_crosses_with_training_variants():
    jobs = study.account_jobs()
    assert len(jobs)==50
    assert len({(j["mode"],j["case"],j["offset"]) for j in jobs})==50
    assert sum(j["mode"]=="fresh" for j in jobs)==40
    assert {j["case"] for j in jobs if j["mode"]=="history"}=={"M6","M3"}


def test_cache_rejects_changed_artifact_or_protocol(monkeypatch,tmp_path):
    monkeypatch.setattr(study,"OUT",tmp_path)
    monkeypatch.setattr(study,"implementation",lambda:"fixed")
    artifact = tmp_path / "model.txt"
    artifact.write_text("valid",encoding="utf-8")
    study.save(tmp_path / "protocol.json",{"version":1})
    task = dict(status="complete",implementation_sha256="fixed",protocol_sha256=study.digest(tmp_path / "protocol.json"),
        artifacts={str(artifact):study.digest(artifact)},result={"model":"ok"})
    study.save(tmp_path / "tasks/fit_demo.json",task)
    assert study.cached("fit","demo")=={"model":"ok"}
    artifact.write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="Artifact hash"):
        study.cached("fit","demo")
    artifact.write_text("valid",encoding="utf-8")
    study.save(tmp_path / "protocol.json",{"version":2})
    with pytest.raises(ValueError,match="protocol"):
        study.cached("fit","demo")


def test_scheduler_passes_string_path_to_windows_psutil(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import psutil
    monkeypatch.setattr(study,"OUT",tmp_path)
    completed = set()
    monkeypatch.setattr(study,"cached",lambda kind,name: {} if name in completed else None)
    class Child:
        pid,returncode = 999999,0
        def __init__(self,*args,**kwargs):
            completed.add("one")
        def poll(self): return 0
        def wait(self): return 0
    def disk_usage(path):
        assert isinstance(path,str)
        return SimpleNamespace(free=10*2**30)
    monkeypatch.setattr(study.subprocess,"Popen",Child)
    monkeypatch.setattr(study.time,"sleep",lambda _:None)
    monkeypatch.setattr(psutil,"disk_usage",disk_usage)
    monkeypatch.setattr(psutil,"virtual_memory",lambda:SimpleNamespace(available=6*2**30))
    study.execute_queue("fit",["one"])
    assert study.read(tmp_path / "status.json")["completed"]==1
