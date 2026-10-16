import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_futu import benchmark_attribution as ba
from paper_futu import prospective_s6 as ps


def payload(code, dates, levels=None):
    levels = levels if levels is not None else np.arange(len(dates))+100.
    return {"code":"200","data":[dict(tradeDate=d.strftime("%Y%m%d"),indexCode=code,
        indexNameCnAll="沪深300全收益指数" if code=="H00300" else "沪深300指数",
        open=float(level),close=float(level)) for d,level in zip(dates,levels)]}


def test_official_identity_duplicates_and_positive_closes():
    dates=pd.bdate_range("2022-01-03",periods=40)
    good=payload("H00300",dates)
    assert len(ba.official_frame(good,"H00300"))==40
    with pytest.raises(ValueError,match="identity"):
        ba.official_frame(good,"000300")
    bad=json.loads(json.dumps(good));bad["data"].append(bad["data"][0])
    with pytest.raises(ValueError,match="Duplicate"):ba.official_frame(bad,"H00300")
    bad=json.loads(json.dumps(good));bad["data"][0]["close"]=0
    with pytest.raises(ValueError,match="closing"):ba.official_frame(bad,"H00300")


def aligned_fixture():
    dates=pd.bdate_range("2021-12-31",periods=42)
    price=ba.official_frame(payload("000300",dates),"000300")
    total=ba.official_frame(payload("H00300",dates,np.arange(len(dates))*1.1+100),"H00300")
    ret=price.close.pct_change().iloc[1:];ret.iloc[0]=0
    daily=pd.concat([pd.DataFrame(dict(offset=i,datetime=dates[1:],benchmark_return=ret.to_numpy(),
        net_return=np.repeat(.001,41),portfolio_nav=np.cumprod(np.repeat(1.001,41)),fees_cny=0)) for i in range(5)])
    return price,total,daily


def test_benchmark_alignment_never_fills_missing_days():
    price,total,daily=aligned_fixture()
    result,audit=ba.align_benchmarks(price,total,daily)
    assert audit["days"]==41
    assert result.total_return.iloc[0]==0
    with pytest.raises(ValueError,match="Missing benchmark"):ba.align_benchmarks(price,total.iloc[:-1],daily)
    bad=daily.copy();bad.loc[bad.index==4,"benchmark_return"]+=.01
    with pytest.raises(ValueError,match="does not match"):ba.align_benchmarks(price,total,bad)


def test_capm_recovers_known_beta_and_alpha():
    x=np.linspace(-.02,.02,200)
    noise=.001*np.cos(np.arange(200)*1.3)
    y=.0004+.7*x+noise
    rf=np.expm1(np.log1p(.01)/252)
    result=ba.capm(y+rf,x,20)
    expected=np.linalg.lstsq(np.column_stack([np.ones(200),x]),y,rcond=None)[0]
    assert result["market_beta"]==pytest.approx(expected[1])
    assert result["alpha_annual_arithmetic"]==pytest.approx(expected[0]*252)
    assert result["alpha_ci_low"]<result["alpha_annual_arithmetic"]<result["alpha_ci_high"]
    assert result["alpha_nominal_p"]<.05


def test_style_features_are_lagged_and_missing_is_not_zero():
    dates=pd.bdate_range("2022-01-03",periods=5)
    index=pd.MultiIndex.from_product([[f"S{i:02d}" for i in range(40)],dates],names=["instrument","datetime"])
    panel=pd.DataFrame(index=index)
    for col in ba.STYLE_COLUMNS:panel[col]=np.repeat(np.arange(40),len(dates))
    panel["return_1d"]=np.repeat(np.arange(40)/10000,len(dates))
    base,_=ba.style_factors(panel,{f"S{i:02d}" for i in range(15)})
    modified=panel.copy();modified.loc[(slice(None),dates[-1]),ba.STYLE_COLUMNS]*=-1
    changed,_=ba.style_factors(modified,{f"S{i:02d}" for i in range(15)})
    pd.testing.assert_frame_equal(base,changed)
    assert base.iloc[0].isna().all()


@pytest.fixture
def frozen(tmp_path):
    root=tmp_path/"project";output=root/"reports/paper_futu/forward"
    for name in ps.CODE_FILES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    model=root/"models/package";model.mkdir(parents=True);(model/"model.txt").write_text("test model")
    financial=root/"data/financial.parquet";financial.parent.mkdir(parents=True,exist_ok=True);financial.write_bytes(b"fixture")
    meta=dict(strategy_id=ps.STRATEGY,portfolio=ps.PORTFOLIO,parameters=dict(reg_alpha=1.,reg_lambda=5.,
        num_leaves=15,min_child_samples=300,n_jobs=4),input_kind="raw",target_kind="rank",
        label_expression="Ref($open, -11) / Ref($open, -1) - 1",retrain_months=3,seed=42,
        features=[f"f{i}" for i in range(15)],reload_equal=True,label_end_max="2026-09-30",
        information_cutoff="2026-09-30",model_sha256=ba.digest(model/"model.txt"),chosen_iteration=40,
        serving_signal_end="2026-12-31",financial_events_path="data/financial.parquet",
        financial_events_sha256=ba.digest(financial))
    ba.save_json(model/"metadata.json",meta)
    main=dict(model_path=str(model/"model.txt"),model_sha256=meta["model_sha256"],chosen_iteration=40,
        effective_signal_date="2026-10-19",first_execution_date="2026-10-20")
    ba.save_json(root/"configs/experiments/s6_head6_40_research_main.json",main)
    config=dict(environment="SIMULATE",market="CN",model_dir="models/package",
        pending_deployment=dict(model_sha256=meta["model_sha256"],target_portfolio=ps.PORTFOLIO,
                                metadata_sha256=ba.digest(model/"metadata.json")))
    ba.save_json(root/"configs/futu_cn_paper.json",config)
    calendar=root/"data/interim/futu_cn_provider/2026-10-11/extracted/qlib_bin/calendars/day_future.txt"
    calendar.parent.mkdir(parents=True);calendar.write_text("2026-10-19\n2026-10-20\n2026-10-21\n")
    attr=root/"reports/attribution/manifest.json"
    ba.save_json(attr,dict(status="complete",strategy_id=ps.STRATEGY,as_of="2026-10-13",
                          alignment=dict(missing_dates=0),artifacts={}))
    ps.freeze(root,output,attr,datetime(2026,10,16,18,tzinfo=ps.TZ))
    return root,output,meta


def test_freeze_is_idempotent_and_detects_mutation(frozen):
    root,out,meta=frozen
    assert ps.status(out)["recorded_sessions"]==0
    assert ps.record(out,"2026-10-16",datetime(2026,10,16,18,tzinfo=ps.TZ))["status"]=="awaiting_first_execution"
    protocol=ps.read(out/"protocol.json")
    (root/ps.CODE_FILES[0]).write_text("changed")
    with pytest.raises(ValueError,match="Frozen code"):ps.verify_protocol(out)


def prepare_close(root,day,meta,assets=101.):
    state=root/"data/interim/futu_cn_paper"
    for d,value in [("2026-10-19",100.),(day,assets)]:
        ba.save_json(state/("nav_saved_"+d+".json"),dict(time=d+"T15:39:00+08:00",assets=value,cash=5.,holdings=20))
    plan=dict(model_sha256=meta["model_sha256"],execution_date="2026-10-20",**ps.PORTFOLIO)
    ba.save_json(state/day/"plan.json",plan);ba.save_json(state/"active_plan.json",plan)
    ba.save_json(state/("execution_"+day+".json"),dict(environment="SIMULATE",submitted_orders=0))
    return state


def test_forward_record_never_backfills_or_relabels_and_is_reproducible(frozen,monkeypatch):
    root,out,meta=frozen;state=prepare_close(root,"2026-10-20",meta)
    def fake_fetch(raw,start,end):
        dates=pd.to_datetime([start,end])
        for code in ["000300","H00300"]:ba.save_json(Path(raw)/(code+".json"),payload(code,dates,[100.,100.5]))
        ba.save_json(Path(raw)/"sources.json",dict(provider="fixture"))
    monkeypatch.setattr(ps,"fetch_official",fake_fetch)
    now=datetime(2026,10,20,18,tzinfo=ps.TZ)
    assert ps.record(out,"2026-10-20",now)["status"]=="recorded"
    assert ps.record(out,"2026-10-20",now)["status"]=="already_recorded"
    status=ps.status(out)
    assert status["recorded_sessions"]==1
    assert status["observed_account_return"]==pytest.approx(.01)
    assert status["benchmark_return"]==pytest.approx(.005)
    rowpath=out/"daily/2026-10-20.json";row=ps.read(rowpath);row["assets"]=500
    ba.save_json(rowpath,row)
    with pytest.raises(ValueError,match="cannot be reproduced"):ps.status(out)


def test_missing_prior_day_blocks_recording(frozen):
    root,out,meta=frozen;prepare_close(root,"2026-10-21",meta)
    with pytest.raises(ValueError,match="Missing prior"):
        ps.record(out,"2026-10-21",datetime(2026,10,21,18,tzinfo=ps.TZ))


def test_intraday_balance_is_rejected(tmp_path):
    p=tmp_path/"nav.json";ba.save_json(p,dict(time="2026-10-20T10:00:00+08:00",assets=100.,cash=5.,holdings=20))
    with pytest.raises(ValueError,match="intraday"):ps.closing_marker(p,"2026-10-20")


def test_fee_estimate_uses_filled_notional_and_minimum(tmp_path):
    p=tmp_path/"orders.csv"
    pd.DataFrame(dict(order_id=["a","b","c"],trd_side=["BUY","SELL","BUY"],
        dealt_qty=[100,200,0],dealt_avg_price=[10,100,0])).to_csv(p,index=False)
    assert ps.filled_costs(p)==dict(filled_notional=21000.,modeled_fees_cny=35.,orders_with_fills=2)
