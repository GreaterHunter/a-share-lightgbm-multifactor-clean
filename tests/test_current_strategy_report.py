"""Guard the selected strategy's published arithmetic and sanitized evidence."""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import build_current_strategy_report as report


def test_drawdown_keeps_initial_cash_as_the_first_high():
    np.testing.assert_allclose(report.drawdown([.9, .95, 1.1, .99]), [-.1, -.05, 0, -.1])


def test_mean_cagr_is_the_mean_of_individual_accounts():
    frames=[]
    for offset, wealth in [(0,4.0),(1,1.0)]:
        frames.append(pd.DataFrame(dict(offset=offset,datetime=pd.bdate_range("2020-01-01",periods=504),
            net_return=np.full(504,wealth**(1/504)-1), benchmark_return=0.,fees_cny=0.,turnover=0.,holdings_count=20)))
    _, summary=report.calculate_tables(pd.concat(frames,ignore_index=True))
    value=summary.loc[summary.period.eq("all"),"mean_cagr"].iloc[0]
    assert np.isclose(value,.5)
    assert not np.isclose(value,2.5**.5-1)


def test_public_daily_reproduces_every_audited_period():
    root=Path(__file__).resolve().parents[1]
    daily=pd.read_csv(root/"reports/current_strategy/phase_daily.csv",parse_dates=["datetime"],float_precision="round_trip")
    report.validate_daily(daily)
    _, summary=report.calculate_tables(daily)
    report.verify_summary(summary)
    full=summary.loc[summary.period.eq("all")].iloc[0]
    assert np.isclose(full.mean_cagr,.18846122974183205)
    assert np.isclose(full.worst_drawdown,.3034725047592275)
    assert np.isclose(full.mean_sharpe,1.0001958154384787)
    assert np.isclose(full.mean_information_ratio,1.5030796428206568)


def test_effective_risk_free_adjustment_leaves_index_relative_ir_unchanged():
    net=pd.Series([.001,-.002,.003,.0005,-.001])
    benchmark=pd.Series([.0001,-.001,.002,-.0005,.0002])
    zero=report.risk_adjusted_metrics(net,benchmark,0.)
    one=report.risk_adjusted_metrics(net,benchmark,.01)
    daily=np.expm1(np.log1p(.01)/252)
    assert np.isclose((1+daily)**252,1.01)
    expected=(net-daily).mean()/(net-daily).std(ddof=1)*np.sqrt(252)
    assert np.isclose(one['sharpe'],expected)
    assert one['sharpe']<zero['sharpe']
    assert one['information_ratio']==zero['information_ratio']


def test_public_evidence_excludes_stock_and_broker_records():
    daily=pd.read_csv(report.OUT/"phase_daily.csv",nrows=1)
    assert daily.columns.tolist()==report.COLUMNS
    assert not {"account_id","account_suffix","instrument","order_id","model_path"} & set(daily.columns)
    evidence=report.read_json(report.OUT/"evidence.json")
    assert b"\r" not in (report.OUT/"phase_daily.csv").read_bytes()
    assert report.digest(report.OUT/"phase_daily.csv")==evidence["public_daily_sha256"]
    assert evidence["strategy_id"]=="S6_head6_40_QREG"
    assert all(not Path(path).is_absolute() and ":" not in path for path in evidence["source_sha256"])
