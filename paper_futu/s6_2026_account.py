"""Fresh-cash 2026 accounts using the unchanged audited exchange and strategy."""
from __future__ import annotations

from functools import partial
import importlib
import os
from unittest.mock import patch

from paper_futu import research_s6_2026 as study

_EXCHANGE = None


def init_qlib():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=str(study.DATA / "provider"), region=REG_CN, kernels=1,
        exp_manager={"class":"MLflowExpManager", "module_path":"qlib.workflow.expm",
            "kwargs":{"uri":(study.OUT / "mlruns").as_uri(), "default_exp_name":study.NAME}})


def schedule_dates(calendar, start, end, offset):
    """The account interval stays fixed when the first rebalance is delayed."""
    dates = calendar[(calendar >= start) & (calendar <= end)]
    trades = dates[offset::5]
    return dates, trades, calendar[calendar.get_indexer(trades)-1]


def replay(case, offset, mode="fresh"):
    global _EXCHANGE
    import numpy as np
    import pandas as pd
    from qlib.data import D
    from qlib.backtest.decision import Order
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.evaluate import backtest_daily
    from paper_futu.final_ablation_account import AuditedExchange, AuditedStrategy
    from paper_futu.overfit_four_core import verify_account
    from src.compare_models import enrich_daily_report, summarize_report
    from src.runtime_budget import require_memory

    require_memory(2., "S6 fresh 2026 account")
    p = study.protocol()
    stream_name = case if mode == "fresh" else f"history_{case}"
    stream = study.read(study.OUT / "streams.json")["streams"][stream_name]
    if study.digest(stream["path"]) != stream["sha256"]:
        raise ValueError("Account prediction hash changed")
    prediction = pd.read_parquet(stream["path"])
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    start = study.START if mode == "fresh" else study.HISTORY_START
    days, scheduled, signal_dates = schedule_dates(calendar,start,study.END,offset)
    prior_day = calendar[calendar < days[0]][-1]
    signal = prediction.loc[prediction.index.get_level_values("datetime").isin(signal_dates), ["prediction"]]
    if len(signal_dates) != signal.index.get_level_values("datetime").nunique():
        raise ValueError("Missing a scheduled prior-close signal")
    if _EXCHANGE is None:
        index = pd.read_parquet(study.DATA / "raw_features.parquet", columns=[]).index
        _EXCHANGE = AuditedExchange(freq="day", start_time=prior_day, end_time=study.END,
            codes=sorted(index.get_level_values("instrument").unique()), deal_price="open",
            open_cost=.0005, close_cost=.0015, min_cost=5., trade_unit=100,
            limit_threshold=("($open / Ref($close, 1) - 1) >= 0.095", "($open / Ref($close, 1) - 1) <= -0.095"))
    market = _EXCHANGE
    market.reset_audit(0., study.PORTFOLIO["top_k"])
    strategy = AuditedStrategy(signal=signal, risk_degree=.95, only_tradable=True, **study.PORTFOLIO)
    module = importlib.import_module("qlib.backtest.backtest")
    with patch.object(module, "tqdm", partial(module.tqdm, disable=True)):
        raw, positions = backtest_daily(start_time=prior_day, end_time=study.END,
            strategy=strategy, executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True,
                verbose=False, trade_type=SimulatorExecutor.TT_SERIAL), account=1_000_000,
            benchmark="SH000300", exchange_kwargs={"exchange":market})
    report = enrich_daily_report(raw.reindex(days),f"{mode}_{case}_{offset}")
    if report[["net_return", "cost", "account", "cash"]].isna().any().any():
        raise ValueError("Fresh account does not cover the common comparison interval")
    benchmark = D.features(["SH000300"], ["$open", "$close"], start_time=days[0], end_time=days[0]).iloc[0]
    report.loc[days[0], "bench"] = benchmark.iloc[1]/benchmark.iloc[0]-1
    report.loc[study.END, "bench"] = study.read(study.DATA / "benchmark_audit.json")["last_daily_return"]
    if report.bench.isna().any():
        raise ValueError("Missing common benchmark return")
    report["active_return"] = report.net_return-report.bench
    report["benchmark_nav"] = (1+report.bench).cumprod()
    report["relative_excess_nav"] = report.portfolio_nav/report.benchmark_nav
    prior = report.account.shift(1, fill_value=1_000_000.)
    orders = pd.DataFrame(market.journal)
    trades = orders.loc[orders.notional.gt(0)]
    if not trades.datetime.isin(scheduled).all():
        raise ValueError("Trade occurred outside the fixed rebalance schedule")
    totals = trades.groupby("datetime")[["notional", "fee"]].sum().reindex(days, fill_value=0.)
    report["fees_cash"], report["traded_notional"] = totals.fee, totals.notional
    report["turnover"] = totals.notional/prior
    np.testing.assert_allclose(report.cost*prior, totals.fee, atol=1e-5, rtol=1e-10)
    holdings, contributions, previous_values = [], [], {}
    for day in days:
        position = positions[day]
        market.check_holding_limit(position, day, "daily_close")
        current_values = {}
        for stock in position.get_stock_list():
            quantity, price = position.get_stock_amount(stock), position.get_stock_price(stock)
            current_values[stock] = quantity*price
            holdings.append(dict(datetime=day, instrument=stock, adjusted_quantity=quantity,
                adjusted_close=price, market_value=quantity*price, cash=position.get_cash()))
        today = trades.loc[trades.datetime.eq(day)]
        flow = today.assign(flow=np.where(today.direction.eq(Order.BUY), today.notional, -today.notional)).groupby("instrument").flow.sum()
        for stock in set(previous_values) | set(current_values) | set(flow.index):
            contributions.append(dict(datetime=day, instrument=stock,
                pnl_before_fees=current_values.get(stock,0)-previous_values.get(stock,0)-flow.get(stock,0)))
        previous_values = current_values
        if position.get_cash() < -1e-5:
            raise ValueError("Negative cash")
    holding = pd.DataFrame(holdings)
    pnl = pd.DataFrame(contributions)
    report["holdings_count"] = holding.groupby("datetime").size().reindex(days, fill_value=0)
    values = holding.groupby("datetime").market_value.sum().reindex(days, fill_value=0)
    np.testing.assert_allclose(values+report.cash, report.account, atol=1e-4, rtol=1e-10)
    gross = pnl.groupby("datetime").pnl_before_fees.sum().reindex(days, fill_value=0)
    np.testing.assert_allclose(prior+gross-totals.fee, report.account, atol=1e-4, rtol=1e-10)
    np.testing.assert_allclose((1+report.net_return).cumprod()*1e6, report.account, atol=1e-4, rtol=1e-10)
    if offset:
        np.testing.assert_array_equal(report.account.iloc[:offset], np.full(offset, 1e6))
        assert report.holdings_count.iloc[:offset].eq(0).all()
    directory = study.OUT / "accounts" / f"{mode}_{case}_{offset}"
    directory.mkdir(parents=True, exist_ok=True)
    report.to_parquet(directory / "daily.parquet")
    orders.to_parquet(directory / "orders.parquet", index=False)
    holding.to_parquet(directory / "holdings.parquet", index=False)
    pnl.to_parquet(directory / "contributions.parquet", index=False)
    pd.DataFrame(market.generation_journal, columns=["datetime","instrument","direction","reason"]).to_parquet(
        directory / "generation_diagnostics.parquet", index=False)
    row = summarize_report(report,f"{mode}_{case}_{offset}").iloc[0].to_dict()
    row.update(period="all",mode=mode,case=case,offset=offset,start=str(days[0].date()),end=str(days[-1].date()),
        total_fees_cny=float(totals.fee.sum()), average_daily_turnover=float(report.turnover.mean()),
        average_holdings=float(report.holdings_count.mean()), minimum_holdings=int(report.holdings_count.min()),
        maximum_holdings=int(report.holdings_count.max()))
    pd.DataFrame([row]).to_csv(directory / "metrics.csv", index=False, encoding="utf-8-sig")
    audit = verify_account(directory)
    result = dict(mode=mode,case=case,offset=offset,directory=str(directory),audit=audit,
        prediction_sha256=stream["sha256"], initial_cash=1_000_000,
        rebalance_dates=[str(d.date()) for d in scheduled], signal_dates=[str(d.date()) for d in signal_dates],
        first_trade=str(trades.datetime.min().date()), metrics=row)
    return result, [path for path in directory.iterdir() if path.is_file()]
