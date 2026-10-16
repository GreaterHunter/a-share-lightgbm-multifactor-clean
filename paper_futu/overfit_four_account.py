"""Same audited execution engine, with an explicit independent output directory."""
from __future__ import annotations

from functools import partial
import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd

from paper_futu import final_ablation_core as frozen
from paper_futu.final_ablation_account import AuditedExchange, AuditedStrategy, summarize_accounts
from paper_futu.overfit_four_core import DATA, OUT, Store, read, verify_account
from src.compare_models import enrich_daily_report
from src.runtime_budget import require_memory

_EXCHANGE = None


def replay(job, key):
    global _EXCHANGE
    from qlib.data import D
    from qlib.backtest.decision import Order
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.evaluate import backtest_daily

    require_memory(2., "四策略独立账户回放")
    if job["variant"] == "V0" and job["seed"] == 42 and job["friction"] == 0:
        old = next(a for a in read(OUT / "protocol.json")["baselines"][job["strategy"]]["accounts"]
                   if a["offset"] == job["offset"])
        directory = __import__("pathlib").Path(old["directory"])
        verify_account(directory)
        return dict(directory=str(directory), source_reused=True, rows=pd.read_csv(directory/"metrics.csv").to_dict("records")), list(directory.iterdir())

    prediction = pd.read_parquet(job["prediction_path"])
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    trade_dates = calendar[(calendar >= frozen.START) & (calendar <= frozen.AS_OF)]
    schedule = trade_dates[job["offset"]::job["rebalance"]]
    signal_dates = calendar[calendar.get_indexer(schedule)-1]
    signal = prediction.loc[prediction.index.get_level_values("datetime").isin(signal_dates), ["prediction"]]
    if len(signal_dates) != signal.index.get_level_values("datetime").nunique():
        raise ValueError("Missing scheduled signals")
    if _EXCHANGE is None:
        schema = pd.read_parquet(DATA / "raw_features.parquet", columns=[])
        codes = sorted(schema.index.get_level_values("instrument").unique())
        _EXCHANGE = AuditedExchange(freq="day", start_time=calendar[calendar < frozen.START][-1],
            end_time=frozen.AS_OF, codes=codes, deal_price="open", open_cost=.0005, close_cost=.0015,
            min_cost=5, trade_unit=100, limit_threshold=(
                "($open / Ref($close, 1) - 1) >= 0.095", "($open / Ref($close, 1) - 1) <= -0.095"))
    market = _EXCHANGE
    market.reset_audit(job["friction"], job["portfolio"]["top_k"])
    strategy = AuditedStrategy(signal=signal, risk_degree=.95, only_tradable=True, **job["portfolio"])
    module = importlib.import_module("qlib.backtest.backtest")
    with patch.object(module, "tqdm", partial(module.tqdm, disable=True)):
        raw, positions = backtest_daily(start_time=calendar[calendar < frozen.START][-1], end_time=frozen.AS_OF,
            strategy=strategy, executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True,
                verbose=False, trade_type=SimulatorExecutor.TT_SERIAL), account=1_000_000,
            benchmark="SH000300", exchange_kwargs={"exchange": market})
    report = enrich_daily_report(raw.reindex(trade_dates), key)
    if report[["net_return", "cost", "account"]].isna().any().any():
        raise ValueError("Missing common comparison interval")
    opening = D.features(["SH000300"], ["$open", "$close"], start_time=trade_dates[0], end_time=trade_dates[0]).iloc[0]
    report.loc[trade_dates[0], "bench"] = opening.iloc[1]/opening.iloc[0]-1
    report.loc[frozen.AS_OF, "bench"] = read(DATA / "benchmark_audit.json")["last_daily_return"]
    if report.bench.isna().any():
        raise ValueError("Missing benchmark")
    report["active_return"] = report.net_return-report.bench
    report["benchmark_nav"] = (1+report.bench).cumprod()
    report["relative_excess_nav"] = report.portfolio_nav/report.benchmark_nav
    prior = report.account.shift(1, fill_value=1_000_000.)
    orders = pd.DataFrame(market.journal)
    trades = orders.loc[orders.notional.gt(0)]
    totals = trades.groupby("datetime")[["notional", "fee"]].sum().reindex(trade_dates, fill_value=0)
    report["fees_cash"], report["traded_notional"] = totals.fee, totals.notional
    report["turnover"] = totals.notional/prior
    np.testing.assert_allclose(report.cost*prior, totals.fee, atol=1e-5, rtol=1e-10)
    holdings, contributions, previous_values = [], [], {}
    for day in trade_dates:
        position = positions[day]
        market.check_holding_limit(position, day, "daily_close")
        current_values = {}
        for stock in position.get_stock_list():
            quantity, price = position.get_stock_amount(stock), position.get_stock_price(stock)
            value = quantity*price
            current_values[stock] = value
            holdings.append(dict(datetime=day, instrument=stock, adjusted_quantity=quantity,
                                 adjusted_close=price, market_value=value, cash=position.get_cash()))
        today = trades.loc[trades.datetime.eq(day)]
        flows = today.assign(flow=np.where(today.direction.eq(Order.BUY), today.notional, -today.notional)).groupby("instrument").flow.sum()
        for stock in set(previous_values) | set(current_values) | set(flows.index):
            contributions.append(dict(datetime=day, instrument=stock,
                pnl_before_fees=current_values.get(stock, 0)-previous_values.get(stock, 0)-flows.get(stock, 0)))
        previous_values = current_values
        if position.get_cash() < -1e-5:
            raise ValueError("Negative account cash")
    holding_frame = pd.DataFrame(holdings)
    pnl_frame = pd.DataFrame(contributions)
    report["holdings_count"] = holding_frame.groupby("datetime").size().reindex(trade_dates, fill_value=0)
    gross_pnl = pnl_frame.groupby("datetime").pnl_before_fees.sum().reindex(trade_dates, fill_value=0)
    np.testing.assert_allclose(prior+gross_pnl-totals.fee, report.account, atol=1e-4, rtol=1e-10)
    np.testing.assert_allclose((1+report.net_return).cumprod()*1e6, report.account, atol=1e-4, rtol=1e-10)
    rows = summarize_accounts(report, key)
    year_rows = [r for r in rows if r["period"].isdigit()]
    all_row = next(r for r in rows if r["period"] == "all")
    if not np.isclose(np.prod([1+r["total_return"] for r in year_rows])-1, all_row["total_return"], atol=1e-10):
        raise ValueError("Annual returns do not compound")
    directory = OUT / "accounts" / key
    directory.mkdir(parents=True, exist_ok=True)
    report.to_parquet(directory/"daily.parquet")
    orders.to_parquet(directory/"orders.parquet", index=False)
    pd.DataFrame(market.generation_journal, columns=["datetime", "instrument", "direction", "reason"]).to_parquet(
        directory/"generation_diagnostics.parquet", index=False)
    holding_frame.to_parquet(directory/"holdings.parquet", index=False)
    pnl_frame.to_parquet(directory/"contributions.parquet", index=False)
    for row in rows:
        row.update(case_id=key, **job)
    pd.DataFrame(rows).to_csv(directory/"metrics.csv", index=False, encoding="utf-8-sig")
    verify_account(directory)
    return dict(directory=str(directory), source_reused=False, rows=rows), list(directory.iterdir())
