"""Isolated S6 fixed-cost 20% close-trigger/next-open stop-loss comparison.

Run --stage all. This never imports the broker execution entry points.
"""
from __future__ import annotations

import argparse
import copy
from functools import partial
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/s6_quarterly_l1l2_small_asof_20261013"
DATA = ROOT / "data/processed/final_ablation_asof_20261013"
OUT = ROOT / "reports/experiments/s6_stop20_asof_20261013"
START, END = "2022-01-04", "2026-10-13"
ENV = dict(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
           NUMEXPR_NUM_THREADS="4", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


class CostBook:
    """Moving average cost including purchase commissions; adjusted units."""
    def __init__(self):
        self.positions = {}

    def fill(self, stock, direction, quantity, value, commission):
        if value <= 0:
            return
        old_q, old_cost = self.positions.get(stock, (0., 0.))
        if direction == 1:
            self.positions[stock] = (old_q+quantity, old_cost+value+commission)
        else:
            if old_q <= 0 or quantity > old_q+max(1e-7, old_q*1e-10):
                raise ValueError("Stop cost ledger oversold")
            fraction = max(0., 1-quantity/old_q)
            self.positions[stock] = (old_q*fraction, old_cost*fraction)

    def return_at(self, stock, close):
        import math
        qty, cost = self.positions.get(stock, (0., 0.))
        if qty <= 1e-7 or cost <= 0 or close is None or not math.isfinite(close) or close <= 0:
            return None
        return qty*close/cost-1


def classes():
    import numpy as np
    import pandas as pd
    from qlib.backtest.decision import Order, TradeDecisionWO
    from paper_futu.final_ablation_account import AuditedExchange, AuditedStrategy

    class StopExchange(AuditedExchange):
        def reset_audit(self, friction, top_k=None):
            super().reset_audit(friction, top_k)
            self.book = CostBook()

        def deal_order(self, order, *args, **kwargs):
            result = super().deal_order(order, *args, **kwargs)
            value, commission, _ = result
            self.book.fill(order.stock_id, int(order.direction), order.deal_amount, value, commission)
            self.journal[-1]["stop_loss"] = bool(getattr(order, "stop_loss", False))
            return result

    class StopStrategy(AuditedStrategy):
        def __init__(self, threshold=None, **kwargs):
            super().__init__(**kwargs)
            self.threshold = threshold
            self.pending = {}
            self.stop_events = []

        def generate_trade_decision(self, execute_result=None):
            if self.trade_calendar.get_trade_step() % 50 == 0:
                from src.runtime_budget import require_memory
                require_memory(1., "S6 single-process replay safety floor")
            if self.threshold is None:
                return super().generate_trade_decision(execute_result)
            step = self.trade_calendar.get_trade_step()
            start, end = self.trade_calendar.get_step_time(step)
            prior_start, prior_end = self.trade_calendar.get_step_time(step, shift=1)
            current = copy.deepcopy(self.trade_position)
            held = set(current.get_stock_list())
            self.pending = {s: v for s, v in self.pending.items() if s in held}
            for stock in sorted(held):
                qty = self.trade_exchange.book.positions.get(stock, (0., 0.))[0]
                if not np.isclose(qty, current.get_stock_amount(stock), atol=1e-6, rtol=1e-10):
                    raise ValueError("Position and stop cost ledger differ")
                close = self.trade_exchange.get_close(stock, prior_start, prior_end)
                ret = self.trade_exchange.book.return_at(stock, close)
                if ret is not None and ret <= -self.threshold+1e-12 and stock not in self.pending:
                    event = dict(instrument=stock, signal_date=pd.Timestamp(prior_start).normalize(),
                                 execution_date=pd.Timestamp(start).normalize(), trigger_return=ret,
                                 quantity=qty, average_cost=self.trade_exchange.book.positions[stock][1]/qty,
                                 observed_close=close)
                    assert event["signal_date"] < event["execution_date"]
                    self.stop_events.append(event)
                    self.pending[stock] = event
            score = self.signal.get_signal(start_time=prior_start, end_time=prior_end)
            if score is None:
                # Stop-only day: leave every other position exactly unchanged.
                target = {s: q for s, q in current.get_stock_amount_dict().items() if s not in self.pending}
                orders = self.trade_exchange.generate_order_for_target_amount_position(
                    target, current.get_stock_amount_dict(), start, end) if self.pending else []
            else:
                # Pending stops cannot be topped up or bought back on this day.
                filtered = score.drop(index=list(self.pending), errors="ignore")
                weights = self.generate_target_weight_position(filtered, current, start, end)
                weights = {s: w for s, w in weights.items() if s not in self.pending}
                orders = self.order_generator.generate_order_list_from_target_weight_position(
                    current=current, trade_exchange=self.trade_exchange,
                    risk_degree=self.get_risk_degree(step), target_weight_position=weights,
                    pred_start_time=prior_start, pred_end_time=prior_end,
                    trade_start_time=start, trade_end_time=end)
            for order in orders:
                order.stop_loss = order.stock_id in self.pending
                if order.stop_loss and order.direction != Order.SELL:
                    raise ValueError("Stop generated an unexpected buy")
            return TradeDecisionWO(orders, self)

    return StopExchange, StopStrategy


def audit():
    streams = read(SOURCE/"streams.json")["streams"]["history_QREG"]
    expected = {str(SOURCE/"protocol.json"): digest(SOURCE/"protocol.json"),
                streams["path"]: streams["sha256"], str(DATA/"manifest.json"): digest(DATA/"manifest.json")}
    for offset in range(5):
        path = SOURCE/f"tasks/account_history_QREG_{offset}.json"
        task = read(path)
        assert task["status"] == "complete"
        expected[str(path)] = digest(path)
        expected.update(task["artifacts"])
    for name in ("final_ablation_account.py", "s6_explicit_account.py", "research_s6_quarterly_regularized.py"):
        path = ROOT/"paper_futu"/name
        expected[str(path)] = digest(path)
    for name in ("portfolio_strategy.py", "compare_models.py"):
        path = ROOT/"src"/name
        expected[str(path)] = digest(path)
    for name, value in read(DATA/"manifest.json")["artifacts"].items():
        if name.startswith("provider") or name == "benchmark_audit.json":
            expected[str(DATA/name)] = value
    for path, value in expected.items():
        if digest(path) != value:
            raise ValueError(f"Source changed: {path}")
    protocol = dict(strategy="S6_head6_40_QREG", threshold=.20, initial_cash=1e6,
                    detection="previous completed close / moving average purchase cost including buy fees - 1",
                    action="sell entire position at next tradable open; pending exits persist after a rebound",
                    reinvestment="cash until next normal rebalance; no same-day stop reentry",
                    scheduled_rebalance_days=5, offsets=list(range(5)),
                    stop_overrides_replacement_limit=True, ordinary_n_drop=6,
                    top_k=20, head_size=6, head_weight=.4, risk_degree=.95,
                    buy_cost=.0005, sell_cost=.0015, minimum_fee=5,
                    sizing="original sealed historical sizing", daily_limit_filter="uniform +/-9.5%",
                    implementation_sha256=digest(Path(__file__)), sources=expected,
                    runtime=dict(max_workers=1, startup_free_gib=1.5, task_free_gib=1.0,
                                 authorization="2026-10-23 user requested direct execution despite the 3 GiB startup block"),
                    start=START, end=END)
    path = OUT/"protocol.json"
    if path.exists() and read(path) != protocol:
        if any((OUT/"accounts").glob("*/complete.json")):
            raise ValueError("Experiment protocol changed; do not overwrite completed caches")
        save(OUT/"pre_execution_protocols"/f"{digest(path)}.json", read(path))
    save(path, protocol)
    print("AUDIT passed", flush=True)


def metrics(daily):
    import numpy as np
    net = daily.net_return.to_numpy()
    wealth = np.cumprod(1+net)
    drawdown = wealth/np.maximum.accumulate(np.r_[1., wealth])[1:]-1
    return dict(total_return=float(wealth[-1]-1), cagr=float(wealth[-1]**(252/len(net))-1),
                max_drawdown=float(-drawdown.min()), fees=float(daily.fees_cash.sum()),
                mean_daily_turnover=float(daily.turnover.mean()), mean_holdings=float(daily.holdings_count.mean()))


def account(mode, offset):
    from src.runtime_budget import require_memory
    require_memory(1.5, "S6 stop-loss account startup")
    for key, value in ENV.items():
        os.environ[key] = value
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from paper_futu.s6_explicit_account import init_qlib, schedule_dates
    from src.compare_models import enrich_daily_report
    from qlib.data import D
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.decision import Order
    from qlib.contrib.evaluate import backtest_daily
    p = read(OUT/"protocol.json")
    if p["implementation_sha256"] != digest(Path(__file__)):
        raise ValueError("Stop-loss implementation differs from protocol")
    name = f"{mode}_{offset}"
    directory = OUT/"accounts"/name
    complete = directory/"complete.json"
    if complete.exists():
        record = read(complete)
        assert record["protocol_sha256"] == digest(OUT/"protocol.json")
        for path, expected in record["artifacts"].items():
            assert digest(path) == expected
        return
    init_qlib(SimpleNamespace(DATA=DATA, OUT=OUT, NAME=OUT.name))
    Market, Strategy = classes()
    stream = read(SOURCE/"streams.json")["streams"]["history_QREG"]
    assert digest(stream["path"]) == stream["sha256"]
    prediction = pd.read_parquet(stream["path"], columns=["prediction"])
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    days, scheduled, signal_dates = schedule_dates(calendar, START, END, offset)
    prior_day = calendar[calendar < days[0]][-1]
    signal = prediction.loc[prediction.index.get_level_values("datetime").isin(signal_dates)]
    assert signal.index.get_level_values("datetime").nunique() == len(signal_dates)
    market = Market(freq="day", start_time=prior_day, end_time=END,
                    codes=sorted(prediction.index.get_level_values("instrument").unique()),
                    deal_price="open", open_cost=.0005, close_cost=.0015, min_cost=5., trade_unit=100,
                    limit_threshold=("($open / Ref($close, 1) - 1) >= 0.095", "($open / Ref($close, 1) - 1) <= -0.095"))
    require_memory(1., "S6 stop-loss account loaded")
    market.reset_audit(0., 20)
    strategy = Strategy(signal=signal, risk_degree=.95, only_tradable=True, top_k=20,
                        top_tier_size=6, top_tier_weight=.4, n_drop=6,
                        threshold=.2 if mode == "stop20" else None)
    module = importlib.import_module("qlib.backtest.backtest")
    with patch.object(module, "tqdm", partial(module.tqdm, disable=True)):
        raw, positions = backtest_daily(start_time=prior_day, end_time=END, strategy=strategy,
            executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True,
                verbose=False, trade_type=SimulatorExecutor.TT_SERIAL), account=1_000_000,
            benchmark="SH000300", exchange_kwargs={"exchange":market})
    report = enrich_daily_report(raw.reindex(days), name)
    orders = pd.DataFrame(market.journal)
    trades = orders.loc[orders.notional.gt(0)]
    assert trades.loc[~trades.datetime.isin(scheduled), "stop_loss"].all()
    assert trades.loc[trades.stop_loss, "direction"].eq(Order.SELL).all()
    totals = trades.groupby("datetime")[["notional", "fee"]].sum().reindex(days, fill_value=0.)
    prior = report.account.shift(1, fill_value=1e6)
    report["fees_cash"], report["traded_notional"] = totals.fee, totals.notional
    report["turnover"] = totals.notional/prior
    holding, contributions, previous = [], [], {}
    for day in days:
        pos = positions[day]
        market.check_holding_limit(pos, day, "stop_close")
        current = {}
        for stock in pos.get_stock_list():
            qty, price = pos.get_stock_amount(stock), pos.get_stock_price(stock)
            current[stock] = qty*price
            holding.append(dict(datetime=day, instrument=stock, adjusted_quantity=qty,
                                adjusted_close=price, market_value=qty*price, cash=pos.get_cash()))
        today = trades.loc[trades.datetime.eq(day)]
        flow = today.assign(flow=np.where(today.direction.eq(Order.BUY), today.notional, -today.notional)).groupby("instrument").flow.sum()
        for stock in set(previous)|set(current)|set(flow.index):
            contributions.append(dict(datetime=day, instrument=stock,
                pnl_before_fees=current.get(stock,0)-previous.get(stock,0)-flow.get(stock,0)))
        previous = current
    holding = pd.DataFrame(holding)
    pnl = pd.DataFrame(contributions)
    report["holdings_count"] = holding.groupby("datetime").size().reindex(days, fill_value=0)
    values = holding.groupby("datetime").market_value.sum().reindex(days, fill_value=0)
    gross = pnl.groupby("datetime").pnl_before_fees.sum().reindex(days, fill_value=0)
    for actual, expected in ((values+report.cash, report.account),
                             (prior+gross-totals.fee, report.account),
                             ((1+report.net_return).cumprod()*1e6, report.account),
                             (report.cost*prior, totals.fee)):
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-10)
    assert report.cash.min() >= -1e-5
    if offset:
        np.testing.assert_array_equal(report.account.iloc[:offset], np.full(offset, 1e6))
    baseline = pd.read_parquet(SOURCE/f"accounts/history_QREG_{offset}/daily.parquet")
    if mode == "baseline":
        for column in ("account", "cash", "net_return", "fees_cash", "turnover", "holdings_count"):
            np.testing.assert_allclose(report[column], baseline[column], atol=1e-5, rtol=1e-10)
        old_orders = pd.read_parquet(SOURCE/f"accounts/history_QREG_{offset}/orders.parquet")
        pd.testing.assert_frame_equal(orders[old_orders.columns], old_orders, check_exact=True)
    else:
        if not (OUT/f"accounts/baseline_{offset}/complete.json").exists():
            raise ValueError("Paired baseline has not passed parity")
    # Match the historical benchmark for comparison; the study focuses on
    # strategy return/risk, without changing the benchmark attribution series.
    report["bench"] = baseline.bench
    report["active_return"] = report.net_return-report.bench
    report["benchmark_nav"] = baseline.benchmark_nav
    report["relative_excess_nav"] = report.portfolio_nav/report.benchmark_nav
    periods = {"all":metrics(report), **{str(y):metrics(g) for y,g in report.groupby(report.index.year)}}
    stop_fills = trades.loc[trades.stop_loss]
    directory.mkdir(parents=True, exist_ok=True)
    for filename, frame in (("daily", report), ("orders", orders), ("holdings", holding), ("contributions", pnl)):
        frame.to_parquet(directory/f"{filename}.parquet", index=filename=="daily")
    save(directory/"stop_events.json", strategy.stop_events)
    save(directory/"generation_diagnostics.json", market.generation_journal)
    record = dict(mode=mode, offset=offset, protocol_sha256=digest(OUT/"protocol.json"), periods=periods,
                  triggers=len(strategy.stop_events), stop_sell_fills=len(stop_fills),
                  stop_notional=float(stop_fills.notional.sum()),
                  unresolved_stops=sorted(set(strategy.pending)&set(positions[days[-1]].get_stock_list())),
                  baseline_parity_passed=mode=="baseline", account_audit_passed=True,
                  artifacts={str(f):digest(f) for f in directory.iterdir() if f.is_file()})
    save(complete, record)
    print(f"COMPLETE {name}: {json.dumps(periods['all'])}; stop_fills={len(stop_fills)}", flush=True)


def report_results():
    import numpy as np
    import pandas as pd
    from scripts.analyze_s6_realized_sales import reconstruct_sales, stock_date_sales
    p = read(OUT/"protocol.json")
    for path, expected in p["sources"].items():
        if digest(path) != expected:
            raise ValueError(f"Source changed during experiment: {path}")
    cases = {}
    records = []
    for mode in ("baseline", "stop20"):
        entries = []
        for offset in range(5):
            directory = OUT/f"accounts/{mode}_{offset}"
            entry = read(directory/"complete.json")
            assert entry["protocol_sha256"] == digest(OUT/"protocol.json")
            for path, expected in entry["artifacts"].items():
                assert digest(path) == expected
            orders = pd.read_parquet(directory/"orders.parquet")
            fills, _ = reconstruct_sales(orders)
            sales = stock_date_sales(fills, pd.read_parquet(directory/"holdings.parquet"))
            winners, losers = sales.loc[sales.net_class.eq("profit")], sales.loc[sales.net_class.eq("loss")]
            entry["trade_metrics"] = dict(win_rate=len(winners)/len(sales), mean_win=float(winners.net_return.mean()),
                mean_loss=float(losers.net_return.mean()), worst_sale=float(sales.net_return.min()),
                below_minus20=int(sales.net_return.lt(-.2).sum()))
            entries.append(entry)
        summary = {}
        for period in entries[0]["periods"]:
            rows = [e["periods"][period] for e in entries]
            summary[period] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            summary[period]["worst_phase_drawdown"] = max(r["max_drawdown"] for r in rows)
        cases[mode] = dict(periods=summary, mean_stop_fills=float(np.mean([e["stop_sell_fills"] for e in entries])),
                           trade_metrics={key:float(np.mean([e["trade_metrics"][key] for e in entries]))
                                          for key in entries[0]["trade_metrics"]})
        records.extend(entries)
    save(OUT/"comparison.json", dict(status="complete", protocol_sha256=digest(OUT/"protocol.json"),
                                    source_hashes_unchanged=True, cases=cases, accounts=records))
    print(json.dumps(cases), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["audit", "account", "report", "all"], default="all")
    parser.add_argument("--mode", choices=["baseline", "stop20"], default="stop20")
    parser.add_argument("--offset", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    if args.stage == "audit": audit()
    elif args.stage == "account": account(args.mode, args.offset)
    elif args.stage == "report": report_results()
    else:
        audit()
        for offset in range(5):
            for mode in ("baseline", "stop20"):
                save(OUT/"status.json", dict(state="running", mode=mode, offset=offset, pid=os.getpid(), updated=time.time()))
                result = subprocess.run([sys.executable,"-u","-m","paper_futu.research_s6_stop_loss",
                    "--stage","account","--mode",mode,"--offset",str(offset)], cwd=ROOT, env={**os.environ, **ENV})
                if result.returncode:
                    save(OUT/"status.json", dict(state="failed", mode=mode, offset=offset, exit_code=result.returncode))
                    raise SystemExit(result.returncode)
        report_results()
        save(OUT/"status.json", dict(state="complete", updated=time.time()))


if __name__ == "__main__":
    main()
