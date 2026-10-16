"""Offline execution sensitivity: sell at next close, buy at following open.

The signal is observed at T close. Existing positions are reduced at T+1
close; only then are the released funds used for additions at T+2 open.
This intentionally does not alter the live paper-trading configuration.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import qlib
from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.exchange import Exchange
from qlib.config import REG_CN
from qlib.contrib.evaluate import backtest_daily
from qlib.data import D

from paper_futu.backtest_correct_open_grid_2022_2026 import END, PROVIDER, REPORT as BASELINE_REPORT
from src.compare_models import build_rebalance_signal, enrich_daily_report, summarize_report
from src.config import PROJECT_ROOT
from src.portfolio_strategy import TieredTopKWeightStrategy


OUTPUT = PROJECT_ROOT / "reports/experiments/staged_close_sell_next_open_buy_2022_2026"


class StagedSellThenBuyStrategy(TieredTopKWeightStrategy):
    """Use one fully funded sell day followed by one buy day per rebalance.

    On initial entry there is no sell to finance, so the first basket is bought
    at T+1 open. Subsequent reductions occur at T+1 close and additions at
    T+2 open. Survivors are not deliberately liquidated.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pending_target: dict[str, float] | None = None
        self.events: list[dict] = []

    def _target_orders(self, target: dict[str, float], current, step: int) -> list[Order]:
        trade_start, trade_end = self.trade_calendar.get_step_time(step)
        pred_start, pred_end = self.trade_calendar.get_step_time(step, shift=1)
        return self.order_generator.generate_order_list_from_target_weight_position(
            current=current,
            trade_exchange=self.trade_exchange,
            risk_degree=self.get_risk_degree(step),
            target_weight_position=target,
            pred_start_time=pred_start,
            pred_end_time=pred_end,
            trade_start_time=trade_start,
            trade_end_time=trade_end,
        )

    def generate_trade_decision(self, execute_result=None):
        step = self.trade_calendar.get_trade_step()
        trade_start, trade_end = self.trade_calendar.get_step_time(step)
        current = deepcopy(self.trade_position)

        if self.pending_target is not None:
            target = self.pending_target
            self.pending_target = None
            orders = [order for order in self._target_orders(target, current, step)
                      if order.direction == Order.BUY]
            self.events.append({"date": trade_start, "phase": "funded_buy",
                                "orders": len(orders), "cash_before": current.get_cash()})
            return TradeDecisionWO(orders, self)

        pred_start, pred_end = self.trade_calendar.get_step_time(step, shift=1)
        score = self.signal.get_signal(start_time=pred_start, end_time=pred_end)
        if score is None:
            return TradeDecisionWO([], self)
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        target = self.generate_target_weight_position(
            score=score, current=current,
            trade_start_time=trade_start, trade_end_time=trade_end,
        )
        orders = self._target_orders(target, current, step)
        if not current.get_stock_list():
            orders = [order for order in orders if order.direction == Order.BUY]
            phase = "initial_buy"
        else:
            orders = [order for order in orders if order.direction == Order.SELL]
            self.pending_target = target
            phase = "close_sell"
        self.events.append({"date": trade_start, "phase": phase,
                            "orders": len(orders), "cash_before": current.get_cash()})
        return TradeDecisionWO(orders, self)


def run(horizon: int, rebalance: int) -> None:
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    name = f"h{horizon}_open_to_open_r{rebalance}_open"
    prediction = pd.read_parquet(BASELINE_REPORT / f"h{horizon}_open_to_open_predictions.parquet")
    baseline = pd.read_csv(BASELINE_REPORT / f"{name}_daily.csv", index_col="datetime", parse_dates=True)
    signal, dates = build_rebalance_signal(prediction, rebalance)
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    first_trade = calendar[calendar > dates[0]].min()
    strategy = StagedSellThenBuyStrategy(
        signal=signal, top_k=10, n_drop=3, top_tier_size=2,
        top_tier_weight=0.4, risk_degree=0.95, only_tradable=True,
    )
    codes = sorted(set(prediction.index.get_level_values("instrument")))
    exchange = Exchange(
        freq="day", start_time=dates[0], end_time=END, codes=codes,
        deal_price=("$open", "$close"), open_cost=0.0005,
        close_cost=0.0015, min_cost=5, trade_unit=100,
        limit_threshold=("($open / Ref($close, 1) - 1) >= 0.095",
                         "($close / Ref($close, 1) - 1) <= -0.095"),
    )
    raw, positions = backtest_daily(
        start_time=dates[0], end_time=END, strategy=strategy,
        executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False),
        account=1_000_000, benchmark="SH000300", exchange_kwargs={"exchange": exchange},
    )
    if not raw.loc[raw.index < first_trade, "turnover"].eq(0).all():
        raise ValueError("Trading occurred on the signal-only warm-up day")
    raw = raw.loc[raw.index >= first_trade].copy()
    if not raw.index.equals(baseline.index):
        raise ValueError("Staged and baseline backtests have different date intervals")
    # Reuse the independently checked terminal benchmark quote so only the
    # execution schedule, not benchmark data, changes in this comparison.
    raw["bench"] = baseline["bench"]
    report = enrich_daily_report(raw, f"{name}_staged")
    prior_nav = report.portfolio_nav.shift(1, fill_value=1) * 1_000_000
    report["fees_cash"] = report.cost * prior_nav
    report["holdings_count"] = pd.Series({date: len(position.get_stock_list())
                                         for date, position in positions.items()}).reindex(report.index, fill_value=0)
    if not np.allclose(report.portfolio_nav * 1_000_000, report.account, rtol=1e-6):
        raise ValueError("Compounded returns disagree with the simulated account")
    if not np.isfinite(report[["net_return", "cost", "bench"]].to_numpy()).all():
        raise ValueError("Nonfinite returns or costs")

    events = pd.DataFrame(strategy.events)
    if events.date.duplicated().any() or events.empty:
        raise ValueError("Execution phases have duplicate dates or no trades")
    if not events.phase.eq("initial_buy").iloc[0]:
        raise ValueError("First portfolio entry was not funded from starting cash")
    for index, event in events.iterrows():
        if event.phase == "close_sell":
            if index + 1 >= len(events) or events.iloc[index + 1].phase != "funded_buy":
                # The final sell may fall on the last available session.
                if event.date != END:
                    raise ValueError("Close sale was not followed by a funded buy")
            elif calendar[calendar > event.date].min() != events.iloc[index + 1].date:
                raise ValueError("Purchase did not take place at the following open")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT / f"{name}_staged_daily.csv", encoding="utf-8-sig")
    events.to_csv(OUTPUT / f"{name}_staged_events.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for scheme, frame in (("both_next_open", baseline), ("sell_close_buy_following_open", report)):
        for period, group in (("all", frame), *[(str(y), g) for y, g in frame.groupby(frame.index.year)]):
            row = summarize_report(group, scheme).iloc[0].to_dict()
            row.update({"scheme": scheme, "period": period,
                        "total_fees_cny": float(group.fees_cash.sum()),
                        "average_holdings": float(group.holdings_count.mean())})
            summaries.append(row)
    result = pd.DataFrame(summaries)
    result.to_csv(OUTPUT / f"{name}_comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / f"{name}_protocol.json").write_text(json.dumps({
        "prediction_source": str(BASELINE_REPORT / f"h{horizon}_open_to_open_predictions.parquet"),
        "first_signal_date": str(dates[0].date()), "first_trade_date": str(first_trade.date()),
        "end_date": str(END.date()), "signal_time": "T close",
        "baseline": "sell and buy at T+1 open",
        "staged": "sell reductions at T+1 close; buy additions with settled cash at T+2 open",
        "initial_entry": "T+1 open using initial cash",
        "common_costs": {"buy": 0.0005, "sell": 0.0015, "minimum_cny": 5},
        "limitation": "Daily-bar ideal fills omit intraday slippage and auction nonfill; target remains open-to-open.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result[["scheme", "period", "total_return", "annualized_return",
                  "max_drawdown", "total_fees_cny", "average_holdings"]].to_string(index=False), flush=True)
    print(f"Saved: {OUTPUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--rebalance", type=int, default=10)
    args = parser.parse_args()
    run(args.horizon, args.rebalance)


if __name__ == "__main__":
    main()
