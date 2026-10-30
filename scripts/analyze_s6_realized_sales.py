"""Read-only analysis of realized stock/date sales in the sealed S6 replay.

No fitting, market requests, account replay, or paper-trading mutations occur.
Cost is a moving weighted average in the replay's adjusted-share units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/s6_quarterly_l1l2_small_asof_20261013"
BREAK_EVEN_CNY = 0.01


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct_sales(orders: pd.DataFrame):
    """Return per-fill realized P&L and the remaining average-cost ledger.

    Directions follow the source journal: 1 buy, 0 sell. Purchase fees enter
    net cost basis; a partial sale releases the proportional purchase cost.
    Input order is retained within a date (the replay executes sells first).
    """
    ledger: dict[str, dict[str, float]] = {}
    sales = []
    for row in orders.sort_values("datetime", kind="stable").itertuples(index=False):
        if row.notional <= 0:
            continue
        stock = str(row.instrument)
        qty, value, fee = map(float, (row.adjusted_quantity, row.notional, row.fee))
        if not np.isfinite([qty, value, fee]).all() or qty <= 0 or fee < 0:
            raise ValueError("Invalid filled trade")
        item = ledger.setdefault(stock, dict(quantity=0.0, gross_cost=0.0, net_cost=0.0))
        if row.direction == 1:
            item["quantity"] += qty
            item["gross_cost"] += value
            item["net_cost"] += value + fee
        elif row.direction == 0:
            held = item["quantity"]
            if held <= 0 or qty > held + max(1e-7, held * 1e-10):
                raise ValueError(f"Oversell or missing opening balance: {stock}")
            fraction = min(1.0, qty / held)
            gross_basis = item["gross_cost"] * fraction
            net_basis = item["net_cost"] * fraction
            sales.append(dict(datetime=pd.Timestamp(row.datetime), instrument=stock,
                              quantity=qty, proceeds=value, sell_fee=fee,
                              gross_basis=gross_basis, net_basis=net_basis,
                              gross_pnl=value-gross_basis,
                              net_pnl=value-fee-net_basis))
            for key in ("quantity", "gross_cost", "net_cost"):
                item[key] *= 1.0 - fraction
            if item["quantity"] < max(1e-7, held * 1e-10):
                item.update(quantity=0.0, gross_cost=0.0, net_cost=0.0)
        else:
            raise ValueError("Unknown direction")
    return pd.DataFrame(sales), ledger


def stock_date_sales(fills: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    """One observation per sold stock per date, not per order or raw share."""
    events = fills.groupby(["datetime", "instrument"], sort=True).sum().reset_index()
    ending = holdings.set_index(["datetime", "instrument"]).adjusted_quantity
    events["ending_quantity"] = [float(ending.get((r.datetime, r.instrument), 0.0))
                                 for r in events.itertuples(index=False)]
    events["full_exit"] = events.ending_quantity <= 1e-7
    for basis in ("gross", "net"):
        pnl = events[f"{basis}_pnl"]
        events[f"{basis}_return"] = pnl / events[f"{basis}_basis"]
        events[f"{basis}_class"] = np.select(
            [pnl > BREAK_EVEN_CNY, pnl < -BREAK_EVEN_CNY],
            ["profit", "loss"], default="flat")
    return events


def summarize(events: pd.DataFrame, rebalance_dates: list[str]) -> dict:
    schedule = pd.to_datetime(rebalance_dates)
    output = dict(scheduled_rebalances=len(schedule),
                  selling_rebalances=int(events.datetime.nunique()),
                  sold_stock_date_events=len(events))
    for basis in ("gross", "net"):
        classes = events[f"{basis}_class"]
        group = {}
        for category in ("profit", "loss", "flat"):
            chosen = events.loc[classes.eq(category)]
            count = len(chosen)
            group[category] = dict(
                count=count,
                average_count_per_selling_rebalance=count / output["selling_rebalances"],
                average_count_per_scheduled_rebalance=count / len(schedule),
                share_of_sales=count / len(events),
                average_return=float(chosen[f"{basis}_return"].mean()) if count else None,
                median_return=float(chosen[f"{basis}_return"].median()) if count else None,
                capital_weighted_return=(float(chosen[f"{basis}_pnl"].sum() /
                                               chosen[f"{basis}_basis"].sum()) if count else None),
                realized_pnl_cny=float(chosen[f"{basis}_pnl"].sum()))
        output[basis] = group
    return output


def analyze_account(offset: int) -> dict:
    task_path = SOURCE / f"tasks/account_history_QREG_{offset}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task["status"] == "complete"
    for name, expected in task["artifacts"].items():
        assert sha256(Path(name)) == expected, f"Source artifact changed: {name}"
    directory = Path(task["result"]["directory"])
    orders = pd.read_parquet(directory / "orders.parquet")
    holdings = pd.read_parquet(directory / "holdings.parquet")
    daily = pd.read_parquet(directory / "daily.parquet")
    assert not holdings.duplicated(["datetime", "instrument"]).any()
    fills, ledger = reconstruct_sales(orders)
    events = stock_date_sales(fills, holdings)
    schedule = task["result"]["rebalance_dates"]
    assert set(events.datetime.dt.strftime("%Y-%m-%d")).issubset(schedule)

    # Reconcile the final realized/unrealized split, all fees, and cash with
    # the sealed daily account. No market prices or models are recomputed.
    last = holdings.loc[holdings.datetime.eq(daily.index[-1])].set_index("instrument")
    assert set(last.index) == {s for s, v in ledger.items() if v["quantity"] > 1e-7}
    quantity_error = max(abs(ledger[s]["quantity"]-row.adjusted_quantity)
                         for s, row in last.iterrows())
    assert quantity_error < 1e-5, quantity_error
    open_net_cost = sum(item["net_cost"] for item in ledger.values())
    unrealized = float(last.market_value.sum())-open_net_cost
    realized = float(events.net_pnl.sum())
    initial = float(task["result"]["initial_cash"])
    net_profit = float(daily.account.iloc[-1])-initial
    profit_error = abs(realized+unrealized-net_profit)
    assert profit_error < 0.01, profit_error
    trades = orders.loc[orders.notional.gt(0)]
    flows = np.where(trades.direction.eq(1), -trades.notional, trades.notional)-trades.fee
    cash_error = abs(initial+float(flows.sum())-float(daily.cash.iloc[-1]))
    fee_error = abs(float(trades.fee.sum())-float(daily.fees_cash.sum()))
    assert max(cash_error, fee_error) < 0.01, (cash_error, fee_error)
    years = {}
    for year in sorted(events.datetime.dt.year.unique()):
        chosen_dates = [d for d in schedule if d.startswith(str(year))]
        years[str(year)] = summarize(events.loc[events.datetime.dt.year.eq(year)], chosen_dates)
    return dict(offset=offset, start=str(daily.index[0].date()), end=str(daily.index[-1].date()),
                task_sha256=sha256(task_path), artifact_sha256=task["artifacts"],
                all_sales=summarize(events, schedule),
                full_exits=summarize(events.loc[events.full_exit], schedule),
                partial_reductions=summarize(events.loc[~events.full_exit], schedule),
                yearly=years,
                audit=dict(maximum_final_quantity_error=quantity_error,
                           realized_net_pnl_cny=realized, unrealized_net_pnl_cny=unrealized,
                           account_profit_cny=net_profit, profit_identity_error_cny=profit_error,
                           cash_error_cny=cash_error, fee_error_cny=fee_error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT/"reports/analysis/s6_realized_sales_asof_20261013/summary.json")
    args = parser.parse_args()
    accounts = [analyze_account(offset) for offset in range(5)]
    averages = {}
    for sale_kind in ("all_sales", "full_exits", "partial_reductions"):
        averages[sale_kind] = {}
        for basis in ("gross", "net"):
            averages[sale_kind][basis] = {}
            for category in ("profit", "loss", "flat"):
                values = [a[sale_kind][basis][category] for a in accounts]
                averages[sale_kind][basis][category] = {
                    key: (float(np.mean([v[key] for v in values if v[key] is not None]))
                          if any(v[key] is not None for v in values) else None)
                    for key in values[0] if key != "count"}
    result = dict(strategy_id="S6_head6_40_QREG", source=str(SOURCE),
                  analysis_code_sha256=sha256(Path(__file__)),
                  break_even_pnl_tolerance_cny=BREAK_EVEN_CNY,
                  cost_method="moving_weighted_average_in_adjusted_share_units",
                  observation="one_stock_one_sale_date_including_partial_reductions",
                  average_return_weighting="equal_stock_date_then_equal_offset",
                  unrealized_positions_excluded_from_sale_statistics=True,
                  limitations=["Sealed historical ideal-open replay, not live fills.",
                               "Historical sizing precedes the 2026-10-20 integer-sizing repair.",
                               "Five offsets share historical dates; not five independent samples."],
                  accounts=accounts, equal_offset_averages=averages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), average=averages,
                          main=accounts[0],
                          audit_passed=True), ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
