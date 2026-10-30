"""Trade-quality metrics from the sealed S6 journal; never replay or trade."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_s6_realized_sales import (
    ROOT, SOURCE, reconstruct_sales, sha256, stock_date_sales,
)


def sale_ages(orders: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Average age of shares sold under proportional/average-cost disposal.

    Partial sales do not reset the purchase date of retained shares. A new
    purchase updates the quantity-weighted mean entry date. Trading days
    exclude the entry day; calendar days are supplied separately.
    """
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar must be unique and sorted")
    positions = {}
    rows = []
    for row in orders.sort_values("datetime", kind="stable").itertuples(index=False):
        if row.notional <= 0:
            continue
        date = pd.Timestamp(row.datetime)
        index = calendar.get_indexer([date])[0]
        if index < 0:
            raise ValueError("Execution date is missing from calendar")
        ordinal = date.toordinal()
        qty = float(row.adjusted_quantity)
        item = positions.setdefault(row.instrument, dict(quantity=0., trading_mass=0., calendar_mass=0.))
        if row.direction == 1:
            item["quantity"] += qty
            item["trading_mass"] += qty * index
            item["calendar_mass"] += qty * ordinal
        elif row.direction == 0:
            held = item["quantity"]
            if held <= 0 or qty > held + max(1e-7, held * 1e-10):
                raise ValueError("Sale without adequate purchase history")
            trade_age = index-item["trading_mass"]/held
            calendar_age = ordinal-item["calendar_mass"]/held
            if min(trade_age, calendar_age) < -1e-7:
                raise ValueError("Negative holding duration")
            rows.append(dict(datetime=date, instrument=row.instrument, quantity=qty,
                             age_trading_mass=qty*max(0., trade_age),
                             age_calendar_mass=qty*max(0., calendar_age)))
            remaining = 1-min(1., qty/held)
            for key in item:
                item[key] *= remaining
            if item["quantity"] < max(1e-7, held * 1e-10):
                item.update(quantity=0., trading_mass=0., calendar_mass=0.)
        else:
            raise ValueError("Unknown direction")
    return pd.DataFrame(rows)


def quality(events: pd.DataFrame) -> dict:
    winners = events.loc[events.net_class.eq("profit")]
    losers = events.loc[events.net_class.eq("loss")]
    result = dict(sale_events=len(events), win_probability=len(winners)/len(events),
                  loss_probability=len(losers)/len(events))
    for label, chosen in (("win", winners), ("loss", losers)):
        result[f"average_{label}_return"] = float(chosen.net_return.mean())
        result[f"average_{label}_holding_trading_days"] = float(chosen.holding_trading_days.mean())
        result[f"average_{label}_holding_calendar_days"] = float(chosen.holding_calendar_days.mean())
    result["profit_cny"] = float(winners.net_pnl.sum())
    result["loss_cny"] = float(losers.net_pnl.sum())
    return result


def extreme(events: pd.DataFrame, maximum: bool) -> dict:
    row = events.loc[events.net_return.idxmax() if maximum else events.net_return.idxmin()]
    return dict(offset=int(row.offset), instrument=row.instrument,
                sale_date=str(row.datetime.date()), net_return=float(row.net_return),
                net_pnl_cny=float(row.net_pnl), net_cost_basis_cny=float(row.net_basis),
                holding_trading_days=float(row.holding_trading_days),
                full_exit=bool(row.full_exit))


def summarize_phases(events: pd.DataFrame) -> dict:
    phases = [quality(part) for _, part in events.groupby("offset")]
    means = {key: float(np.mean([phase[key] for phase in phases])) for key in phases[0]}
    ratio = means["average_win_return"] / abs(means["average_loss_return"])
    means["payoff_ratio"] = ratio
    means["win_loss_adjusted_payoff_ratio"] = ratio*means["win_probability"]/means["loss_probability"]
    means["monetary_profit_factor"] = means["profit_cny"]/abs(means["loss_cny"])
    means["maximum_sale_return"] = extreme(events, True)
    means["minimum_sale_return"] = extreme(events, False)
    means["observations_across_phases"] = len(events)
    means["phases"] = phases
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT/"reports/analysis/s6_trade_quality_asof_20261013/summary.json")
    args = parser.parse_args()
    original_path = ROOT/"reports/analysis/s6_realized_sales_asof_20261013/summary.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    all_events = []
    task_hashes = {}
    for offset in range(5):
        task_path = SOURCE/f"tasks/account_history_QREG_{offset}.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task["status"] != "complete":
            raise ValueError("Account is not complete")
        for path, expected in task["artifacts"].items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"Sealed account changed: {path}")
        task_hashes[str(offset)] = sha256(task_path)
        directory = Path(task["result"]["directory"])
        orders = pd.read_parquet(directory/"orders.parquet")
        holdings = pd.read_parquet(directory/"holdings.parquet")
        daily = pd.read_parquet(directory/"daily.parquet")
        fills, _ = reconstruct_sales(orders)
        ages = sale_ages(orders, daily.index)
        if not (fills[["datetime", "instrument"]].equals(ages[["datetime", "instrument"]])
                and np.allclose(fills.quantity, ages.quantity)):
            raise ValueError("Cost and duration ledgers disagree")
        fills["age_trading_mass"] = ages.age_trading_mass
        fills["age_calendar_mass"] = ages.age_calendar_mass
        events = stock_date_sales(fills, holdings)
        events["holding_trading_days"] = events.age_trading_mass/events.quantity
        events["holding_calendar_days"] = events.age_calendar_mass/events.quantity
        events["offset"] = offset
        reference = next(a for a in original["accounts"] if a["offset"] == offset)
        for period in ("all", "2022", "2023", "2024", "2025", "2026"):
            subset = events if period == "all" else events.loc[events.datetime.dt.year.eq(int(period))]
            metrics = quality(subset)
            old = reference["all_sales"] if period == "all" else reference["yearly"][period]
            for name, category in (("win", "profit"), ("loss", "loss")):
                if not np.isclose(metrics[f"average_{name}_return"],
                                  old["net"][category]["average_return"], rtol=0, atol=1e-12):
                    raise ValueError("Return metrics differ from the previous analysis")
        all_events.append(events)
    events = pd.concat(all_events, ignore_index=True)
    summaries = {}
    full_exits = {}
    for period in ("all", "2022", "2023", "2024", "2025", "2026"):
        subset = events if period == "all" else events.loc[events.datetime.dt.year.eq(int(period))]
        summaries[period] = summarize_phases(subset)
        full_exits[period] = summarize_phases(subset.loc[subset.full_exit])
    result = dict(strategy="S6_head6_40_QREG", start="2022-01-04", end="2026-10-13",
                  source=str(SOURCE), task_sha256=task_hashes,
                  implementation_sha256=sha256(Path(__file__)),
                  cost_ledger_sha256=sha256(ROOT/"scripts/analyze_s6_realized_sales.py"),
                  prior_analysis_sha256=sha256(original_path),
                  transaction_unit="one stock per sale date, including partial reductions",
                  mean_weighting="equal stock/date within each phase, then equal phase",
                  maximum_definition="extreme single sale event across all five phases",
                  holding_age_definition="quantity-weighted entry age of sold shares; entry day excluded",
                  adjusted_payoff_definition="(mean winning return / abs(mean losing return)) * (win probability / loss probability)",
                  monetary_profit_factor_definition="total winning net P&L / abs(total losing net P&L)",
                  source_hash_and_previous_return_checks_passed=True,
                  all_sales=summaries, full_exits=full_exits,
                  limitations=["Realized sales only; open positions are excluded.",
                               "Historical ideal-open replay before the integer-sizing repair.",
                               "Five phases are overlapping observations, not independent samples.",
                               "Replacing individual returns by group averages does not reproduce actual compounded wealth."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), all_sales=summaries), ensure_ascii=True))


if __name__ == "__main__":
    main()
