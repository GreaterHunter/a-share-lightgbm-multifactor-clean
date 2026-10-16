"""Read-only audit of the paper plan's lot feasibility and opening execution gap.

This module never places, modifies, or cancels an order.  Run it after an
execution session to compare the simulator's filled prices with that day's
actual opening quotes.  It does not replace a minute-data/slippage backtest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paper_futu.cn_common import ROOT, STATE_DIR, target_share_quantity


def planned_lot_feasibility(preview: dict, capital: float) -> pd.DataFrame:
    """Apply the *existing* target weights and A-share buying units to capital."""
    if capital <= 0:
        raise ValueError("Capital must be positive")
    selected = preview["selected"]
    weights = preview["weights"]
    if set(weights) != set(selected):
        raise ValueError("Preview weights and selected stocks differ")
    # The trade preview holds the market price used when sizing the actual plan.
    prices = {item["code"]: float(item["reference_price"])
              for item in preview["actions"]}
    missing = set(selected) - set(prices)
    if missing:
        raise ValueError(f"No buying reference price for {sorted(missing)}")
    rows = []
    for code in selected:
        price = prices[code]
        weight = weights[code]
        shares = target_share_quantity(code, weight, capital, price)
        minimum_shares = 200 if code.startswith("SH.688") else 100
        rows.append({
            "capital": capital,
            "code": code,
            "price": price,
            "target_weight": weight,
            "target_cash": capital * weight,
            "minimum_buy_shares": minimum_shares,
            "minimum_lot_cash": price * minimum_shares,
            "shares": shares,
            "buyable": shares >= minimum_shares,
            "invested_cash": shares * price,
        })
    return pd.DataFrame(rows)


def execution_gap(orders: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Return paid/open price gaps for verified, filled simulated orders."""
    needed_orders = {"code", "trd_side", "order_status", "dealt_qty", "dealt_avg_price"}
    needed_quotes = {"code", "open_price"}
    if not needed_orders.issubset(orders) or not needed_quotes.issubset(snapshots):
        raise ValueError("Broker order or quote fields are incomplete")
    if orders.empty or not orders.order_status.eq("FILLED_ALL").all():
        raise ValueError("The original orders are not all fully filled")
    merged = orders[list(needed_orders)].merge(
        snapshots[list(needed_quotes)], on="code", how="left", validate="many_to_one"
    )
    merged["open_price"] = pd.to_numeric(merged.open_price, errors="coerce")
    merged["dealt_avg_price"] = pd.to_numeric(merged.dealt_avg_price, errors="coerce")
    if (merged[["open_price", "dealt_avg_price"]].isna().any().any()
            or (merged[["open_price", "dealt_avg_price"]] <= 0).any().any()):
        raise ValueError("Missing or nonpositive opening/fill price")
    # Positive means worse than the exact-open ideal for either trade side.
    side = merged.trd_side.map({"BUY": 1, "SELL": -1})
    if side.isna().any():
        raise ValueError("Unknown trade side")
    merged["adverse_gap_pct"] = side * (merged.dealt_avg_price / merged.open_price - 1) * 100
    merged["notional"] = merged.dealt_qty.astype(float) * merged.dealt_avg_price
    return merged.sort_values("code").reset_index(drop=True)


def broker_execution_gap(date: str, journal_path: Path) -> pd.DataFrame:
    """Read only the order IDs from our own journal, then query Futu SIMULATE."""
    from futu import OpenQuoteContext, TrdEnv

    from paper_futu.execute_cn_plan import account_context, paper_account_id
    from paper_futu.smoke_order import require_ok

    events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    submitted = [event for event in events if event.get("event") == "submitted"]
    if not submitted or len({str(event["order_id"]) for event in submitted}) != len(submitted):
        raise ValueError("No unique submitted order IDs in this execution journal")
    ids = {str(event["order_id"]) for event in submitted}
    trade = account_context()
    quote = None
    try:
        account_id = paper_account_id(trade)
        all_orders = require_ok(
            trade.order_list_query(start=date, trd_env=TrdEnv.SIMULATE,
                                   acc_id=account_id, refresh_cache=True),
            "读取模拟订单",
        )
        orders = all_orders.loc[all_orders.order_id.astype(str).isin(ids)].copy()
        if len(orders) != len(ids) or set(orders.order_id.astype(str)) != ids:
            raise ValueError("Broker orders do not exactly match the execution journal")
        quote = OpenQuoteContext(host="127.0.0.1", port=11111)
        codes = sorted(orders.code.unique())
        snapshots = require_ok(quote.get_market_snapshot(codes), "读取当日开盘价")
        return execution_gap(orders, snapshots)
    finally:
        if quote is not None:
            quote.close()
        trade.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="The original execution date, YYYY-MM-DD")
    parser.add_argument("--broker-fills", action="store_true", help="Query paper fills and opening quotes read-only")
    args = parser.parse_args()
    journal_path = STATE_DIR / f"orders_{args.date}.jsonl"
    # A subsequent dry-run may overwrite preview_*.json, but the immutable
    # execution_start entry records the reference prices actually used.
    events = ([json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
              if journal_path.exists() else [])
    starts = [event for event in events if event.get("event") == "execution_start"]
    if len(starts) == 1:
        preview = starts[0]
    elif not starts:
        preview = json.loads((STATE_DIR / f"preview_{args.date}.json").read_text(encoding="utf-8"))
    else:
        raise ValueError("Execution journal contains multiple starts")
    feasibility = pd.concat(
        [planned_lot_feasibility(preview, cash) for cash in (5000, 10000, float(preview["nav"]))],
        ignore_index=True,
    )
    output = ROOT / "reports" / "paper_futu"
    output.mkdir(parents=True, exist_ok=True)
    lot_path = output / f"lot_feasibility_{args.date}.csv"
    feasibility.to_csv(lot_path, index=False, encoding="utf-8-sig")
    print(feasibility.groupby("capital")["buyable"].agg(["sum", "count"]).to_string())
    print(f"Lot audit: {lot_path}")
    if args.broker_fills:
        fills = broker_execution_gap(args.date, journal_path)
        gap_path = output / f"opening_execution_gap_{args.date}.csv"
        fills.to_csv(gap_path, index=False, encoding="utf-8-sig")
        weighted = (fills.adverse_gap_pct * fills.notional).sum() / fills.notional.sum()
        print(f"Filled-order notional-weighted adverse gap to open: {weighted:.4f}%")
        print(f"Fill audit: {gap_path}")


if __name__ == "__main__":
    main()
