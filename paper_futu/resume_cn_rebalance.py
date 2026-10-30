"""Explicit buy-only recovery of a failed batch; preserve its selected names.

Never rerun selection or replay a filled sell. All prior receipts must be final,
and any ambiguous/repeated recovery requires manual reconciliation.
"""
from __future__ import annotations

import argparse
import json
import math
import msvcrt
import time

from paper_futu.cn_common import STATE_DIR, china_now, load_deployment, validate_plan_deployment, validate_execution_session, tiered_weights, target_share_quantity
from paper_futu.execute_cn_plan import account_context, paper_account_id, save_account_snapshot, fresh_quotes
from paper_futu.cn_broker_queries import query_positions, PositionCache
from paper_futu.execution_override import execution_allowed
from paper_futu.smoke_order import TERMINAL, require_ok, append_event
from paper_futu.independent_cn_market import write_json, sha


def verified_original(plan, events, orders):
    starts = [e for e in events if e.get("event") == "execution_start"]
    if len(starts) != 1:
        raise ValueError("Recovery needs exactly one original batch")
    start = starts[0]
    for key in ["signal_date", "execution_date", "model_dir", "top_k", "n_drop"]:
        if start.get(key) != plan[key]:
            raise ValueError("Original batch/active plan mismatch")
    selected = start["selected"]
    if len(selected) != plan["top_k"] or len(set(selected)) != len(selected):
        raise ValueError("Invalid original selected names")
    expected = tiered_weights(selected, **{k:plan[k] for k in ["risk_degree","top_k","top_tier_size","top_tier_weight"]})
    if expected != start["weights"] or not set(selected).issubset(plan["predictions"]):
        raise ValueError("Original weights changed")
    submitted = [e for e in events if e.get("event") == "submitted"]
    before = [e for e in events if e.get("event") == "before_submit"]
    if not submitted or len(submitted) != len(before) or any(e["side"] != "SELL" for e in submitted):
        raise ValueError("This buy-only recovery requires an unambiguous sell-only original batch")
    ids = [str(e["order_id"]) for e in submitted]
    if len(set(ids)) != len(ids) or set(orders.order_id.astype(str)) != set(ids):
        raise ValueError("Unaccounted/manual/missing orders; reconcile first")
    by_id = orders.assign(order_id=orders.order_id.astype(str)).set_index("order_id")
    for receipt in submitted:
        row = by_id.loc[str(receipt["order_id"])]
        if (row.order_status != "FILLED_ALL" or row.trd_side != "SELL" or row.code != receipt["code"]
                or row.qty != receipt["qty"] or row.dealt_qty != receipt["qty"]):
            raise ValueError("Original sell not fully verified")
    return start, ids


def remaining_buys(start, positions, nav, quotes):
    held = {r.code:int(r.qty) for r in positions.itertuples() if r.qty > 0}
    if not set(held).issubset(start["selected"]):
        raise ValueError("Unexpected holding after sells")
    targets = {a["code"]:a["target"] for a in start["actions"] if a["code"] in start["selected"]}
    actions = []
    for code in start["selected"]:
        if code not in targets and code not in held:
            raise ValueError("Cannot reconstruct the original target")
        original = targets.get(code, held.get(code,0))
        live = target_share_quantity(code, start["weights"][code], nav, float(quotes.loc[code].last_price))
        desired = min(original, live)
        quantity = max(0, desired-held.get(code,0))
        if quantity:
            actions.append(dict(code=code,qty=quantity,held=held.get(code,0),original_target=original,live_capped_target=desired))
    return actions


def execute(submit):
    from paper_futu.cn_execution_session import rebalance_main
    return rebalance_main(["--resume"] + (["--submit"] if submit else []))


def main():
    parser = argparse.ArgumentParser(description="Explicit resume of a version-5 frozen batch")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    return execute(args.submit)


if __name__ == "__main__":
    main()
