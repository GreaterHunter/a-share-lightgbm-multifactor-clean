"""Capture a read-only account snapshot and compare old/new integer sizing."""
from __future__ import annotations

import argparse
import json

from paper_futu.cn_common import ROOT,STATE_DIR,tiered_weights,target_share_quantity
from paper_futu.cn_execution_session import FutuBroker, save_result
from paper_futu.cn_integer_targets import allocate_targets
from paper_futu.prospective_s6 import read,digest,write_once

OUTPUT = ROOT/"reports/paper_futu/execution_v5_integer_retry_20261020"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",action="store_true")
    args = parser.parse_args()
    snapshot_path = OUTPUT/"account_before_correction.json"
    if snapshot_path.exists():
        try:
            read(snapshot_path)
        except json.JSONDecodeError:
            # Preserve an interrupted/failed serialization, never disguise it as valid evidence.
            incomplete = OUTPUT/"account_before_correction.incomplete.json"
            if incomplete.exists():
                raise ValueError("Repeated incomplete snapshot; inspect instead of overwriting")
            snapshot_path.rename(incomplete)
    if args.capture and not snapshot_path.exists():
        from paper_futu.cn_common import china_now
        source = STATE_DIR/"orders_2026-10-20.jsonl"
        start = next(e for s in source.read_text(encoding="utf-8").splitlines()
                     if (e:=json.loads(s)).get("event") == "execution_start")
        broker = FutuBroker()
        try:
            positions,funds = broker.positions(True),broker.funds(True)
            all_quotes = broker.quotes(sorted(set(start["selected"])|set(positions)))
            fields = ["code","name","last_price","prev_close_price","ask_price","bid_price",
                      "lot_size","price_spread","suspension","update_time"]
            quotes = {c:{k:r[k] for k in fields if k in r} for c,r in all_quotes.items()}
            broker_orders = broker.orders(True)
            value = dict(observed_at=china_now().isoformat(),positions=positions,funds=funds,
                quotes=quotes,selected=start["selected"],weights=start["weights"],
                order_journal_sha256=digest(source),orders=broker_orders)
            value = json.loads(json.dumps(value,default=str),parse_constant=lambda value:None)
            write_once(snapshot_path,value)
        finally:
            broker.close()
    snapshot = read(snapshot_path)
    positions,funds,quotes = (snapshot[k] for k in ["positions","funds","quotes"])
    allocation = allocate_targets(snapshot["selected"],snapshot["weights"],funds["total_assets"],positions,quotes,
                                  funds["cash"]-funds.get("frozen_cash",0))
    old = {c:target_share_quantity(c,snapshot["weights"][c],funds["total_assets"],float(quotes[c]["last_price"]))
           for c in snapshot["selected"]}
    fault_nav,fault_price,fault_weight = 1009718.671,654.47,.95*.4/6
    old_fault = target_share_quantity("SZ.300308",fault_weight,fault_nav,fault_price)
    assert old_fault == 0
    assert 100*fault_price <= (fault_weight+.005)*fault_nav
    assert allocation["target_exposure"] <= funds["total_assets"]*.95+1e-6
    assert allocation["expected_post_cash"] >= funds["total_assets"]*.05-1e-6
    differences = [dict(code=c,held=positions.get(c,{}).get("qty",0),legacy_target=old[c],
                        revised_target=allocation["targets"][c]) for c in snapshot["selected"]
                   if old[c] != allocation["targets"][c]]
    value = dict(status="verified",snapshot_sha256=digest(snapshot_path),
        original_fault=dict(code="SZ.300308",nav=fault_nav,price=fault_price,held=100,
                            legacy_target=0,one_lot_within_tolerance=True),
        recent_snapshot_comparison=differences,revised_allocation=allocation,
        orders_submitted=0,note="Read-only same-snapshot comparison, not a new performance backtest")
    save_result(OUTPUT/"sizing_audit.json",value)
    print(json.dumps(dict(status="verified",changed_integer_targets=len(differences),
                         zhongji_new_target=allocation["targets"]["SZ.300308"],
                         target_exposure=allocation["target_exposure"],
                         expected_cash=allocation["expected_post_cash"],orders_submitted=0),ensure_ascii=False))


if __name__ == "__main__":
    main()
