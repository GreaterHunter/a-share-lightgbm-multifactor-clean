"""Explicit, date-bound, buy-only correction to a TOTAL share target."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json

from paper_futu.cn_common import STATE_DIR, china_now, load_deployment, validate_plan_deployment
from paper_futu.cn_execution_session import (FutuBroker, account_lock,
    run_manifest, save_result)
from paper_futu.cn_execution_engine import ReconciliationRequired, checked_quote
from paper_futu.cn_integer_targets import legal_order_quantity
from paper_futu.prospective_s6 import write_once, digest
from paper_futu.cn_runtime_protocol import verify_active_protocol


def correction_manifest(plan, original, positions, funds, quotes, code, target, now):
    if code not in original["selected"] or code not in original["weights"]:
        raise ValueError("Correction must stay inside the original frozen selection")
    held = int(positions.get(code,{}).get("qty",0))
    if target < held:
        raise ValueError("Buy-only correction never sells an existing position")
    if target > held and legal_order_quantity(code,target-held,"BUY") != target-held:
        raise ValueError("Residual correction is not a legal buy quantity")
    nav = funds["total_assets"]
    weight = float(original["weights"][code])
    reference = float(quotes[code]["last_price"])
    checked_quote(code,quotes[code],now)
    if target*reference > (weight+.005)*nav+1e-6:
        raise ValueError("One-lot correction exceeds the agreed 0.5 percentage-point tolerance")
    start = now
    end = min(start+timedelta(minutes=20),start.replace(hour=14,minute=55,second=0,microsecond=0))
    if start >= end-timedelta(seconds=30):
        raise ValueError("No authorized continuous-session correction window remains")
    quantities = {c:int(r["qty"]) for c,r in positions.items() if r["qty"] > 0}
    targets = dict(quantities); targets[code] = target
    references = {c:float(r["last_price"]) for c,r in quotes.items()}
    actions = [] if held == target else [dict(code=code,side="BUY",held=held,target=target,qty=target-held,
                                              weight=weight,reference_price=reference)]
    identity = f"correction:{plan['execution_date']}:{code}:{target}:{plan['model_sha256']}"
    return dict(version=5,kind="single_position_correction",environment="SIMULATE",
        execution_date=plan["execution_date"],signal_date=plan["signal_date"],model_sha256=plan["model_sha256"],
        active_plan_sha256=digest(STATE_DIR/"active_plan.json"),batch_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
        nav=nav,initial_cash=funds["cash"],initial_quantities=quantities,targets=targets,
        selected=original["selected"],weights=original["weights"],actions=actions,reference_prices=references,
        top_k=plan["top_k"],risk_degree=plan["risk_degree"],not_before=start.isoformat(),expires_at=end.isoformat(),
        correction_code=code,correction_target_quantity=target,weight_tolerance=.005,
        user_authorization="Implement approved repair plan; independently restore SZ.300308 to total 100 shares on 2026-10-20; no other orders",
        no_next_day_retry=True,other_positions_unchanged=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-date",required=True)
    parser.add_argument("--code",required=True)
    parser.add_argument("--target-qty",type=int,required=True)
    parser.add_argument("--submit",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args(argv)
    # This release carries exactly the human-approved one-time correction, not a generic order console.
    if (args.execution_date,args.code,args.target_qty) != ("2026-10-20","SZ.300308",100):
        raise ValueError("This authorization covers only 2026-10-20 SZ.300308 total 100 shares")
    now = china_now()
    if now.date().isoformat() != args.execution_date or (now.hour*60+now.minute >= 14*60+55 and not args.resume):
        print(json.dumps(dict(status="authorization_expired",orders_submitted=0,no_next_day_retry=True)))
        return
    with account_lock():
        if args.submit:
            verify_active_protocol()
        plan = json.loads((STATE_DIR/"active_plan.json").read_text(encoding="utf-8"))
        model,meta = load_deployment(); validate_plan_deployment(plan,model,meta)
        if plan["execution_date"] != args.execution_date:
            raise ValueError("Correction differs from original execution date")
        source_receipt = STATE_DIR/f"execution_{args.execution_date}.json"
        source_journal = STATE_DIR/f"orders_{args.execution_date}.jsonl"
        if not source_receipt.exists():
            raise ValueError("Original batch is not closed; do not create an overlapping correction")
        starts = [e for s in source_journal.read_text(encoding="utf-8").splitlines()
                  if (e:=json.loads(s)).get("event") == "execution_start"]
        if len(starts) != 1:
            raise ValueError("Cannot identify original selection")
        stem = f"correction_v5_{args.execution_date}_{args.code.replace('.','')}_{args.target_qty}"
        path = STATE_DIR/(stem+".plan.json")
        result_path = STATE_DIR/(stem+".result.json")
        journal = STATE_DIR/(stem+".jsonl")
        if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "completed":
            print("Correction already completed; no additional shares bought")
            return
        if args.resume and not path.exists():
            raise ValueError("No frozen correction to resume")
        if journal.exists() and not args.resume:
            raise ValueError("Existing correction intents; reconcile using --resume, never restart")
        broker = FutuBroker()
        try:
            reconcile_only = False
            try:
                broker.session(args.execution_date,meta["model_year"])
            except ValueError:
                if not args.resume:
                    raise
                reconcile_only = True
            if path.exists():
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if (manifest["active_plan_sha256"] != digest(STATE_DIR/"active_plan.json")
                    or manifest["original_receipt_sha256"] != digest(source_receipt)
                    or manifest["original_journal_sha256"] != digest(source_journal)):
                    raise ValueError("Correction binding changed")
                if now >= datetime.fromisoformat(manifest["expires_at"])-timedelta(seconds=30):
                    if not args.resume:
                        print("Frozen correction window expired; not extending it")
                        return
                    reconcile_only = True
            else:
                positions,funds = broker.positions(True),broker.funds(True)
                if int(positions.get(args.code,{}).get("qty",0)) >= args.target_qty:
                    print("Account already holds at least 100 shares; no correction order needed")
                    return
                quotes = broker.quotes(sorted(set(positions)|{args.code}))
                manifest = correction_manifest(plan,starts[0],positions,funds,quotes,args.code,args.target_qty,now)
                manifest.update(original_receipt_sha256=digest(source_receipt),original_journal_sha256=digest(source_journal))
            if not args.submit:
                print(json.dumps(manifest,ensure_ascii=False,indent=2))
                print("Preview only; no correction submitted")
                return manifest
            if not path.exists():
                write_once(path,manifest)
                write_once(STATE_DIR/(stem+".authorization.json"),{k:manifest[k] for k in
                    ["environment","execution_date","model_sha256","batch_id","not_before","expires_at","user_authorization","correction_code","correction_target_quantity"]})
            result = run_manifest(broker,manifest,journal,result_path,reconcile_only)
            if result["status"] == "reconciliation_required":
                raise ReconciliationRequired(result.get("error") or "Correction requires manual reconciliation")
            return result
        finally:
            broker.close()


if __name__ == "__main__":
    main()
