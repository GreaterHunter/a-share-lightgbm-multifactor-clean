"""Freeze S6 rules and observe existing paper-account files. Never sends orders."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_futu.benchmark_attribution import fetch_official, official_frame, digest, save_json

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))
STRATEGY = "S6_head6_40_QREG"
PORTFOLIO = dict(top_k=20, n_drop=6, top_tier_size=6, top_tier_weight=.4, risk_degree=.95, rebalance_every=5)
CODE_FILES = [
    "paper_futu/prospective_s6.py", "paper_futu/benchmark_attribution.py",
    "paper_futu/cn_common.py", "paper_futu/cn_worker.py", "paper_futu/cn_deployment.py",
    "paper_futu/build_cn_signal.py", "paper_futu/fetch_cn_market.py",
    "paper_futu/execute_cn_plan.py", "src/factor_library.py", "src/forward_labels.py",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)


def verify_metadata(meta):
    if meta.get("strategy_id") != STRATEGY or meta.get("portfolio") != PORTFOLIO:
        raise ValueError("Model/portfolio is not the selected frozen S6")
    expected = dict(reg_alpha=1., reg_lambda=5., num_leaves=15, min_child_samples=300, n_jobs=4)
    if any(meta["parameters"].get(k) != v for k,v in expected.items()):
        raise ValueError("Training constraints changed")
    if (meta.get("input_kind") != "raw" or meta.get("target_kind") != "rank"
            or meta.get("label_expression") != "Ref($open, -11) / Ref($open, -1) - 1"
            or meta.get("retrain_months") != 3 or meta.get("seed") != 42
            or len(meta.get("features", [])) != 15 or not meta.get("reload_equal")):
        raise ValueError("Model information or label contract changed")
    if meta["label_end_max"] > meta["information_cutoff"]:
        raise ValueError("Training labels cross the model cutoff")


def verify_attribution(path):
    path = Path(path)
    manifest = read(path)
    if manifest.get("status") != "complete" or manifest.get("strategy_id") != STRATEGY:
        raise ValueError("Complete total-return attribution is required before freezing")
    if manifest["alignment"]["missing_dates"] != 0:
        raise ValueError("Incomplete total-return benchmark")
    for name, expected in manifest["artifacts"].items():
        if digest(path.parent/name) != expected:
            raise ValueError("Attribution artifact changed")
    return manifest


def freeze(root, output, attribution, now=None):
    root, output, attribution = Path(root).resolve(), Path(output).resolve(), Path(attribution).resolve()
    now = now or datetime.now(TZ)
    protocol_path = output/"protocol.json"
    if protocol_path.exists():
        verify_protocol(output)
        return dict(status="already_frozen", protocol_sha256=digest(protocol_path))
    attribution_manifest = verify_attribution(attribution)
    main = read(root/"configs/experiments/s6_head6_40_research_main.json")
    deployment = read(root/"configs/futu_cn_paper.json")
    pending = deployment.get("pending_deployment", {})
    if deployment.get("environment") != "SIMULATE" or deployment.get("market") != "CN":
        raise ValueError("Only the existing CN simulated deployment can be observed")
    model_dir = Path(main["model_path"]).resolve().parent
    if not model_dir.is_relative_to((root/"models").resolve()):
        raise ValueError("Model must stay inside the project")
    meta = read(model_dir/"metadata.json")
    verify_metadata(meta)
    if digest(root/meta["financial_events_path"]) != meta["financial_events_sha256"]:
        raise ValueError("Pinned financial event source differs from the model metadata")
    if pending.get("metadata_sha256") != digest(model_dir/"metadata.json"):
        raise ValueError("Scheduled deployment metadata changed")
    if meta["chosen_iteration"] != main["chosen_iteration"]:
        raise ValueError("Registered tree count differs from the selected model")
    if digest(model_dir/"model.txt") != meta["model_sha256"] or meta["model_sha256"] != main["model_sha256"]:
        raise ValueError("Model hash mismatch")
    if pending.get("model_sha256") != meta["model_sha256"] or pending.get("target_portfolio") != PORTFOLIO:
        raise ValueError("Existing scheduled transition differs; do not change it implicitly")
    signal, execution = main["effective_signal_date"], main["first_execution_date"]
    if now.date().isoformat() >= signal:
        raise ValueError("Freeze before the scheduled signal; retrospective enrollment is prohibited")
    calendar_path = root/"data/interim/futu_cn_provider/2026-10-11/extracted/qlib_bin/calendars/day_future.txt"
    dates = sorted(calendar_path.read_text().splitlines())
    if signal not in dates or execution not in dates or dates[dates.index(signal)+1] != execution:
        raise ValueError("Scheduled execution is not the next verified trading day")
    immutable = {str(root/name):digest(root/name) for name in CODE_FILES}
    for path in [model_dir/"model.txt", model_dir/"metadata.json",
                 root/meta["financial_events_path"], attribution]:
        immutable[str(path)] = digest(path)
    version = dict(strategy_id=STRATEGY, model_sha256=meta["model_sha256"],
                   metadata_sha256=digest(model_dir/"metadata.json"), model_dir=str(model_dir.relative_to(root)),
                   features=meta["features"], parameters=meta["parameters"], chosen_iteration=meta["chosen_iteration"],
                   information_cutoff=meta["information_cutoff"], label_end_max=meta["label_end_max"],
                   serving_signal_end=meta["serving_signal_end"], financial_events_sha256=meta["financial_events_sha256"])
    protocol = dict(version=1, frozen_at_utc=now.astimezone(timezone.utc).isoformat(), project_root=str(root),
        strategy=version, portfolio=PORTFOLIO, signal_anchor=signal, first_execution_date=execution,
        baseline_close_date=signal, initial_account="existing paper account with inherited holdings; not fresh cash",
        primary_benchmark="H00300", secondary_benchmark="000300", risk_free_annual=.01,
        trading_dates=[d for d in dates if signal <= d <= meta["serving_signal_end"]],
        label="Open(T+11)/Open(T+1)-1; daily cross-sectional rank minus 0.5",
        training_policy="Expanding 2010+ mature samples; fresh quarterly fits; prior three full-year folds select trees once per year",
        tree_candidates=[10,20,40,80,160,320,640],
        quarterly_update="New validated model package required before next quarter; never silently reuse expired model",
        financial_policy="Pinned verified event snapshot; any refreshed financial dataset requires a new audited version before use",
        scheduled_execution="Existing 09:31 continuous-session limit orders; not ideal opening-auction fills",
        cash_flows="No deposits/withdrawals during this version. Any external flow suspends performance attribution for review.",
        fee_policy="Broker asset NAV is observed without double deduction. Modeled fees shown separately; actual broker fee coverage unverified.",
        review_sessions=[20,63,126], risk_review_drawdown=.30,
        risk_action="Review alert only; observer cannot liquidate, cancel, place orders, or alter deployment",
        stop_rule="No retrospective optimization; changed rules require a new timestamped forward version",
        known_history="2022-2026 repeatedly researched, not an independent test",
        attribution_manifest_sha256=digest(attribution), attribution_as_of=attribution_manifest["as_of"],
        immutable_bindings=immutable)
    write_once(protocol_path, protocol)
    write_once(output/"seal.json",dict(protocol_sha256=digest(protocol_path)))
    return dict(status="frozen_awaiting_first_execution", first_execution_date=execution,
                protocol_sha256=digest(protocol_path), model_sha256=meta["model_sha256"])


def verify_protocol(output):
    output = Path(output)
    if digest(output/"protocol.json") != read(output/"seal.json")["protocol_sha256"]:
        raise ValueError("Frozen protocol changed; create a new version instead")
    p = read(output/"protocol.json")
    for path, expected in p["immutable_bindings"].items():
        if digest(path) != expected:
            raise ValueError("Frozen code/data/model changed: " + Path(path).name)
    return p


def closing_marker(path, day):
    marker = read(path)
    stamp = datetime.fromisoformat(marker["time"])
    if stamp.tzinfo is None or stamp.astimezone(TZ).date().isoformat() != day:
        raise ValueError("Closing observation date/timezone mismatch")
    local = stamp.astimezone(TZ)
    if (local.hour, local.minute) < (15,35):
        raise ValueError("Do not use an intraday balance as the closing observation")
    for name in ["assets","cash"]:
        if not np.isfinite(marker[name]):
            raise ValueError("Nonfinite account balance")
    if marker["assets"] <= 0 or marker["holdings"] < 0:
        raise ValueError("Invalid account observation")
    return marker


def filled_costs(path):
    if not Path(path).exists():
        return dict(filled_notional=0., modeled_fees_cny=0., orders_with_fills=0)
    orders = pd.read_csv(path)
    if orders.order_id.astype(str).duplicated().any():
        raise ValueError("Duplicate order receipt")
    fills = orders.loc[pd.to_numeric(orders.dealt_qty).gt(0)].copy()
    if len(fills) and not fills.trd_side.isin(["BUY","SELL"]).all():
        raise ValueError("Unknown trade side")
    notional = pd.to_numeric(fills.dealt_qty)*pd.to_numeric(fills.dealt_avg_price)
    if not np.isfinite(notional).all() or (notional <= 0).any():
        raise ValueError("Invalid fill notional")
    rates = np.where(fills.trd_side.eq("BUY"), .0005, .0015)
    return dict(filled_notional=float(notional.sum()),
                modeled_fees_cny=float(np.maximum(5., notional*rates).sum()), orders_with_fills=len(fills))


def record(output, day=None, now=None):
    output = Path(output)
    p = verify_protocol(output)
    now = now or datetime.now(TZ)
    day = day or now.astimezone(TZ).date().isoformat()
    if day > now.astimezone(TZ).date().isoformat():
        raise ValueError("Cannot record future account results")
    if day < p["first_execution_date"]:
        return dict(status="awaiting_first_execution", first_execution_date=p["first_execution_date"])
    if day > p["strategy"]["serving_signal_end"]:
        raise ValueError("Frozen quarter expired; a validated next-quarter package is required")
    if day not in p["trading_dates"]:
        return dict(status="non_trading_day", date=day)
    path = output/"daily"/(day+".json")
    if path.exists():
        validate_records(output, p)
        return dict(status="already_recorded", date=day, record_sha256=digest(path))
    root = Path(p["project_root"])
    state = root/"data/interim/futu_cn_paper"
    deployment = read(root/"configs/futu_cn_paper.json")
    meta_path = root/deployment["model_dir"]/"metadata.json"
    meta = read(meta_path)
    verify_metadata(meta)
    if (digest(meta_path) != p["strategy"]["metadata_sha256"]
            or digest(meta_path.parent/"model.txt") != p["strategy"]["model_sha256"]):
        raise ValueError("Actual deployment package differs from the frozen package")
    if deployment.get("environment") != "SIMULATE" or meta["model_sha256"] != p["strategy"]["model_sha256"]:
        raise ValueError("The frozen S6 is not active; no legacy-account result is relabeled as S6")
    plan = read(state/day/"plan.json")
    if plan.get("model_sha256") != meta["model_sha256"]:
        raise ValueError("Daily plan does not belong to the frozen model")
    if any(plan.get(k) != v for k,v in PORTFOLIO.items()):
        raise ValueError("Daily portfolio rule changed")
    marker_path = state/("nav_saved_"+day+".json")
    marker = closing_marker(marker_path,day)
    if marker["holdings"] > PORTFOLIO["top_k"]:
        raise ValueError("Account exceeds the frozen holding limit")
    position = p["trading_dates"].index(day)
    previous_day = p["trading_dates"][position-1]
    previous_path = output/"daily"/(previous_day+".json")
    if previous_day == p["baseline_close_date"]:
        baseline_path = output/"baseline.json"
        source = state/("nav_saved_"+previous_day+".json")
        baseline = closing_marker(source,previous_day)
        if not baseline_path.exists():
            write_once(baseline_path,dict(date=previous_day,assets=baseline["assets"],
                                         source_sha256=digest(source), legacy_holdings_carried=True))
        previous_assets = read(baseline_path)["assets"]
        previous_hash = digest(baseline_path)
    else:
        if not previous_path.exists():
            raise ValueError("Missing prior trading-day record; do not bridge or fill the gap")
        validate_records(output,p)
        previous_assets = read(previous_path)["assets"]
        previous_hash = digest(previous_path)
    # Existing active plan identifies the order cycle; this observer never calls its executor.
    active = read(state/"active_plan.json")
    if active.get("model_sha256") != meta["model_sha256"]:
        raise ValueError("Active trading plan/model mismatch")
    execution_path = state/("execution_"+day+".json")
    order_path = state/("order_results_"+day+".csv")
    if active.get("execution_date") == day:
        if not execution_path.exists():
            raise ValueError("Scheduled execution lacks a completion receipt")
        execution = read(execution_path)
        if execution.get("environment") != "SIMULATE":
            raise ValueError("Only simulated execution receipts are accepted")
        if execution.get("submitted_orders",0) and not order_path.exists():
            raise ValueError("Filled/unfilled order receipts are missing")
    raw_dir = output/"benchmark_raw"/day
    fetch_official(raw_dir, previous_day, day)
    returns={}
    for code in ["000300","H00300"]:
        frame=official_frame(read(raw_dir/(code+".json")),code)
        returns[code]=float(frame.loc[day,"close"]/frame.loc[previous_day,"close"]-1)
    net=marker["assets"]/previous_assets-1
    evidence={marker_path.name:digest(marker_path),"daily_plan":digest(state/day/"plan.json"),
              "active_plan":digest(state/"active_plan.json"),"benchmark_sources":digest(raw_dir/"sources.json")}
    if execution_path.exists(): evidence["execution"]=digest(execution_path)
    if order_path.exists(): evidence["order_receipts"]=digest(order_path)
    result=dict(date=day, observed_at=marker["time"], assets=marker["assets"], cash=marker["cash"],
        holdings=marker["holdings"], broker_nav_return=net, total_return_benchmark=returns["H00300"],
        price_benchmark=returns["000300"], model_sha256=meta["model_sha256"],
        cash_flow_assumption="zero by frozen no-external-flow policy; not independently broker-verified",
        fee_treatment="Model estimates reported separately; no second deduction from broker NAV",
        **filled_costs(order_path), previous_record_sha256=previous_hash, evidence=evidence,
        inherited_holdings_transition=(day==p["first_execution_date"]))
    write_once(path,result)
    validate_records(output,p)
    return dict(status="recorded",date=day,holdings=marker["holdings"],record_sha256=digest(path))


def validate_records(output, protocol):
    output=Path(output)
    previous=output/"baseline.json"
    expected=protocol["trading_dates"][1:]
    for index,path in enumerate(sorted((output/"daily").glob("*.json"))):
        row=read(path)
        if index>=len(expected) or row["date"]!=expected[index] or path.stem!=row["date"]:
            raise ValueError("Duplicate or missing forward trading-day observation")
        if not previous.exists() or row["previous_record_sha256"]!=digest(previous):
            raise ValueError("Forward ledger hash chain changed")
        if row["model_sha256"]!=protocol["strategy"]["model_sha256"]:
            raise ValueError("Forward ledger model changed")
        prior=read(previous)
        if not np.isclose(row["broker_nav_return"],row["assets"]/prior["assets"]-1,atol=1e-12):
            raise ValueError("Forward account return cannot be reproduced")
        previous=path


def status(output):
    output=Path(output)
    p=verify_protocol(output)
    validate_records(output,p)
    rows=[read(path) for path in sorted((output/"daily").glob("*.json"))]
    result=dict(status="observing" if rows else "frozen_awaiting_first_execution",
        first_execution_date=p["first_execution_date"], recorded_sessions=len(rows),
        model_sha256=p["strategy"]["model_sha256"],primary_benchmark="H00300",
        observer_permissions="read local broker receipts; fetch public indices; no broker connection or order submission")
    if rows:
        r=np.array([v["broker_nav_return"] for v in rows]);b=np.array([v["total_return_benchmark"] for v in rows])
        nav=np.cumprod(1+r);peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
        result.update(observed_account_return=float(nav[-1]-1),benchmark_return=float(np.prod(1+b)-1),
            geometric_excess=float(nav[-1]/np.prod(1+b)-1),max_drawdown=float(-(nav/peak-1).min()),
            modeled_fees_cny=sum(v["modeled_fees_cny"] for v in rows),broker_fee_coverage_verified=False,
            risk_review_required=bool(-(nav/peak-1).min()>=p["risk_review_drawdown"]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["freeze","record","status"],default="status")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/paper_futu/s6_forward_v1")
    parser.add_argument("--attribution",type=Path)
    parser.add_argument("--date")
    args=parser.parse_args()
    try:
        if args.stage=="freeze":
            if not args.attribution:parser.error("--attribution manifest is required")
            result=freeze(ROOT,args.output,args.attribution)
        elif args.stage=="record":result=record(args.output,args.date)
        else:result=status(args.output)
        save_json(args.output/"observer_status.json",dict(time=datetime.now(TZ).isoformat(),**result))
        if __import__("sys").stdout is not None:print(json.dumps(result,ensure_ascii=False))
    except Exception as error:
        save_json(args.output/"observer_status.json",dict(time=datetime.now(TZ).isoformat(),status="blocked",reason=str(error)))
        raise


if __name__=="__main__":
    main()

