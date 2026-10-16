"""Package the audited S6 QREG model and prepare an explicitly authorized switch.

No broker/order API is imported. Preparing a transition preserves holdings and
the cancelled recovery order; activation happens at the next five-session close.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import shutil

from paper_futu import cn_common as common
from paper_futu.cn_deployment import digest, write_json, pending_model
from src.forward_labels import label_contract, validate_open_label_metadata

NAME = "s6_quarterly_l1l2_small_asof_20261013"
STRATEGY = "S6_head6_40_QREG"
MODEL_REL = "models/paper_futu/s6_qreg_20260930"
RESEARCH = "configs/experiments/s6_head6_40_research_main.json"
PORTFOLIO = dict(top_k=20, n_drop=6, top_tier_size=6, top_tier_weight=.4,
                 risk_degree=.95, rebalance_every=5)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def maintenance_lock():
    """Use the worker's OS lock, not an unreliable stale PID file."""
    path = common.STATE_DIR / "worker.lock"
    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Trading worker is active; do not revise its deployment concurrently") from error
        try:
            yield
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def package():
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from src.factor_library import ENHANCED_FEATURE_NAMES

    root = common.ROOT
    report = root / "reports/experiments" / NAME
    complete, verification = read(report / "completion.json"), read(report / "final_verification.json")
    if complete["status"] != "complete" or verification["status"] != "passed":
        raise ValueError("Selected experiment is not completed and audited")
    if digest(report / "report_manifest.json") != complete["manifest_sha256"]:
        raise ValueError("Selected research manifest changed")
    # Do not rewrite historical protection records after the authorized deployment.
    for path, expected in read(report / "report_manifest.json")["artifacts"].items():
        if digest(path) != expected:
            raise ValueError(f"Selected research artifact changed: {path}")
    source = root / "models/experiments" / NAME / "2026-09-30"
    fitted = read(source / "metadata.json")
    if digest(source / "model.txt") != fitted["model_sha256"] or fitted["label_end_max"] > fitted["cutoff"]:
        raise ValueError("Selected model hash or label boundary failed")
    params = fitted["parameters"]
    if any(params[k] != v for k,v in dict(reg_alpha=1.,reg_lambda=5.,num_leaves=15,min_child_samples=300).items()):
        raise ValueError("Selected model is not the requested joint variant")
    data = root / "data/processed/final_ablation_asof_20261013"
    manifest = read(data / "manifest.json")
    finance = data / "financial_events.parquet"
    if digest(finance) != manifest["artifacts"]["financial_events.parquet"]:
        raise ValueError("Financial snapshot hash changed")
    target = root / MODEL_REL
    target.mkdir(parents=True, exist_ok=True)
    if (target / "model.txt").exists() and digest(target / "model.txt") != fitted["model_sha256"]:
        raise ValueError("Refusing to overwrite a different packaged model")
    shutil.copy2(source / "model.txt", target / "model.txt")
    metadata = {**fitted, **label_contract(10), "strategy_id":STRATEGY,"model_year":2026,
        "features":ENHANCED_FEATURE_NAMES,"input_kind":"raw","target_kind":"rank",
        "retrain_months":3,"information_cutoff":fitted["cutoff"],
        "serving_signal_end":"2026-12-31","next_refit_cutoff":"2026-12-31",
        "refresh_policy":"fresh quarterly fits; pre-year three-fold tree choice fixed within year",
        "quarter_refresh":"validated model required before serving the next quarter; fail closed if absent",
        "portfolio":PORTFOLIO,"chosen_iteration":fitted["trees"],
        "selection_rule":"audited_limited_replacement_v1",
        "financial_events_path":finance.relative_to(root).as_posix(),
        "financial_events_sha256":digest(finance),"financial_latest_report":"2026-06-30",
        "financial_updates":"pinned verified events; no unverified Q3 values added",
        "financial_revision_archive_complete":False,"snapshot_id":manifest["snapshot_id"],
        "source_model_dir":source.relative_to(root).as_posix(),
        "source_metadata_sha256":digest(source / "metadata.json"),
        "research_protocol_sha256":digest(report / "protocol.json"),
        "actual_execution_window":"next session 09:31 limit orders; not guaranteed opening-auction price",
        "live_sizing":"live NAV and live quote; differs from idealized previous-close sizing in research",
        "historical_worst_drawdown":.303473,"historical_30pct_risk_flag":True}
    validate_open_label_metadata(metadata)
    if (target / "metadata.json").exists() and read(target / "metadata.json") != metadata:
        raise ValueError("Packaged metadata differs; create a new version")
    write_json(target / "metadata.json",metadata)
    feature_path = data / "raw_features.parquet"
    if digest(feature_path) != manifest["artifacts"]["raw_features.parquet"]:
        raise ValueError("Inference snapshot changed")
    frame = pd.read_parquet(feature_path,columns=ENHANCED_FEATURE_NAMES)
    last = frame.loc[frame.index.get_level_values("datetime")=="2026-10-13"]
    original = lgb.Booster(model_file=str(source / "model.txt"))
    restored = lgb.Booster(model_file=str(target / "model.txt"))
    if restored.feature_name() != ENHANCED_FEATURE_NAMES or restored.num_trees() != 40:
        raise ValueError("Unexpected feature order or tree count")
    a,b = original.predict(last,num_threads=4),restored.predict(last,num_threads=4)
    if not np.isfinite(b).all(): raise ValueError("Non-finite deployed prediction")
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
    write_json(root / "reports/paper_futu/s6_qreg_switch/package_verification.json",
        dict(status="passed",rows=len(last),model_sha256=fitted["model_sha256"],
             metadata_sha256=digest(target / "metadata.json"),reload_max_error=float(np.max(np.abs(a-b))),
             source_research_unchanged=True,orders_submitted=0))
    return metadata


def next_signal(old_plan, calendar, today):
    """Keep the original signal anchor; do not restart a cadence on each year."""
    dates = sorted(set(calendar))
    index = dates.index(old_plan["signal_date"])
    scheduled = dates[index + PORTFOLIO["rebalance_every"]::PORTFOLIO["rebalance_every"]]
    future = [d for d in scheduled if d > today]
    if not future: raise ValueError("Trading calendar does not contain the next rebalance")
    return future[0]


def prepare():
    metadata = package()
    verify_readiness()
    root = common.ROOT
    readiness = root / "reports/paper_futu/s6_qreg_switch/readiness_20261015/plan.json"
    plan = read(readiness)
    common.validate_plan_deployment(plan,(root / MODEL_REL).resolve(),metadata)
    if plan["signal_date"] != "2026-10-15" or len(plan["ordered_candidates"]) < 20:
        raise ValueError("Latest saved-day readiness check is missing")
    with maintenance_lock():
        current_dir,current = common.load_deployment()
        config = read(common.DEPLOYMENT_PATH)
        existing = config.get("pending_deployment",{})
        if config["model_dir"] == MODEL_REL or existing.get("target_strategy_id") == STRATEGY:
            if existing: pending_model(config,current)
            return status()
        active = common.STATE_DIR / "active_plan.json"
        old = read(active)
        common.validate_plan_deployment(old,current_dir,current)
        if not (common.STATE_DIR / f"execution_{old['execution_date']}.json").exists():
            raise ValueError("Previous execution is not reconciled")
        catchup = common.STATE_DIR / "catchup_plan.json"
        if catchup.exists() and read(catchup).get("status") != "cancelled":
            raise ValueError("Resolve the old catchup before preparing a different strategy")
        calendar = (common.HISTORICAL_PROVIDER / "calendars/day_future.txt").read_text().splitlines()
        effective = next_signal(old,calendar,common.china_now().date().isoformat())
        common.validate_serving_date(metadata,effective)
        transition_id = f"s6_qreg_{metadata['model_sha256'][:12]}_{effective.replace('-','')}"
        archive = common.STATE_DIR / "deployment_history" / transition_id
        archive.mkdir(parents=True,exist_ok=True)
        paths = [common.DEPLOYMENT_PATH,active,catchup,root / RESEARCH]
        before = {}
        for path in paths:
            if path.exists():
                shutil.copy2(path,archive / path.name)
                before[path.relative_to(root).as_posix()] = digest(path)
        old["next_rebalance_signal_date"] = effective
        old["schedule_transition_id"] = transition_id
        pending = dict(kind="authorized_strategy_revision",transition_id=transition_id,
            model_dir=MODEL_REL,model_sha256=metadata["model_sha256"],
            metadata_sha256=digest(root / MODEL_REL / "metadata.json"),
            previous_model_sha256=current["model_sha256"],effective_signal_date=effective,
            target_strategy_id=STRATEGY,target_portfolio=PORTFOLIO,
            user_authorization="Select this quarterly L1/L2 smaller-tree S6 as main model and update CN paper trading",
            preserve_existing_positions=True,preserve_cancelled_catchup=True,
            superseded_pending_deployment=existing,created_at_utc=datetime.now(timezone.utc).isoformat())
        # Write plan first: a crash before config replacement fails closed because
        # its source-plan hash no longer agrees with the superseded transition.
        write_json(active,old)
        pending["source_plan_sha256"] = digest(active)
        write_json(common.DEPLOYMENT_PATH,{**config,"pending_deployment":pending,
            "selected_main_strategy":STRATEGY,"selected_main_model_dir":MODEL_REL})
        main = dict(strategy_id=STRATEGY,research_main=True,variant="QREG",retrain_months=3,
            model_path=str(root / MODEL_REL / "model.txt"),model_sha256=metadata["model_sha256"],
            model_cutoff=metadata["information_cutoff"],snapshot_id=metadata["snapshot_id"],
            portfolio=PORTFOLIO,rebalance=5,risk_degree=.95,seed=42,horizon=10,
            input_kind="raw",target_kind="rank",label="Open(T+11)/Open(T+1)-1",
            parameters=metadata["parameters"],chosen_iteration=40,
            paper_deployment="prepared_for_scheduled_transition",effective_signal_date=effective,
            first_execution_date=calendar[calendar.index(effective)+1],
            historical_30pct_drawdown_limit_exceeded=True,
            independent_test=False,deployment_changed=True)
        write_json(root / RESEARCH,main)
        record = dict(status="prepared",transition=pending,research_main=main,before=before,
            after={path.relative_to(root).as_posix():digest(path) for path in paths if path.exists()},
            orders_submitted_by_transition=0,old_research_records_preserved=True,
            broker_validation="OpenD unavailable at preparation; online read-only preview still required")
        write_json(archive / "prepared.json",record)
        write_json(root / "reports/paper_futu/s6_qreg_switch/transition.json",record)
        common.load_deployment()
        pending_model(read(common.DEPLOYMENT_PATH),current)
        if catchup.exists() and digest(catchup) != before[catchup.relative_to(root).as_posix()]:
            raise ValueError("Cancelled catchup was unexpectedly changed")
        return main


def status():
    config = read(common.DEPLOYMENT_PATH)
    directory,metadata = common.load_deployment()
    plan = read(common.STATE_DIR / "active_plan.json")
    common.validate_plan_deployment(plan,directory,metadata)
    pending = config.get("pending_deployment")
    if pending: pending_model(config,metadata)
    return dict(active_model=config["model_dir"],active_model_sha256=metadata["model_sha256"],
        selected_main=config.get("selected_main_strategy"),pending=pending,
        next_signal=plan["next_rebalance_signal_date"],orders_submitted_by_this_command=0)


def verify_readiness():
    """Compare the actual daily adapter with frozen research, not just its formula."""
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from src.factor_library import ENHANCED_FEATURE_NAMES as names

    root = common.ROOT
    out = root / "reports/paper_futu/s6_qreg_switch"
    directory = root / MODEL_REL
    metadata = read(directory / "metadata.json")
    if digest(directory / "model.txt") != metadata["model_sha256"]:
        raise ValueError("Packaged model changed")
    frozen = root / "data/processed/final_ablation_asof_20261013"
    manifest = read(frozen / "manifest.json")
    if digest(frozen / "raw_features.parquet") != manifest["artifacts"]["raw_features.parquet"]:
        raise ValueError("Frozen raw feature snapshot changed")
    reference = pd.read_parquet(frozen / "raw_features.parquet",columns=names)
    reference = reference.loc[reference.index.get_level_values("datetime")=="2026-10-13"]
    actual = pd.read_parquet(out / "parity_20261013/features_raw.parquet")[names].astype("float32")
    if actual.index.has_duplicates or np.isinf(actual.to_numpy()).any():
        raise ValueError("Daily adapter has duplicate or infinite features")
    common_index = reference.index.intersection(actual.index)
    if len(common_index) != len(actual) or len(actual) < 290:
        raise ValueError("Unexpected daily-adapter evaluation sample")
    left,right = reference.loc[common_index],actual.loc[common_index]
    differences = (left-right).abs().max()
    if (left.isna()!=right.isna()).any().any() or (differences > 1e-5).any():
        raise ValueError("Live features differ from the selected research snapshot")
    model = lgb.Booster(model_file=str(directory / "model.txt"))
    a,b = model.predict(left,num_threads=4),model.predict(right,num_threads=4)
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    plan = read(out / "readiness_20261015/plan.json")
    common.validate_plan_deployment(plan,directory.resolve(),metadata)
    # Cached-position preview is deliberately separate from a broker-verified
    # execution preview. Never interpret yesterday's sellable amount as current.
    positions = pd.read_csv(common.STATE_DIR / "positions_latest.csv")
    current = positions.loc[positions.qty > 0,"code"].tolist()
    selected = common.select_audited_holdings(plan["predictions"],plan["ordered_candidates"],
        current,20,6,current)
    weights = common.tiered_weights(selected,top_k=20,top_tier_size=6,top_tier_weight=.4)
    result = dict(status="passed",model_sha256=metadata["model_sha256"],feature_date="2026-10-13",
        comparison_rows=len(actual),reference_rows=len(reference),
        excluded_suspended_stocks=sorted(set(reference.index.get_level_values("instrument"))-
                                        set(actual.index.get_level_values("instrument"))),
        maximum_feature_difference=differences.to_dict(),missing_masks_equal=True,
        prediction_max_abs_error=float(np.max(np.abs(a-b))),
        latest_readiness_signal_date=plan["signal_date"],latest_prediction_rows=len(plan["predictions"]),
        cached_holdings=len(current),cached_preview_holdings=len(selected),
        cached_preview_replacements=len(set(current)-set(selected)),
        cached_preview_selected=selected,cached_preview_weights=weights,
        preview_is_broker_verified=False,orders_submitted=0,
        artifacts={str(path.relative_to(root)):digest(path) for path in
            [directory / "model.txt",directory / "metadata.json",out / "readiness_20261015/plan.json",
             out / "parity_20261013/features_raw.parquet"]},
        deployment_code_sha256={name:digest(root / name) for name in [
            "paper_futu/deploy_s6_quarterly.py","paper_futu/cn_common.py","paper_futu/cn_deployment.py",
            "paper_futu/build_cn_signal.py","paper_futu/execute_cn_plan.py","paper_futu/cn_worker.py",
            "scripts/install_cn_paper_schedule.ps1","src/run_main_pipeline.py"]})
    write_json(out / "readiness_verification.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["package","verify","prepare","status"],default="status")
    args = parser.parse_args()
    result = {"package":package,"verify":verify_readiness,"prepare":prepare,"status":status}[args.stage]()
    print(json.dumps(result,indent=2,ensure_ascii=True))


if __name__ == "__main__":
    main()
