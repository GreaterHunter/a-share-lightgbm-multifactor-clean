"""Versioned model transition at a scheduled close; this module never sends orders."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from paper_futu import cn_common as common
from src.forward_labels import OPEN_TO_OPEN, label_contract, validate_open_label_metadata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def pending_model(config, current_metadata):
    pending = config["pending_deployment"]
    if pending["previous_model_sha256"] != current_metadata["model_sha256"]:
        raise ValueError("Pending transition does not originate from the active model")
    directory = (common.ROOT / pending["model_dir"]).resolve()
    if not directory.is_relative_to((common.ROOT / "models").resolve()):
        raise ValueError("Pending model must stay inside models")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    validate_open_label_metadata(metadata)
    if digest(directory / "model.txt") != pending["model_sha256"] or pending["model_sha256"] != metadata["model_sha256"]:
        raise ValueError("Pending model hash changed")
    if pending.get("kind") == "authorized_strategy_revision":
        if (pending.get("target_strategy_id") != metadata.get("strategy_id")
                or pending.get("target_portfolio") != metadata["portfolio"]
                or pending.get("metadata_sha256") != digest(directory / "metadata.json")
                or not pending.get("user_authorization")):
            raise ValueError("Authorized strategy transition binding changed")
    elif metadata["portfolio"] != current_metadata["portfolio"]:
        raise ValueError("A label transition must not change the portfolio rules")
    return directory, metadata


def model_for_signal(signal_date):
    from paper_futu.execution_override import late_activation_allowed
    directory, metadata = common.load_deployment(require_aligned=not late_activation_allowed(signal_date, common.china_now()))
    config = json.loads(common.DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    pending = config.get("pending_deployment")
    if pending and signal_date >= pending["effective_signal_date"]:
        directory, metadata = pending_model(config, metadata)
    common.validate_serving_date(metadata, signal_date)
    return directory, metadata


def activate_pending(plan):
    """Commit config and ready plan, preserving the schedule and old recovery logs.

    Each replace is atomic. A crash between the two replacements fails the
    worker's hash guard rather than letting it submit mismatched orders.
    """
    config = json.loads(common.DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    pending = config.get("pending_deployment")
    if not pending or plan["signal_date"] < pending["effective_signal_date"]:
        return False
    from paper_futu.execution_override import late_activation_allowed
    _, current = common.load_deployment(require_aligned=not late_activation_allowed(plan["signal_date"], common.china_now()))
    directory, metadata = pending_model(config, current)
    common.validate_plan_deployment(plan, directory, metadata)
    active_path = common.STATE_DIR / "active_plan.json"
    if digest(active_path) != pending["source_plan_sha256"]:
        raise ValueError("Scheduled plan changed after transition preparation; review before activation")
    old_plan = json.loads(active_path.read_text(encoding="utf-8"))
    if plan["signal_date"] != old_plan["next_rebalance_signal_date"]:
        raise ValueError("Transition must occur on the existing rebalance signal date")
    catchup_path = common.STATE_DIR / "catchup_plan.json"
    if catchup_path.exists():
        catchup = json.loads(catchup_path.read_text(encoding="utf-8"))
        complete = common.STATE_DIR / f"catchup_complete_{catchup['execution_date']}.json"
        if catchup.get("status") != "cancelled":
            if not complete.exists():
                raise ValueError("Old-model catchup is unresolved; do not activate a different model")
            recovery = json.loads(complete.read_text(encoding="utf-8"))
            if (recovery.get("status") != "filled_all" or recovery.get("code") != catchup["code"]
                    or recovery.get("execution_date") != catchup["execution_date"]
                    or recovery.get("source_execution_date") != catchup.get("source_execution_date")
                    or not 0 < recovery.get("qty", 0) <= catchup.get("max_qty", 0)):
                raise ValueError("Old-model catchup completion is invalid; manual reconciliation required")
    archive = common.STATE_DIR / "deployment_history" / pending["transition_id"]
    archive.mkdir(parents=True, exist_ok=True)
    new_config = {key: value for key, value in config.items() if key != "pending_deployment"}
    new_config.update(model_dir=pending["model_dir"], label_price_basis=OPEN_TO_OPEN,
                      model_metadata_sha256=digest(directory / "metadata.json"))
    write_json(common.DEPLOYMENT_PATH, new_config)
    write_json(active_path, plan)
    write_json(archive / "activated.json", {"activated_at": datetime.now(timezone.utc).isoformat(),
               "new_model_sha256": metadata["model_sha256"], "signal_date": plan["signal_date"],
               "execution_date": plan["execution_date"], "orders_submitted_by_transition": 0})
    return True


def prepare_transition(as_of):
    """Package the already fitted/reload-checked model and schedule a safe switch."""
    from copy import deepcopy
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from src.factor_library import ENHANCED_FEATURE_NAMES

    as_of = pd.Timestamp(as_of).normalize()
    source = common.ROOT / f"models/experiments/cn_latest_mature_open10_{as_of:%Y%m%d}/latest"
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    contract = label_contract(10)
    for name in ["label_expression", "target", "label_horizon"]:
        if metadata[name] != contract[name]:
            raise ValueError(f"Candidate is not the aligned 10-day model: {name}")
    if (pd.Timestamp(metadata["information_cutoff"]) != as_of
            or pd.Timestamp(metadata["last_label_end_date"]) > as_of
            or metadata["features"] != ENHANCED_FEATURE_NAMES
            or digest(source / "model.txt") != metadata["model_sha256"]):
        raise ValueError("Candidate data boundary, feature order or hash is invalid")
    current_dir, current = common.load_deployment(require_aligned=False)
    config = json.loads(common.DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    if config.get("pending_deployment"):
        pending_model(config, current)
        if config["pending_deployment"]["model_sha256"] == metadata["model_sha256"]:
            return config["pending_deployment"]
        raise ValueError("Another transition is pending; do not overwrite it")
    active_path = common.STATE_DIR / "active_plan.json"
    plan = json.loads(active_path.read_text(encoding="utf-8"))
    common.validate_plan_deployment(plan, current_dir, current)
    if plan["next_rebalance_signal_date"] <= common.china_now().date().isoformat():
        raise ValueError("Scheduled transition date has already arrived")
    if metadata["portfolio"] != current["portfolio"]:
        raise ValueError("Candidate would change more than labels/training")
    dataset = common.ROOT / f"data/processed/cn_latest_mature_open10_{as_of:%Y%m%d}/supervised_dataset.parquet"
    if digest(dataset) != metadata["dataset_sha256"]:
        raise ValueError("Candidate dataset hash changed")
    features = dataset.parent / "ranked_features.parquet"
    audit_path = common.ROOT / f"reports/experiments/cn_latest_mature_open10_{as_of:%Y%m%d}/data_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if digest(features) != audit["ranked_features_sha256"]:
        raise ValueError("Latest inference feature snapshot hash changed")
    # The supervised file correctly excludes immature labels. Latest features
    # are in a separate inference snapshot and must not be taken from training.
    frame = pd.read_parquet(features, columns=ENHANCED_FEATURE_NAMES)
    rows = frame.loc[frame.index.get_level_values("datetime") == as_of].copy()
    if rows.empty or rows.index.has_duplicates or np.isinf(rows.to_numpy()).any():
        raise ValueError("Latest prediction features are invalid")
    original = lgb.Booster(model_file=str(source / "model.txt"))
    if original.feature_name() != ENHANCED_FEATURE_NAMES or original.num_trees() != metadata["chosen_iteration"]:
        raise ValueError("Candidate fitted trees or feature order do not match metadata")
    prediction = original.predict(rows, num_threads=4)
    if not np.isfinite(prediction).all():
        raise ValueError("Candidate predictions are abnormal")
    directory = common.ROOT / f"models/paper_futu/cn_open_to_open_10d_{as_of:%Y%m%d}"
    directory.mkdir(parents=True, exist_ok=True)
    target_model = directory / "model.txt"
    if target_model.exists() and digest(target_model) != metadata["model_sha256"]:
        raise ValueError("Do not overwrite another prepared model")
    shutil.copy2(source / "model.txt", target_model)
    restored = lgb.Booster(model_file=str(target_model)).predict(rows, num_threads=4)
    if not np.allclose(prediction, restored, rtol=0, atol=1e-12):
        raise ValueError("Packaged predictions differ from the fitted model")
    candidate = {**deepcopy(metadata), **contract, "portfolio": deepcopy(current["portfolio"]),
                 "source_model_dir": source.relative_to(common.ROOT).as_posix(),
                 "deployment_status": "prepared_for_next_scheduled_rebalance",
                 "actual_execution_window": "09:31 continuous-session limit orders; not guaranteed opening price"}
    validate_open_label_metadata(candidate)
    write_json(directory / "metadata.json", candidate)
    transition_id = f"open_labels_{as_of:%Y%m%d}_{metadata['model_sha256'][:12]}"
    archive = common.STATE_DIR / "deployment_history" / transition_id
    archive.mkdir(parents=True, exist_ok=True)
    for path in [common.DEPLOYMENT_PATH, active_path, common.STATE_DIR / "catchup_plan.json"]:
        if path.exists() and not (archive / path.name).exists():
            shutil.copy2(path, archive / path.name)
    pending = {"transition_id": transition_id, "model_dir": directory.relative_to(common.ROOT).as_posix(),
               "model_sha256": candidate["model_sha256"], "previous_model_sha256": current["model_sha256"],
               "source_plan_sha256": digest(active_path), "effective_signal_date": plan["next_rebalance_signal_date"],
               "label_price_basis": OPEN_TO_OPEN, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "preserve_existing_positions": True, "preserve_confirmed_catchup": True,
               "latest_prediction_rows": len(rows), "saved_prediction_check_max_abs_error": float(np.max(np.abs(prediction-restored)))}
    config.update(label_price_basis=current.get("label_price_basis", "legacy_open_to_close"), pending_deployment=pending)
    write_json(common.DEPLOYMENT_PATH, config)
    write_json(archive / "prepared.json", pending)
    common.load_deployment()  # Verify the transitional config without contacting OpenD.
    pending_model(config, current)
    return pending
