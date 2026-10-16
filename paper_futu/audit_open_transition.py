"""Read-only label/model/plan verification, with no broker connection or orders."""

import json

from paper_futu import cn_common as common
from paper_futu.cn_deployment import digest, pending_model, write_json
from src.forward_labels import validate_open_label_metadata


def main():
    directory, current = common.load_deployment()
    config = json.loads(common.DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    pending = config["pending_deployment"]
    candidate_dir, candidate = pending_model(config, current)
    validate_open_label_metadata(candidate)
    archive = common.STATE_DIR / "deployment_history" / pending["transition_id"]
    protected = ["active_plan.json", "catchup_plan.json"]
    unchanged = {name: digest(common.STATE_DIR / name) == digest(archive / name) for name in protected}
    if not unchanged["active_plan.json"]:
        raise ValueError("Original active plan changed during transition preparation")
    catchup_cancelled = False
    if not unchanged["catchup_plan.json"]:
        original = json.loads((archive / "catchup_plan.json").read_text(encoding="utf-8"))
        current_catchup = json.loads((common.STATE_DIR / "catchup_plan.json").read_text(encoding="utf-8"))
        cancellation_fields = {"status", "cancelled_at", "reason"}
        unchanged_fields = {key: value for key, value in current_catchup.items() if key not in cancellation_fields}
        catchup_cancelled = current_catchup.get("status") == "cancelled" and unchanged_fields == original
        if not catchup_cancelled:
            raise ValueError("Original catchup changed beyond an explicit cancellation")
    plan = json.loads((common.STATE_DIR / "active_plan.json").read_text(encoding="utf-8"))
    common.validate_plan_deployment(plan, directory, current)
    recovery = json.loads((common.STATE_DIR / "catchup_plan.json").read_text(encoding="utf-8"))
    if recovery["model_sha256"] != current["model_sha256"] or recovery["model_dir"] != plan["model_dir"]:
        raise ValueError("Confirmed catchup no longer matches its original model")
    report = {"status": "prepared_not_activated", "label_expression": candidate["label_expression"],
              "training_information_cutoff": candidate["information_cutoff"],
              "last_mature_signal_date": candidate["last_retained_signal_date"],
              "last_label_end_date": candidate["last_label_end_date"], "train_rows": candidate["train_rows"],
              "chosen_iteration": candidate["chosen_iteration"], "model_dir": candidate_dir.relative_to(common.ROOT).as_posix(),
              "model_sha256": candidate["model_sha256"], "effective_signal_date": pending["effective_signal_date"],
              "unchanged_original_plans": unchanged, "portfolio_unchanged": current["portfolio"] == candidate["portfolio"],
              "catchup_cancelled": catchup_cancelled, "original_catchup": recovery,
              "orders_submitted_by_this_audit": 0,
              "execution_limitation": "Ideal open-price daily backtest; paper actual fills at/after 09:31 may differ"}
    write_json(archive / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
