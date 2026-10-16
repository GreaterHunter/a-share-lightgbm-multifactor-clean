"""Audit every prepared full-15-factor account and publish its five-phase ranking."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_futu import final_ablation_core as core
from scripts.compare_all_full15_offsets import OUTPUT, SOURCE, source_account_config


REPORT_PATH = OUTPUT / "all_full15_phase_ranking.csv"


@lru_cache(maxsize=None)
def file_hash(path: str) -> str:
    return core.digest(path)


def main() -> None:
    manifest = core.read_json(OUTPUT / "manifest.json")
    if core.code_hash() != manifest["scientific_code_sha256"]:
        raise ValueError("Scientific code differs from the frozen experiment")
    comparison_script = Path(__file__).with_name("compare_all_full15_offsets.py")
    if core.digest(comparison_script) != manifest["comparison_script_sha256"]:
        raise ValueError("Comparison script differs from the prepared manifest")
    if core.digest(SOURCE / "model_selection.json") != manifest["model_selection_sha256"]:
        raise ValueError("Frozen model selection changed")
    if any(core.digest(SOURCE / name) != expected
           for name, expected in manifest["source_sha256"].items()):
        raise ValueError("A frozen source table changed")

    store = core.Store()
    if store.data != manifest["snapshot_id"]:
        raise ValueError("Frozen data snapshot differs from the prepared manifest")

    records = []
    missing = []
    observed_case_ids: set[str] = set()
    for candidate in manifest["candidates"]:
        if len(candidate["offsets"]) != 5 or len(candidate["case_ids"]) != 5:
            raise ValueError("Each candidate must have five prescribed phases")
        config = source_account_config(candidate["case_ids"][0])
        if file_hash(config["prediction_path"]) != config["prediction_sha256"]:
            raise ValueError(f"Prediction artifact changed: {config['prediction_path']}")
        returns = []
        drawdowns = []
        fees = []
        for offset, case_id in zip(candidate["offsets"], candidate["case_ids"]):
            expected_config = {**config, "offset": int(offset)}
            if store.key("account", expected_config) != case_id:
                raise ValueError(f"Account identity mismatch: {case_id}")
            observed_case_ids.add(case_id)
            core.REPORT = SOURCE if case_id in manifest["known"] else OUTPUT
            result = store.cached("account", expected_config)
            if result is None:
                missing.append(case_id)
                continue
            full_period = [row for row in result["rows"] if row["period"] == "all"]
            if len(full_period) != 1:
                raise ValueError(f"Expected one full-period account row: {case_id}")
            row = full_period[0]
            if row["case_id"] != case_id or int(row["offset"]) != int(offset):
                raise ValueError(f"Account row does not match its phase: {case_id}")
            annualized = float(row["annualized_return"])
            drawdown = -float(row["max_drawdown"])
            fee = float(row["total_fees_cny"])
            if not all(np.isfinite(value) for value in (annualized, drawdown, fee)):
                raise ValueError(f"Non-finite account metric: {case_id}")
            if drawdown < 0 or fee < 0:
                raise ValueError(f"Invalid drawdown or fee: {case_id}")
            returns.append(annualized)
            drawdowns.append(drawdown)
            fees.append(fee)
        if len(returns) != 5:
            continue
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=0))
        records.append({
            "candidate_key": candidate["candidate_key"],
            "spec": candidate["spec"],
            "horizon": candidate["horizon"],
            "rebalance": candidate["rebalance"],
            "policy": candidate["policy"],
            "months": candidate["months"],
            **candidate["portfolio"],
            "phase_count": 5,
            "mean_cagr": mean,
            "phase_std_cagr": std,
            "phase_variance_cagr": std**2,
            "mean_minus_std": mean - std,
            "median_cagr": float(np.median(returns)),
            "min_cagr": float(np.min(returns)),
            "max_cagr": float(np.max(returns)),
            "worst_drawdown": max(drawdowns),
            "mean_fees_cny": float(np.mean(fees)),
            "offset_0_cagr": returns[0],
            "all_offsets_under_30pct_drawdown": max(drawdowns) <= .30,
            "phase_cagrs": json.dumps(returns),
        })

    if missing:
        raise ValueError(f"Full comparison incomplete: {len(set(missing))} audited accounts missing")
    if len(records) != manifest["candidate_count"]:
        raise ValueError("Full comparison omitted a completed candidate")
    if len(observed_case_ids) != len(manifest["known"]) + len(manifest["tasks"]):
        raise ValueError("Observed account identities do not match prepared manifest")

    ranked = pd.DataFrame(records).sort_values(
        ["mean_minus_std", "mean_cagr", "candidate_key"],
        ascending=[False, False, True],
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    core.save_json(OUTPUT / "all_full15_phase_report_manifest.json", {
        "candidate_count": len(ranked),
        "account_count": len(observed_case_ids),
        "phase_count": 5,
        "scientific_code_sha256": manifest["scientific_code_sha256"],
        "snapshot_id": manifest["snapshot_id"],
        "source_manifest_sha256": core.digest(OUTPUT / "manifest.json"),
        "audit_script_sha256": core.digest(__file__),
        "report_sha256": core.digest(REPORT_PATH),
        "sort": "mean_cagr - population_std_across_five_prescribed_rebalance_phases",
    })
    print(ranked.head(10).to_string(index=False))
    print(f"Audited report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
