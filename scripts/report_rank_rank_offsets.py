"""Audit and summarize five rebalance phases for the full-15 rank/rank model.

This reads the frozen source study and the independent comparison cache. It does
not alter the original study, training artifacts, or paper-trading deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_futu import final_ablation_core as core
from scripts.compare_all_full15_offsets import OUTPUT, SOURCE, source_account_config


SPEC = "lgb_15_rank_rank"
REPORT_PATH = OUTPUT / "rank_rank_phase_ranking.csv"


def main() -> None:
    manifest = core.read_json(OUTPUT / "manifest.json")
    if core.code_hash() != manifest["scientific_code_sha256"]:
        raise ValueError("Scientific code differs from the frozen experiment")
    if core.digest(Path(__file__).with_name("compare_all_full15_offsets.py")) != manifest["comparison_script_sha256"]:
        raise ValueError("Comparison script differs from the prepared manifest")
    if any(core.digest(SOURCE / name) != expected for name, expected in manifest["source_sha256"].items()):
        raise ValueError("A frozen source table changed")

    candidates = [item for item in manifest["candidates"] if item["spec"] == SPEC]
    if len(candidates) != 770:
        raise ValueError(f"Expected 770 rank/rank configurations, found {len(candidates)}")

    store = core.Store()
    if store.data != manifest["snapshot_id"]:
        raise ValueError("Frozen data snapshot differs from the manifest")

    records = []
    missing = []
    for candidate in candidates:
        if len(candidate["offsets"]) != 5:
            raise ValueError("Each rank/rank configuration must have five prescribed phases")
        initial_config = source_account_config(candidate["case_ids"][0])
        returns = []
        drawdowns = []
        fees = []
        for offset, case_id in zip(candidate["offsets"], candidate["case_ids"]):
            config = {**initial_config, "offset": int(offset)}
            if store.key("account", config) != case_id:
                raise ValueError(f"Account identity mismatch: {case_id}")
            core.REPORT = SOURCE if case_id in manifest["known"] else OUTPUT
            result = store.cached("account", config)
            if result is None:
                missing.append(case_id)
                continue
            full_period = [row for row in result["rows"] if row["period"] == "all"]
            if len(full_period) != 1:
                raise ValueError(f"Expected one full-period result: {case_id}")
            row = full_period[0]
            if row["case_id"] != case_id or int(row["offset"]) != int(offset):
                raise ValueError(f"Account row does not match its phase: {case_id}")
            returns.append(float(row["annualized_return"]))
            drawdowns.append(-float(row["max_drawdown"]))
            fees.append(float(row["total_fees_cny"]))
        if len(returns) != 5:
            continue
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=0))
        records.append({
            "candidate_key": candidate["candidate_key"],
            "spec": SPEC,
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
        print(f"Rank/rank group incomplete: {len(set(missing))} audited account tasks remain")
        return
    if len(records) != len(candidates):
        raise ValueError("Rank/rank summary omitted a completed configuration")

    ranked = pd.DataFrame(records).sort_values(
        ["mean_minus_std", "mean_cagr", "candidate_key"],
        ascending=[False, False, True],
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    core.save_json(OUTPUT / "rank_rank_phase_report_manifest.json", {
        "spec": SPEC,
        "candidate_count": len(ranked),
        "phase_count": 5,
        "scientific_code_sha256": manifest["scientific_code_sha256"],
        "snapshot_id": manifest["snapshot_id"],
        "source_manifest_sha256": core.digest(OUTPUT / "manifest.json"),
        "report_script_sha256": core.digest(__file__),
        "report_sha256": core.digest(REPORT_PATH),
        "sort": "mean_cagr - population_std_across_five_prescribed_rebalance_phases",
    })
    print(ranked.head(10).to_string(index=False))
    print(f"Audited report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
