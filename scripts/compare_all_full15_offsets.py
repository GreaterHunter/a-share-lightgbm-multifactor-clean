"""Compare every completed full-15-factor model/holding case across five rebalance phases.

The frozen final-ablation directory is read-only. New audited account replays and
their task records go into a separate experiment directory.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from paper_futu import final_ablation_core as core
from src.runtime_budget import require_memory


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/final_ablation_asof_20261013"
OUTPUT = ROOT / "reports/experiments/all_full15_offsets_asof_20261013"
SOURCE_TABLES = (
    "model_ablation.csv",
    "portfolio_coarse.csv",
    "portfolio_fine.csv",
    "cadence_comparison.csv",
    "robustness_all.csv",
)
CASE_COLUMNS = (
    "prediction_path", "rebalance", "top_k", "top_tier_size",
    "top_tier_weight", "n_drop", "spec", "engine", "policy",
    "horizon", "months", "seed",
)


def selected_rows() -> pd.DataFrame:
    known_specs = core.read_json(SOURCE / "model_selection.json")["all_specs"]
    full_specs = {spec["name"] for spec in known_specs if len(spec["features"]) == 15}
    frames = []
    for name in SOURCE_TABLES:
        frame = pd.read_csv(SOURCE / name).assign(source_table=name)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.loc[
        result["period"].eq("all")
        & result["spec"].isin(full_specs)
        & result["seed"].eq(42)
        & result["friction"].eq(0)
    ].copy()
    result["offset"] = result["offset"].astype(int)
    for column in ("rebalance", "top_k", "top_tier_size", "n_drop", "horizon", "months"):
        result[column] = result[column].astype(int)
    result["top_tier_weight"] = result["top_tier_weight"].astype(float)
    if result.empty:
        raise ValueError("No completed full-15-factor account cases")
    return result


@lru_cache(maxsize=None)
def source_account_config(case_id: str) -> dict:
    """Read the exact original account config; CSV floats may lose one ULP."""
    record = core.read_json(SOURCE / "tasks" / f"{case_id}.json")
    if record["kind"] != "account" or record["status"] != "complete":
        raise ValueError(f"Original account task is not complete: {case_id}")
    if record["code_sha256"] != core.code_hash():
        raise ValueError(f"Original account used a different scientific source: {case_id}")
    return record["config"]


def replay_config(row: pd.Series, offset: int) -> dict:
    exact = source_account_config(str(row["case_id"])).copy()
    if not (
        exact["prediction_path"] == str(row["prediction_path"])
        and int(exact["rebalance"]) == int(row["rebalance"])
        and int(exact["offset"]) == int(row["offset"])
        and math.isclose(float(exact["portfolio"]["top_tier_weight"]),
                         float(row["top_tier_weight"]), abs_tol=1e-14)
    ):
        raise ValueError(f"CSV row does not match original task config: {row['case_id']}")
    exact["offset"] = int(offset)
    return exact


def prepare() -> None:
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Comparison manifest already exists; resume it instead of replacing it")
    rows = selected_rows()
    original = rows.loc[rows["offset"].eq(0)].drop_duplicates(list(CASE_COLUMNS))
    store = core.Store()
    known = {}
    for row in rows.itertuples(index=False):
        record = pd.Series(row._asdict())
        if int(record["offset"]) not in core.offsets(int(record["rebalance"])):
            continue
        config = replay_config(record, int(record["offset"]))
        expected = store.key("account", config)
        if str(record["case_id"]) != expected:
            raise ValueError(f"Source account key mismatch: {record['case_id']} != {expected}")
        if expected in known:
            previous = known[expected]
            for metric in ("annualized_return", "max_drawdown", "total_fees_cny"):
                if not math.isclose(float(record[metric]), previous[metric], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Conflicting source results for {expected}: {metric}")
        known[expected] = {
            "annualized_return": float(record["annualized_return"]),
            "max_drawdown": float(record["max_drawdown"]),
            "total_fees_cny": float(record["total_fees_cny"]),
            "source_table": str(record["source_table"]),
        }
    candidates = []
    tasks = []
    for row in original.sort_values(list(CASE_COLUMNS)).itertuples(index=False):
        record = pd.Series(row._asdict())
        template = replay_config(record, 0)
        offsets = core.offsets(int(record["rebalance"]))
        ids = [store.key("account", {**template, "offset": offset}) for offset in offsets]
        candidate = {
            "candidate_key": core.identity({name: record[name] for name in CASE_COLUMNS})[:20],
            "spec": str(record["spec"]),
            "horizon": int(record["horizon"]),
            "policy": str(record["policy"]),
            "months": int(record["months"]),
            "seed": int(record["seed"]),
            "prediction_path": str(record["prediction_path"]),
            "rebalance": int(record["rebalance"]),
            "portfolio": template["portfolio"],
            "offsets": offsets,
            "case_ids": ids,
        }
        candidates.append(candidate)
        for offset, case_id in zip(offsets, ids):
            if case_id not in known:
                tasks.append({"candidate_key": candidate["candidate_key"], "offset": offset,
                              "case_id": case_id, "prediction_path": candidate["prediction_path"],
                              "portfolio": candidate["portfolio"], "rebalance": candidate["rebalance"]})
    if len({item["candidate_key"] for item in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate identity")
    manifest = {
        "description": "Completed full-15-factor model/portfolio cases; five prescribed rebalance phases",
        "scientific_code_sha256": core.code_hash(),
        "snapshot_id": store.data,
        "source_sha256": {name: core.digest(SOURCE / name) for name in SOURCE_TABLES},
        "model_selection_sha256": core.digest(SOURCE / "model_selection.json"),
        "comparison_script_sha256": core.digest(__file__),
        "candidate_count": len(candidates),
        "known_account_count": len(known),
        "missing_account_count": len(tasks),
        "candidates": candidates,
        "tasks": tasks,
        "known": known,
    }
    core.save_json(OUTPUT / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("candidate_count", "known_account_count", "missing_account_count")}, indent=2))


def isolate_output() -> None:
    # Module globals are redirected in this process only; scientific source files stay unchanged.
    core.REPORT = OUTPUT
    core.NAME = "all_full15_offsets_asof_20261013"
    from paper_futu import final_ablation_account as account
    account.REPORT = OUTPUT
    core.init_qlib()


def worker(shard: int, shards: int, max_tasks: int | None = None) -> None:
    manifest = core.read_json(OUTPUT / "manifest.json")
    if core.code_hash() != manifest["scientific_code_sha256"]:
        raise ValueError("Frozen scientific code changed")
    if core.digest(__file__) != manifest["comparison_script_sha256"]:
        raise ValueError("Comparison script changed since preparation")
    for name, digest in manifest["source_sha256"].items():
        if core.digest(SOURCE / name) != digest:
            raise ValueError(f"Frozen source table changed: {name}")
    if not (0 <= shard < shards):
        raise ValueError("Invalid shard")
    isolate_output()
    from paper_futu.final_ablation_account import replay
    jobs = [job for index, job in enumerate(manifest["tasks"]) if index % shards == shard]
    if max_tasks is not None:
        jobs = jobs[:max_tasks]
    progress_path = OUTPUT / f"worker_{shard}_of_{shards}.json"
    started = time.time()
    done = 0
    for job in jobs:
        require_memory(2.0, "完整15因子调仓起点回测")
        progress = {"status": "running", "pid": __import__("os").getpid(),
                    "shard": shard, "shards": shards, "done_this_run": done,
                    "assigned_this_run": len(jobs), "current_case_id": job["case_id"],
                    "started": started, "updated": time.time()}
        core.save_json(progress_path, progress)
        try:
            result = replay(job["prediction_path"], job["portfolio"], job["rebalance"],
                            job["offset"], 0.0)
            if result["rows"][0]["case_id"] != job["case_id"]:
                raise ValueError("Replayed account case ID mismatch")
        except BaseException as error:
            core.save_json(progress_path, {**progress, "status": "failed", "error": repr(error),
                                           "updated": time.time()})
            raise
        done += 1
        if done % 10 == 0 or done == len(jobs):
            core.save_json(progress_path, {**progress, "status": "running", "done_this_run": done,
                                           "current_case_id": None, "updated": time.time()})
            print(f"shard {shard}/{shards}: {done}/{len(jobs)}", flush=True)
    core.save_json(progress_path, {"status": "complete", "pid": __import__("os").getpid(),
                                   "shard": shard, "shards": shards, "done_this_run": done,
                                   "assigned_this_run": len(jobs), "started": started,
                                   "updated": time.time()})


def status() -> None:
    manifest = core.read_json(OUTPUT / "manifest.json")
    complete = sum((OUTPUT / "accounts" / job["case_id"] / "metrics.csv").exists()
                   for job in manifest["tasks"])
    print(json.dumps({"candidates": manifest["candidate_count"],
                      "accounts_cached_at_source": manifest["known_account_count"],
                      "accounts_to_replay": manifest["missing_account_count"],
                      "replayed": complete,
                      "remaining": manifest["missing_account_count"] - complete}, indent=2))


def report() -> None:
    manifest = core.read_json(OUTPUT / "manifest.json")
    records = []
    missing = []
    for candidate in manifest["candidates"]:
        values = []
        drawdowns = []
        for offset, case_id in zip(candidate["offsets"], candidate["case_ids"]):
            if case_id in manifest["known"]:
                result = manifest["known"][case_id]
            else:
                path = OUTPUT / "accounts" / case_id / "metrics.csv"
                if not path.exists():
                    missing.append(case_id)
                    continue
                frame = pd.read_csv(path)
                all_rows = frame.loc[frame["period"].eq("all")]
                if len(all_rows) != 1:
                    raise ValueError(f"Expected one full-period account row: {case_id}")
                result = all_rows.iloc[0].to_dict()
                if str(result["case_id"]) != case_id or int(result["offset"]) != offset:
                    raise ValueError(f"Replayed account identity mismatch: {case_id}")
            values.append(float(result["annualized_return"]))
            drawdowns.append(-float(result["max_drawdown"]))
        if len(values) != 5:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        records.append({**{k: candidate[k] for k in ("candidate_key", "spec", "horizon",
                                                 "policy", "months", "rebalance")},
                        **candidate["portfolio"], "phase_count": 5,
                        "mean_cagr": mean, "phase_std_cagr": std,
                        "phase_variance_cagr": std**2, "mean_minus_std": mean-std,
                        "median_cagr": float(np.median(values)),
                        "worst_drawdown": max(drawdowns),
                        "offset_0_cagr": values[0],
                        "all_offsets_under_30pct_drawdown": max(drawdowns) <= 0.30})
    if missing:
        print(f"Incomplete: {len(set(missing))} account replays remain; no final ranking written")
        status()
        return
    ranked = pd.DataFrame(records).sort_values(
        ["mean_minus_std", "mean_cagr", "candidate_key"], ascending=[False, False, True])
    if len(ranked) != manifest["candidate_count"]:
        raise ValueError("Incomplete candidate comparison")
    ranked.insert(0, "rank", range(1, len(ranked)+1))
    ranked.to_csv(OUTPUT / "all_full15_phase_ranking.csv", index=False, encoding="utf-8-sig")
    print(ranked.head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "worker", "status", "report"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "worker":
        worker(args.shard, args.shards, args.max_tasks)
    elif args.stage == "status":
        status()
    else:
        report()


if __name__ == "__main__":
    main()
