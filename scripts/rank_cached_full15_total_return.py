"""Read-only attribution of the audited grid; never train or replay accounts.

Source reports, frozen protocol and deployment files are not modified. CSV/JSON
outputs are analytical intermediates in a separate analysis directory.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paper_futu.benchmark_attribution import capm, official_frame

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "reports/experiments/all_full15_offsets_asof_20261013"
SOURCE = ROOT / "reports/experiments/final_ablation_asof_20261013"
OUT = ROOT / "reports/analysis/full15_total_return_asof_20261013"
RAW = ROOT / "data/external/csi300_total_return_asof_20261013"
RF_DAILY = np.expm1(np.log1p(0.01) / 252)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def excluded_single_head(portfolio):
    return (int(portfolio["top_tier_size"]) == 1
            and float(portfolio["top_tier_weight"]) > 1 / int(portfolio["top_k"]) + 1e-12)


def compute_account(frame, price, total):
    dates = frame.index
    assert isinstance(dates, pd.DatetimeIndex) and dates.is_unique and dates.is_monotonic_increasing
    assert len(dates) == 1154 and str(dates[0].date()) == "2022-01-04"
    assert str(dates[-1].date()) == "2026-10-13"
    benchmarks = []
    for levels in (price, total):
        assert dates.isin(levels.index).all()
        values = levels.close.pct_change(fill_method=None).reindex(dates).to_numpy()
        values[0] = levels.loc[dates[0], "close"] / levels.loc[dates[0], "open"] - 1
        assert np.isfinite(values).all()
        benchmarks.append(values)
    np.testing.assert_allclose(frame.bench, benchmarks[0], atol=5e-6, rtol=0)
    net = frame.net_return.to_numpy()
    assert np.isfinite(net).all() and (net > -1).all()
    nav = np.cumprod(1 + net)
    np.testing.assert_allclose(frame.portfolio_nav, nav, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(frame.account / 1_000_000, nav, rtol=1e-10, atol=1e-10)
    peaks = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    regression = capm(net, benchmarks[1] - RF_DAILY, lags=20)
    return dict(cagr=float(nav[-1] ** (252 / len(net)) - 1),
                drawdown=float(-(nav / peaks - 1).min()),
                fees_cny=float(frame.fees_cash.sum()),
                alpha=regression["alpha_annual_arithmetic"],
                beta=regression["market_beta"],
                alpha_ci_low=regression["alpha_ci_low"],
                alpha_ci_high=regression["alpha_ci_high"],
                alpha_nominal_p=regression["alpha_nominal_p"],
                benchmark_cagr=float(np.prod(1 + benchmarks[1]) ** (252 / len(net)) - 1))


def main():
    started = time.time()
    manifest_path = GRID / "manifest.json"
    ranking_path = GRID / "all_full15_phase_ranking.csv"
    audit = load(GRID / "all_full15_phase_report_manifest.json")
    assert sha(manifest_path) == audit["source_manifest_sha256"]
    assert sha(ranking_path) == audit["report_sha256"]
    manifest = load(manifest_path)
    previous = pd.read_csv(ranking_path).set_index("candidate_key")
    assert len(manifest["candidates"]) == len(previous) == 1551
    source_hashes = {str(p): sha(p) for p in (manifest_path, ranking_path,
                     GRID / "all_full15_phase_report_manifest.json",
                     ROOT / "reports/paper_futu/s6_forward_v1/protocol.json",
                     ROOT / "reports/paper_futu/s6_forward_v1/seal.json")}
    official_sources = load(RAW / "sources.json")
    for item in official_sources["sources"]:
        assert sha(RAW / item["file"]) == item["sha256"]
        source_hashes[str(RAW / item["file"])] = item["sha256"]
    price = official_frame(load(RAW / "000300.json"), "000300")
    total = official_frame(load(RAW / "H00300.json"), "H00300")
    rows, phase_rows, daily_hashes, checked_predictions = [], [], {}, set()
    columns = ["net_return", "bench", "portfolio_nav", "fees_cash", "account"]
    for i, candidate in enumerate(manifest["candidates"]):
        case_rows = []
        assert len(candidate["offsets"]) == len(candidate["case_ids"]) == 5
        for offset, case_id in zip(candidate["offsets"], candidate["case_ids"]):
            task_root = SOURCE if case_id in manifest["known"] else GRID
            record = load(task_root / "tasks" / f"{case_id}.json")
            assert record["status"] == "complete" and record["kind"] == "account"
            assert record["task_id"] == case_id
            assert record["code_sha256"] == manifest["scientific_code_sha256"]
            assert record["snapshot_id"] == manifest["snapshot_id"]
            cfg = record["config"]
            assert cfg["portfolio"] == candidate["portfolio"]
            assert cfg["prediction_path"] == candidate["prediction_path"]
            assert cfg["offset"] == offset and cfg["rebalance"] == candidate["rebalance"]
            assert cfg["initial_cash"] == 1_000_000 and cfg["risk_degree"] == 0.95
            assert cfg["friction"] == 0 and cfg["annualization"] == 252
            assert cfg["execution"] == "T+1 open"
            prediction = Path(cfg["prediction_path"])
            if prediction not in checked_predictions:
                assert sha(prediction) == cfg["prediction_sha256"]
                source_hashes[str(prediction)] = cfg["prediction_sha256"]
                checked_predictions.add(prediction)
            daily_path = Path(record["result"]["directory"]) / "daily.parquet"
            digest = sha(daily_path)
            assert record["artifacts"][str(daily_path)] == digest
            daily_hashes[str(daily_path)] = digest
            frame = pd.read_parquet(daily_path, columns=columns)
            values = compute_account(frame, price, total)
            original = next(r for r in record["result"]["rows"] if r["period"] == "all")
            np.testing.assert_allclose(values["cagr"], original["annualized_return"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(values["drawdown"], -original["max_drawdown"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(values["fees_cny"], original["total_fees_cny"], atol=1e-5, rtol=0)
            phase = dict(candidate_key=candidate["candidate_key"], case_id=case_id,
                         offset=offset, **values)
            case_rows.append(phase)
            phase_rows.append(phase)
        phases = pd.DataFrame(case_rows)
        p = candidate["portfolio"]
        values = dict(candidate_key=candidate["candidate_key"], spec=candidate["spec"],
                      horizon=candidate["horizon"], rebalance=candidate["rebalance"],
                      policy=candidate["policy"], months=candidate["months"], **p,
                      phase_count=5, mean_cagr=float(phases.cagr.mean()),
                      std_cagr=float(phases.cagr.std(ddof=0)),
                      mean_alpha=float(phases.alpha.mean()),
                      std_alpha=float(phases.alpha.std(ddof=0)),
                      mean_beta=float(phases.beta.mean()),
                      worst_drawdown=float(phases.drawdown.max()),
                      mean_fees_cny=float(phases.fees_cny.mean()),
                      all_offsets_drawdown_le_30=bool(phases.drawdown.le(.30).all()),
                      excluded_single_head=excluded_single_head(p))
        np.testing.assert_allclose(values["mean_cagr"], previous.loc[candidate["candidate_key"], "mean_cagr"], atol=1e-10, rtol=0)
        rows.append(values)
        if (i + 1) % 100 == 0:
            print(json.dumps(dict(configs=i + 1, accounts=len(phase_rows),
                                  elapsed_seconds=round(time.time() - started, 1))), flush=True)
    result = pd.DataFrame(rows)
    assert len(phase_rows) == len(daily_hashes) == 7755
    eligible = result.loc[~result.excluded_single_head].copy()
    by_return = eligible.sort_values(["mean_cagr", "candidate_key"], ascending=[False, True]).head(30)
    by_alpha = eligible.sort_values(["mean_alpha", "candidate_key"], ascending=[False, True]).head(30)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [("all_configurations", result), ("per_phase", pd.DataFrame(phase_rows)),
                        ("top30_cagr", by_return), ("top30_alpha", by_alpha)]:
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    verification = dict(candidates=1551, accounts=7755, eligible=len(eligible),
                        excluded_single_head=int(result.excluded_single_head.sum()),
                        prediction_streams=len(checked_predictions),
                        start="2022-01-04", end="2026-10-13", days=1154,
                        market="H00300", annual_rf=.01, alpha_annualization=252,
                        alpha_method="mean of per-phase daily CAPM intercept times 252; HAC20 intervals per phase only",
                        rank_rule="Separate descending mean CAGR and mean market alpha; no combined score",
                        concentration_filter="Exclude head1 tilt above equal weight; keep Top5 equal and head2 tilts",
                        baseline_rule="First day official close/open, then close/previous close",
                        freeze_unchanged=True, sources=source_hashes, daily_hashes=daily_hashes,
                        script_sha256=sha(__file__), elapsed_seconds=time.time()-started)
    for p, expected in source_hashes.items():
        assert sha(p) == expected
    verification["outputs"] = {p.name: sha(p) for p in OUT.glob("*.csv")}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in verification.items() if k not in ("sources", "daily_hashes")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
