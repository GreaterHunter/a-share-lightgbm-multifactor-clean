"""Evaluate a freshly refitted model with Top10 40/60 tiered portfolio weights."""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN

from paper_futu.backtest_recent_window import DATE, PROVIDER, REPORT
from paper_futu.cn_common import MODEL_DIR, ROOT, STATE_DIR
from src.analyze_portfolio_robustness import run_tiered_backtest
from src.compare_models import calculate_max_drawdown, enrich_daily_report, summarize_report_by_year
from src.config import load_main_strategy_config, load_research_config
from src.factor_library import ENHANCED_FEATURE_NAMES

VARIANT_MODEL = ROOT / "models/paper_futu/cn_2026_top10_tiered"
OUTPUT = REPORT / "top10_tiered"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=VARIANT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    model_dir = (ROOT / args.model_dir).resolve()
    output = (ROOT / args.output_dir).resolve()
    if not model_dir.is_relative_to((ROOT / "models").resolve()):
        raise ValueError("Model must be inside the project's models directory")
    if not output.is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("Report output must be inside the project's reports directory")
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    portfolio = metadata["portfolio"]
    drop = int(portfolio["n_drop"])
    if portfolio["top_k"] != 10 or portfolio["top_tier_size"] != 2 or portfolio["top_tier_weight"] != 0.4:
        raise ValueError("This experiment requires the Top10 2/8 tiered allocation")
    variant = f"top10_tiered_2_8_drop{drop}"
    original_metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    new_digest = hashlib.sha256((model_dir / "model.txt").read_bytes()).hexdigest()
    if new_digest != metadata["model_sha256"] or new_digest != original_metadata["model_sha256"]:
        raise ValueError("Refit model differs from the frozen baseline; predictions must be regenerated")
    new_model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    features = pd.read_parquet(STATE_DIR / DATE / "features_ranked.parquet")
    reference = pd.read_csv(STATE_DIR / DATE / "predictions.csv", parse_dates=["datetime"]).set_index(["instrument", "datetime"])
    values = pd.Series(new_model.predict(features[ENHANCED_FEATURE_NAMES], num_threads=4), index=features.index)
    if not np.allclose(values, reference.reindex(features.index).prediction, atol=1e-12, rtol=0):
        raise ValueError("Fresh model latest scores differ from baseline")
    # Byte-identical fitted trees yield identical predictions on the same features.
    predictions = pd.read_parquet(REPORT / "predictions.parquet")
    codes = sorted(predictions.index.get_level_values("instrument").unique())
    options = {**load_research_config(), **load_main_strategy_config(), **metadata["portfolio"]}
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    raw = run_tiered_backtest(predictions, options, rebalance_every=10, top_k=10,
        top_tier_size=2, top_tier_weight=0.40, period_end=DATE, quote_codes=codes)
    report = enrich_daily_report(raw, variant)
    report.to_csv(output / "full_daily.csv", encoding="utf-8-sig")
    yearly = summarize_report_by_year(report, variant)
    yearly.to_csv(output / "yearly.csv", index=False, encoding="utf-8-sig")
    baseline_report = pd.read_csv(REPORT / "full_daily_replay.csv", index_col=0, parse_dates=True)
    if not report.index.equals(baseline_report.index) or not np.allclose(report.bench, baseline_report.bench, atol=1e-12, rtol=0):
        raise ValueError("Top10 and Top50 backtests use different trading dates or benchmarks")
    baseline_yearly = summarize_report_by_year(baseline_report, "original_tiered_top50")
    yearly_results = [baseline_yearly]
    if output != OUTPUT.resolve() and (OUTPUT / "yearly.csv").exists():
        yearly_results.append(pd.read_csv(OUTPUT / "yearly.csv"))
    pd.concat([*yearly_results, yearly], ignore_index=True).to_csv(
        output / "yearly_comparison.csv", index=False, encoding="utf-8-sig"
    )
    window = report.tail(30)
    if len(window) != 30 or str(window.index.min().date()) != "2026-08-24" or str(window.index.max().date()) != DATE:
        raise ValueError("Comparison window differs from previous experiments")
    wealth = float((1 + window.net_return).prod())
    benchmark = float((1 + window.bench).prod())
    before = report.loc[report.index < window.index.min()].iloc[-1]
    if not np.isclose(window.account.iloc[-1] / before.account, wealth, atol=1e-10):
        raise ValueError("Compounded return does not match account balance change")
    summary = {"variant": variant, "start": str(window.index.min().date()),
        "end": DATE, "trading_days": 30, "strategy_net_return": wealth - 1,
        "benchmark_return": benchmark - 1, "geometric_excess_return": wealth / benchmark - 1,
        "arithmetic_excess_percentage_points": (wealth - benchmark) * 100,
        "max_drawdown": calculate_max_drawdown(window.net_return),
        "cost_rate_sum": float(window.cost.sum()), "positive_days": int(window.net_return.gt(0).sum()),
        "top_k": 10, "top_tier_size": 2, "top_tier_weight": 0.4, "n_drop": drop,
        "each_top_two_total_asset_weight": 0.19, "each_other_eight_total_asset_weight": 0.07125,
        "portfolio": metadata["portfolio"], "retrained_model_path": str(model_dir / "model.txt"),
        "retrained_model_hash_identical_to_baseline": True,
        "prediction_reuse_reason": "fresh deterministic refit produced byte-identical trees; latest scores independently verified",
        "historical_models": "2022-2025 annual models and scores unchanged; fresh refit is the 2026 model only",
        "continuous_account_since": str(report.index.min().date()),
        "not_live_paper_results": True,
        "not_reset_at_30_day_window_start": True,
        "model_train_end": metadata["train_end"], "selected_trees": metadata["chosen_iteration"]}
    window.to_csv(output / "last_30_daily.csv", encoding="utf-8-sig")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    baseline = pd.read_csv(REPORT / "top10_comparison/comparison.csv")
    if output != OUTPUT.resolve() and (OUTPUT / "summary.json").exists():
        drop5 = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        baseline = pd.concat([baseline, pd.DataFrame([drop5])], ignore_index=True)
    pd.concat([baseline, pd.DataFrame([summary])], ignore_index=True).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, indent=2), flush=True)
    print(yearly.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
