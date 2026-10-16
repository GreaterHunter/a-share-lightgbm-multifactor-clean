"""One-time locked 2025 validation for the three pre-specified candidates."""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import pandas as pd
import qlib
from qlib.config import REG_CN

from src.analyze_portfolio_robustness import run_tiered_backtest
from src.compare_models import enrich_daily_report, summarize_report
from src.config import (
    PROJECT_ROOT,
    load_main_strategy_config,
    load_model_config,
    load_research_config,
)
from src.factor_library import ENHANCED_FEATURE_NAMES
from src.evaluate_model import calculate_daily_model_ic
from src.run_horizon_10_experiment import build_horizon_dataset
from src.run_temporal_ablation import fit_one_fold, save_prediction_frame


EXPERIMENT_NAME = "locked_2025_validation"
TEST_END = "2025-12-31"
LOCKED_FOLD = {
    "fold": "test_2025",
    "train_start": "2013-01-01",
    "train_end": "2023-12-31",
    "valid_start": "2024-01-01",
    "valid_end": "2024-12-31",
    "test_start": "2025-01-01",
    "test_end": TEST_END,
}
CANDIDATES = {
    10: (10,),
    15: (5, 10),
}


def prediction_path(horizon: int) -> Path:
    model_name = f"ranked_enhanced_{horizon}d"
    return (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / EXPERIMENT_NAME
        / f"{model_name}_test_2025_predictions.parquet"
    )


def train_locked_models() -> None:
    research = load_research_config()
    model_config = load_model_config()
    qlib.init(
        provider_uri=str(PROJECT_ROOT / research["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    records = []
    for horizon in CANDIDATES:
        dataset, raw_label, rank_label = build_horizon_dataset(research, horizon, price_basis="legacy_open_to_close")
        model_name = f"ranked_enhanced_{horizon}d"
        spec = {
            "features": ENHANCED_FEATURE_NAMES,
            "target": rank_label,
            "evaluation_label": raw_label,
            "retained": [raw_label, rank_label],
        }
        print(f"训练锁定模型：{model_name}", flush=True)
        _, test_predictions, metrics = fit_one_fold(
            dataset=dataset,
            model_name=model_name,
            spec=spec,
            fold=LOCKED_FOLD,
            experiment_name=EXPERIMENT_NAME,
            model_parameters=model_config["parameters"],
            training_parameters=model_config["training"],
            label_horizon=horizon,
            min_cross_section_size=int(research["min_cross_section_size"]),
        )
        save_prediction_frame(
            test_predictions,
            EXPERIMENT_NAME,
            model_name,
            "test_2025",
        )
        records.append(metrics)
        del dataset, test_predictions
        gc.collect()

    output_dir = PROJECT_ROOT / "reports" / "experiments" / EXPERIMENT_NAME
    summary = pd.DataFrame(records)
    summary.to_csv(
        output_dir / "training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n2025验证模型训练摘要：")
    print(summary.to_string(index=False))


def backtest_locked_candidates() -> None:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    research = load_research_config()
    strategy = load_main_strategy_config()
    config = {**research, **strategy}
    qlib.init(
        provider_uri=str(PROJECT_ROOT / research["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    daily_reports = []
    summaries = []
    ic_summaries = []
    for horizon, rebalance_periods in CANDIDATES.items():
        path = prediction_path(horizon)
        if not path.exists():
            raise FileNotFoundError(f"缺少2025预测：{path}")
        predictions = pd.read_parquet(path).sort_index()
        dates = predictions.index.get_level_values("datetime")
        if dates.min().year != 2025 or dates.max().year != 2025:
            raise ValueError(f"{horizon}日模型预测不严格属于2025年")

        daily_ic = calculate_daily_model_ic(
            prediction_frame=predictions,
            min_cross_section_size=int(research["min_cross_section_size"]),
            label_name=f"future_return_{horizon}d",
        )
        ic_summaries.append(
            {
                "label_horizon": horizon,
                "valid_days": len(daily_ic),
                "mean_ic": daily_ic["ic"].mean(),
                "mean_rank_ic": daily_ic["rank_ic"].mean(),
                "positive_rank_ic_ratio": (daily_ic["rank_ic"] > 0).mean(),
            }
        )

        for rebalance_every in rebalance_periods:
            name = f"ranked_enhanced_{horizon}d_rebalance_{rebalance_every}d"
            print(f"回测锁定候选：{name}", flush=True)
            raw = run_tiered_backtest(
                prediction_frame=predictions,
                config=config,
                rebalance_every=rebalance_every,
                top_k=int(config["top_k"]),
                top_tier_size=int(config["top_tier_size"]),
                top_tier_weight=float(config["top_tier_weight"]),
                period_end=TEST_END,
            )
            report = enrich_daily_report(raw, name)
            summary = summarize_report(report, name)
            summary["label_horizon"] = horizon
            summary["rebalance_every"] = rebalance_every
            daily_reports.append(report)
            summaries.append(summary)

    output_dir = PROJECT_ROOT / "reports" / "experiments" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.concat(daily_reports, axis=0)
    comparison = pd.concat(summaries, ignore_index=True)
    ic_summary = pd.DataFrame(ic_summaries)
    daily.to_csv(output_dir / "backtest_daily.csv", encoding="utf-8-sig")
    comparison.to_csv(
        output_dir / "backtest_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ic_summary.to_csv(
        output_dir / "test_ic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n锁定2025验证结果：")
    print(comparison.to_string(index=False))
    print("\n2025预测IC：")
    print(ic_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "train", "backtest"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"all", "train"}:
        train_locked_models()
    if args.stage in {"all", "backtest"}:
        backtest_locked_candidates()


if __name__ == "__main__":
    main()
