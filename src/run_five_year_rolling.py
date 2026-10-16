"""Run the main strategy with a five-year rolling training window.

For each test year Y:
- train the selection model on Y-6 through Y-2 (five calendar years);
- validate on Y-1 and choose the boosting iteration;
- refit on the five training years plus the validation year;
- predict only year Y.
"""

from __future__ import annotations

import gc
import warnings

import pandas as pd
import qlib
from qlib.config import REG_CN

from src.analyze_portfolio_robustness import run_tiered_backtest
from src.compare_models import (
    enrich_daily_report,
    summarize_report,
    summarize_report_by_year,
)
from src.config import (
    PROJECT_ROOT,
    load_main_strategy_config,
    load_model_config,
    load_research_config,
)
from src.factor_library import ENHANCED_FEATURE_NAMES
from src.run_horizon_10_experiment import build_horizon_dataset
from src.run_temporal_ablation import fit_one_fold, save_prediction_frame


EXPERIMENT_NAME = "main_10d_five_year_rolling_2021_2025"
MODEL_NAME = "ranked_enhanced_10d"
LABEL_HORIZON = 10
TEST_YEARS = range(2021, 2026)


def build_folds() -> list[dict[str, str]]:
    """Create strict five-year-train, one-year-valid, one-year-test folds."""
    folds = []
    for test_year in TEST_YEARS:
        train_start_year = test_year - 6
        train_end_year = test_year - 2
        valid_year = test_year - 1
        folds.append(
            {
                "fold": f"test_{test_year}",
                "train_start": f"{train_start_year}-01-01",
                "train_end": f"{train_end_year}-12-31",
                "valid_start": f"{valid_year}-01-01",
                "valid_end": f"{valid_year}-12-31",
                "test_start": f"{test_year}-01-01",
                "test_end": f"{test_year}-12-31",
            }
        )
    return folds


def train_models() -> None:
    research = load_research_config()
    model_config = load_model_config()
    qlib.init(
        provider_uri=str(PROJECT_ROOT / research["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    dataset, raw_label, rank_label = build_horizon_dataset(
        research,
        LABEL_HORIZON,
        price_basis="legacy_open_to_close",
    )
    spec = {
        "features": ENHANCED_FEATURE_NAMES,
        "target": rank_label,
        "evaluation_label": raw_label,
        "retained": [raw_label, rank_label],
    }
    predictions = []
    records = []

    for fold in build_folds():
        print(
            f"训练 {fold['fold']}："
            f"{fold['train_start']} 至 {fold['train_end']} 训练，"
            f"{fold['valid_start']} 至 {fold['valid_end']} 验证",
            flush=True,
        )
        _, fold_test, metrics = fit_one_fold(
            dataset=dataset,
            model_name=MODEL_NAME,
            spec=spec,
            fold=fold,
            experiment_name=EXPERIMENT_NAME,
            model_parameters=model_config["parameters"],
            training_parameters=model_config["training"],
            label_horizon=LABEL_HORIZON,
            min_cross_section_size=int(research["min_cross_section_size"]),
        )
        predictions.append(fold_test)
        records.append(metrics)

    combined = pd.concat(predictions).sort_index()
    if combined.index.duplicated().any():
        raise ValueError("滚动预测包含重复的股票日期样本")
    dates = combined.index.get_level_values("datetime")
    if dates.min().year != 2021 or dates.max().year != 2025:
        raise ValueError("滚动预测没有完整覆盖2021—2025")

    prediction_path = save_prediction_frame(
        combined,
        EXPERIMENT_NAME,
        MODEL_NAME,
        "test",
    )
    output_dir = PROJECT_ROOT / "reports" / "experiments" / EXPERIMENT_NAME
    summary = pd.DataFrame(records)
    summary.to_csv(
        output_dir / "training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n训练摘要：")
    print(summary.to_string(index=False))
    print(f"\n预测保存位置：{prediction_path}")
    del dataset, combined
    gc.collect()


def backtest() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
    )
    research = load_research_config()
    strategy = load_main_strategy_config()
    config = {**research, **strategy}
    qlib.init(
        provider_uri=str(PROJECT_ROOT / research["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    output_dir = PROJECT_ROOT / "reports" / "experiments" / EXPERIMENT_NAME
    prediction_path = output_dir / f"{MODEL_NAME}_test_predictions.parquet"
    predictions = pd.read_parquet(prediction_path).sort_index()

    portfolio_name = "ranked_enhanced_10d_rebalance_10d_five_year_window"
    raw = run_tiered_backtest(
        prediction_frame=predictions,
        config=config,
        rebalance_every=10,
        top_k=int(config["top_k"]),
        top_tier_size=int(config["top_tier_size"]),
        top_tier_weight=float(config["top_tier_weight"]),
        period_end="2025-12-31",
    )
    report = enrich_daily_report(raw, portfolio_name)
    overall = summarize_report(report, portfolio_name)
    yearly = summarize_report_by_year(report, portfolio_name)
    report.to_csv(output_dir / "backtest_daily.csv", encoding="utf-8-sig")
    overall.to_csv(
        output_dir / "backtest_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    yearly.to_csv(
        output_dir / "backtest_yearly.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n五年滚动窗口总体回测：")
    print(overall.to_string(index=False))
    print("\n逐年回测：")
    print(yearly.to_string(index=False))


def main() -> None:
    train_models()
    backtest()


if __name__ == "__main__":
    main()
