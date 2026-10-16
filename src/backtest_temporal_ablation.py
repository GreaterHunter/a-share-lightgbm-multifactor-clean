"""Backtest static and annual rolling temporal ablation predictions."""

import argparse
import warnings

import pandas as pd
import qlib
from qlib.config import REG_CN

from src.compare_models import (
    enrich_daily_report,
    run_one_model_backtest,
    summarize_report,
    summarize_report_by_year,
    validate_comparable_frames,
)
from src.config import PROJECT_ROOT, load_research_config
from src.evaluate_model import validate_prediction_frame
from src.run_temporal_ablation import MODEL_SPECS


EXPERIMENTS = {
    "static_valid": {
        "directory": "static_2y",
        "split": "valid",
        "period_end": "2022-12-31",
    },
    "static_test": {
        "directory": "static_2y",
        "split": "test",
        "period_end": "2024-12-31",
    },
    "rolling_test": {
        "directory": "rolling_annual",
        "split": "test",
        "period_end": "2024-12-31",
    },
}


def load_experiment_predictions(
    experiment_directory: str,
    split_name: str,
    model_name: str,
) -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / experiment_directory
        / f"{model_name}_{split_name}_predictions.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"找不到实验预测：{path}")

    frame = pd.read_parquet(path)
    validate_prediction_frame(frame)
    return frame


def run_experiment(experiment_name: str) -> None:
    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
    )

    experiment = EXPERIMENTS[experiment_name]
    config = load_research_config()
    provider_uri = PROJECT_ROOT / config["provider_uri"]

    qlib.init(
        provider_uri=str(provider_uri),
        region=REG_CN,
        kernels=1,
    )

    model_names = list(MODEL_SPECS)
    prediction_frames = {
        model_name: load_experiment_predictions(
            experiment_directory=experiment["directory"],
            split_name=experiment["split"],
            model_name=model_name,
        )
        for model_name in model_names
    }
    validate_comparable_frames(
        prediction_frames=prediction_frames,
        model_names=model_names,
    )

    daily_reports = []
    summaries = []
    yearly_summaries = []

    for model_name in model_names:
        raw_report = run_one_model_backtest(
            model_name=model_name,
            prediction_frame=prediction_frames[model_name],
            config=config,
            period_end=experiment["period_end"],
        )
        report = enrich_daily_report(raw_report, model_name)
        summary = summarize_report(report, model_name)
        yearly = summarize_report_by_year(report, model_name)

        daily_reports.append(report)
        summaries.append(summary)
        yearly_summaries.append(yearly)

    daily_comparison = pd.concat(daily_reports, axis=0)
    model_comparison = pd.concat(summaries, ignore_index=True)
    yearly_comparison = pd.concat(yearly_summaries, ignore_index=True)

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / experiment["directory"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_path = output_dir / f"{experiment_name}_daily_backtest.csv"
    summary_path = output_dir / f"{experiment_name}_comparison.csv"
    yearly_path = output_dir / f"{experiment_name}_yearly.csv"

    daily_comparison.to_csv(daily_path, encoding="utf-8-sig")
    model_comparison.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    yearly_comparison.to_csv(
        yearly_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n实验完成：{experiment_name}")
    print("\n总体结果：")
    print(model_comparison.to_string(index=False))
    print("\n逐年结果：")
    print(yearly_comparison.to_string(index=False))
    print("\n保存位置：")
    print(summary_path)
    print(yearly_path)
    print(daily_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS),
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_experiment(arguments.experiment)
