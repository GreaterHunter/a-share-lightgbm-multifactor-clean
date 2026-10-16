"""Backtest a ranked-enhanced horizon model under multiple schedules."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import qlib
from qlib.config import REG_CN

from src.analyze_portfolio_robustness import run_tiered_backtest
from src.compare_models import (
    enrich_daily_report,
    summarize_report,
    summarize_report_by_year,
)
from src.config import PROJECT_ROOT, load_research_config


DEFAULT_HORIZON = 10
DEFAULT_REBALANCE_PERIODS = (5, 10)


def run_horizon_backtest(
    horizon: int,
    rebalance_periods: tuple[int, ...],
    strategy_config: dict | None = None,
    output_dir: Path | None = None,
) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not rebalance_periods or any(period <= 0 for period in rebalance_periods):
        raise ValueError("rebalance periods must be positive")

    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
    )
    model_name = f"ranked_enhanced_{horizon}d"
    experiment_name = f"horizon_{horizon}d_rolling"
    config = load_research_config()
    if strategy_config is not None:
        config = {**config, **strategy_config}
    qlib.init(
        provider_uri=str(PROJECT_ROOT / config["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )

    prediction_path = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / experiment_name
        / f"{model_name}_test_predictions.parquet"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"{horizon}-day predictions not found: {prediction_path}"
        )
    predictions = pd.read_parquet(prediction_path).sort_index()

    daily_reports = []
    overall_reports = []
    yearly_reports = []
    for rebalance_every in rebalance_periods:
        portfolio_name = (
            f"ranked_enhanced_{horizon}d_rebalance_{rebalance_every}d"
        )
        print(f"\nBacktesting: {portfolio_name}")
        raw_report = run_tiered_backtest(
            prediction_frame=predictions,
            config=config,
            rebalance_every=rebalance_every,
            top_k=int(config.get("top_k", 50)),
            top_tier_size=int(config.get("top_tier_size", 10)),
            top_tier_weight=float(config.get("top_tier_weight", 0.40)),
        )
        report = enrich_daily_report(raw_report, portfolio_name)
        summary = summarize_report(report, portfolio_name)
        yearly = summarize_report_by_year(report, portfolio_name)
        summary["label_horizon"] = horizon
        summary["rebalance_every"] = rebalance_every
        yearly["label_horizon"] = horizon
        yearly["rebalance_every"] = rebalance_every
        daily_reports.append(report)
        overall_reports.append(summary)
        yearly_reports.append(yearly)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports" / "experiments" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.concat(daily_reports, axis=0)
    overall = pd.concat(overall_reports, ignore_index=True)
    yearly = pd.concat(yearly_reports, ignore_index=True)
    daily.to_csv(output_dir / "backtest_daily.csv", encoding="utf-8-sig")
    overall.to_csv(output_dir / "backtest_comparison.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output_dir / "backtest_yearly.csv", index=False, encoding="utf-8-sig")

    print(f"\n{horizon}-day model backtest comparison:")
    print(overall.to_string(index=False))
    print("\nYearly comparison:")
    print(yearly.to_string(index=False))


def main() -> None:
    run_horizon_backtest(
        DEFAULT_HORIZON,
        DEFAULT_REBALANCE_PERIODS,
    )


if __name__ == "__main__":
    main()
