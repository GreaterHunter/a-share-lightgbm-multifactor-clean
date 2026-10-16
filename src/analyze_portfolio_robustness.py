"""Backtest tiered Top 50 portfolios and diagnose prediction robustness."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import qlib
from qlib.backtest.executor import SimulatorExecutor
from qlib.config import REG_CN
from qlib.contrib.evaluate import backtest_daily
from qlib.data import D

from src.compare_models import (
    build_rebalance_signal,
    enrich_daily_report,
    find_first_trade_date,
    summarize_report,
    summarize_report_by_year,
)
from src.config import PROJECT_ROOT, load_research_config
from src.evaluate_model import validate_prediction_frame
from src.factor_library import LABEL_NAME
from src.portfolio_strategy import TieredTopKWeightStrategy


MODEL_NAME = "ranked_enhanced"
REBALANCE_PERIODS = (3, 5, 10)
TOP_K = 50
TOP_TIER_SIZE = 10
TOP_TIER_WEIGHT = 0.40
PERIOD_END = "2024-12-31"


def load_rolling_predictions() -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / "rolling_annual"
        / f"{MODEL_NAME}_test_predictions.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    frame = pd.read_parquet(path)
    validate_prediction_frame(frame)
    if LABEL_NAME not in frame.columns:
        raise ValueError(f"Prediction data is missing label column: {LABEL_NAME}")
    return frame.sort_index()


def run_tiered_backtest(
    prediction_frame: pd.DataFrame,
    config: dict,
    rebalance_every: int,
    buy_cost_multiplier: float = 1.0,
    sell_cost_multiplier: float = 1.0,
    top_k: int = TOP_K,
    top_tier_size: int = TOP_TIER_SIZE,
    top_tier_weight: float = TOP_TIER_WEIGHT,
    period_end: str = PERIOD_END,
    quote_codes: list[str] | None = None,
) -> pd.DataFrame:
    signal, signal_dates = build_rebalance_signal(
        prediction_frame=prediction_frame,
        rebalance_every=rebalance_every,
    )
    first_trade_date = find_first_trade_date(
        first_signal_date=signal_dates[0],
        period_end=period_end,
    )

    strategy = TieredTopKWeightStrategy(
        signal=signal,
        top_k=top_k,
        n_drop=int(config["n_drop"]),
        top_tier_size=top_tier_size,
        top_tier_weight=top_tier_weight,
        only_tradable=True,
        risk_degree=float(config["risk_degree"]),
    )
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        verbose=False,
    )
    aligned = config.get("label_price_basis") == "open_to_open"
    # A cash-only signal-day warm-up supplies yesterday's close for order sizing.
    # Morning tradability cannot use the same day's still-unknown closing return.
    if aligned:
        threshold = float(config["limit_threshold"])
        change = "($open / Ref($close, 1) - 1)"
        limits = (f"{change} >= {threshold}", f"{change} <= -{threshold}")
    else:
        limits = float(config["limit_threshold"])  # Explicit historical protocol.

    report, _ = backtest_daily(
        start_time=signal_dates[0] if aligned else first_trade_date,
        end_time=pd.Timestamp(period_end),
        strategy=strategy,
        executor=executor,
        account=float(config["initial_cash"]),
        benchmark=str(config["benchmark"]),
        exchange_kwargs={
            **({"codes": quote_codes} if quote_codes is not None else {}),
            "limit_threshold": limits,
            "deal_price": "open",
            "open_cost": float(config["buy_cost"]) * buy_cost_multiplier,
            "close_cost": float(config["sell_cost"]) * sell_cost_multiplier,
            "min_cost": float(config["minimum_cost"]),
            "trade_unit": 100,
        },
    )
    if report.empty:
        raise ValueError("Tiered portfolio backtest returned an empty report")
    if aligned:
        warmup = report.loc[report.index < first_trade_date]
        if not warmup.turnover.eq(0).all() or not np.allclose(warmup.account, float(config["initial_cash"])):
            raise ValueError("Signal-day warm-up must stay entirely in cash")
        report = report.loc[report.index >= first_trade_date].copy()
        if report.empty:
            raise ValueError("No trading sessions after signal-day warm-up")
        quotes = D.features([str(config["benchmark"])], ["$open", "$close"],
                            start_time=str(report.index[0].date()), end_time=str(report.index[0].date()), freq="day")
        opening, close = quotes.iloc[0]["$open"], quotes.iloc[0]["$close"]
        if not np.isfinite([opening, close]).all() or opening <= 0:
            raise ValueError("Benchmark first-session open/close is invalid")
        report.loc[report.index[0], "bench"] = close / opening - 1
    return report


def calculate_daily_prediction_ic(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    def one_day(day: pd.DataFrame) -> pd.Series:
        pair = day[["prediction", LABEL_NAME]].dropna()
        if len(pair) < 30:
            return pd.Series({"ic": np.nan, "rank_ic": np.nan})
        return pd.Series(
            {
                "ic": pair["prediction"].corr(pair[LABEL_NAME], method="pearson"),
                "rank_ic": pair["prediction"].corr(pair[LABEL_NAME], method="spearman"),
            }
        )

    daily = prediction_frame.groupby(level="datetime", sort=True).apply(one_day)
    daily.index = pd.DatetimeIndex(daily.index, name="datetime")
    return daily.dropna(how="all")


def summarize_ic_by_year(daily_ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in daily_ic.groupby(daily_ic.index.year):
        rows.append(
            {
                "year": int(year),
                "days": len(frame),
                "mean_ic": frame["ic"].mean(),
                "icir_non_annualized": frame["ic"].mean() / frame["ic"].std(),
                "positive_ic_ratio": frame["ic"].gt(0).mean(),
                "mean_rank_ic": frame["rank_ic"].mean(),
                "rank_icir_non_annualized": (
                    frame["rank_ic"].mean() / frame["rank_ic"].std()
                ),
                "positive_rank_ic_ratio": frame["rank_ic"].gt(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def calculate_decile_returns(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    working = prediction_frame[["prediction", LABEL_NAME]].dropna().copy()
    percentile = working.groupby(level="datetime")["prediction"].rank(
        method="first",
        pct=True,
    )
    working["decile"] = np.ceil(percentile * 10).clip(1, 10).astype(int)

    daily_decile = (
        working.groupby(["datetime", "decile"])[LABEL_NAME]
        .mean()
        .rename("mean_future_5d_return")
        .reset_index()
    )
    summary = (
        daily_decile.groupby("decile")["mean_future_5d_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_future_5d_return"})
    )
    summary["mean_return_bps"] = summary["mean_future_5d_return"] * 10_000
    return summary


def main() -> None:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    config = load_research_config()
    qlib.init(
        provider_uri=str(PROJECT_ROOT / config["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    predictions = load_rolling_predictions()

    daily_reports = []
    overall_rows = []
    yearly_rows = []

    for rebalance_every in REBALANCE_PERIODS:
        portfolio_name = f"tiered_top50_{rebalance_every}d"
        print(f"\nBacktesting {portfolio_name}")
        raw = run_tiered_backtest(
            prediction_frame=predictions,
            config=config,
            rebalance_every=rebalance_every,
        )
        report = enrich_daily_report(raw, portfolio_name)
        overall = summarize_report(report, portfolio_name)
        yearly = summarize_report_by_year(report, portfolio_name)
        overall["rebalance_every"] = rebalance_every
        yearly["rebalance_every"] = rebalance_every
        daily_reports.append(report)
        overall_rows.append(overall)
        yearly_rows.append(yearly)

    # Cost stress is performed on the pre-declared 5-day schedule, not selected
    # after observing which rebalance period has the highest return.
    stress_name = "tiered_top50_5d_double_cost"
    print(f"\nBacktesting {stress_name}")
    stress_raw = run_tiered_backtest(
        prediction_frame=predictions,
        config=config,
        rebalance_every=5,
        buy_cost_multiplier=2.0,
        sell_cost_multiplier=2.0,
    )
    stress_report = enrich_daily_report(stress_raw, stress_name)
    stress_overall = summarize_report(stress_report, stress_name)
    stress_yearly = summarize_report_by_year(stress_report, stress_name)
    stress_overall["rebalance_every"] = 5
    stress_yearly["rebalance_every"] = 5
    daily_reports.append(stress_report)
    overall_rows.append(stress_overall)
    yearly_rows.append(stress_yearly)

    daily_ic = calculate_daily_prediction_ic(predictions)
    yearly_ic = summarize_ic_by_year(daily_ic)
    decile_returns = calculate_decile_returns(predictions)

    output_dir = PROJECT_ROOT / "reports" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = pd.concat(overall_rows, ignore_index=True)
    yearly = pd.concat(yearly_rows, ignore_index=True)
    daily = pd.concat(daily_reports, axis=0)

    overall.to_csv(output_dir / "tiered_portfolio_comparison.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output_dir / "tiered_portfolio_yearly.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "tiered_portfolio_daily.csv", encoding="utf-8-sig")
    daily_ic.to_csv(output_dir / "prediction_daily_ic.csv", encoding="utf-8-sig")
    yearly_ic.to_csv(output_dir / "prediction_yearly_ic.csv", index=False, encoding="utf-8-sig")
    decile_returns.to_csv(output_dir / "prediction_decile_returns.csv", index=False, encoding="utf-8-sig")

    print("\nPortfolio robustness summary:")
    print(overall.to_string(index=False))
    print("\nYearly prediction IC:")
    print(yearly_ic.to_string(index=False))
    print("\nPrediction decile returns:")
    print(decile_returns.to_string(index=False))


if __name__ == "__main__":
    main()
