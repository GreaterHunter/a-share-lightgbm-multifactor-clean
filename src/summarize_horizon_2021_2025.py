"""Audit the 26 Top10/Top5 historical pairs and save compact comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


REPORT_DIR = PROJECT_ROOT / "reports/experiments/horizon_expanded_2021_2025"
KEY = ["label_horizon", "rebalance_every"]
SCHEDULE = {
    5: (5,),
    10: (1, 5, 10, 30),
    20: (1, 5, 10, 30),
    30: (1, 5, 10, 30),
    40: (5, 10, 20, 30, 40),
    60: (1, 5, 10, 20, 30, 40, 50, 60),
}
EXPECTED = {(horizon, interval) for horizon, intervals in SCHEDULE.items() for interval in intervals}
METRICS = [
    "trading_days", "total_return", "annualized_return", "benchmark_total_return",
    "benchmark_annualized_return", "total_excess_return", "annualized_excess_return",
    "information_ratio", "max_drawdown", "average_daily_turnover", "total_fees_cny",
    "gross_net_compound_gap_pp", "signal_days",
]


def load_frame(kind: str, size: int, *, yearly: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(REPORT_DIR / f"{kind}_expanded_top{size}.csv")
    key = KEY + (["year"] if yearly else [])
    if frame.duplicated(key).any():
        raise ValueError(f"Duplicate strategy/year in {kind} Top{size}")
    if set(zip(frame[KEY[0]], frame[KEY[1]])) != EXPECTED:
        raise ValueError(f"Incomplete 26-strategy grid in {kind} Top{size}")
    if yearly and set(frame["year"]) != set(range(2021, 2026)):
        raise ValueError(f"Incomplete yearly results in {kind} Top{size}")
    if not yearly and frame["trading_days"].nunique() != 1:
        raise ValueError(f"Trading date count varies in {kind} Top{size}")
    return frame


def main() -> None:
    top10 = load_frame("comparison", 10)
    top5 = load_frame("comparison", 5)
    comparison = top10[KEY + METRICS].merge(
        top5[KEY + METRICS], on=KEY, suffixes=("_top10", "_top5"), validate="one_to_one",
    ).sort_values(KEY).reset_index(drop=True)
    if not np.allclose(
        comparison["benchmark_total_return_top10"],
        comparison["benchmark_total_return_top5"], atol=1e-12,
    ):
        raise ValueError("Top5 and Top10 have different benchmark intervals")
    if comparison["trading_days_top10"].nunique() != 1 or not (
        comparison["trading_days_top10"] == comparison["trading_days_top5"]
    ).all():
        raise ValueError("Top5 and Top10 have different trading date counts")
    comparison["top5_minus_top10_annual_return_pp"] = 100 * (
        comparison["annualized_return_top5"] - comparison["annualized_return_top10"]
    )
    comparison.to_csv(
        REPORT_DIR / "comparison_top10_vs_top5_2021_2025.csv",
        index=False, encoding="utf-8-sig",
    )

    annual10 = load_frame("yearly", 10, yearly=True)
    annual5 = load_frame("yearly", 5, yearly=True)
    yearly_columns = KEY + ["year", "total_return", "benchmark_total_return",
                            "total_excess_return", "max_drawdown", "total_fees_cny"]
    yearly = annual10[yearly_columns].merge(
        annual5[yearly_columns], on=KEY + ["year"],
        suffixes=("_top10", "_top5"), validate="one_to_one",
    ).sort_values(KEY + ["year"]).reset_index(drop=True)
    if len(yearly) != 26 * 5:
        raise ValueError("Expected five calendar-year rows for each of 26 strategies")
    if not np.allclose(
        yearly["benchmark_total_return_top10"],
        yearly["benchmark_total_return_top5"], atol=1e-12,
    ):
        raise ValueError("Yearly benchmarks differ between Top5 and Top10")
    yearly.to_csv(
        REPORT_DIR / "yearly_top10_vs_top5_2021_2025.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"Audited {len(comparison)} pairs across {comparison['trading_days_top10'].iloc[0]} trade days")
    print(f"Top5 higher annualized return: {comparison['top5_minus_top10_annual_return_pp'].gt(0).sum()}/26")
    print(f"Top5 higher fees: {(comparison['total_fees_cny_top5'] > comparison['total_fees_cny_top10']).sum()}/26")
    print(comparison[[*KEY, "annualized_return_top10", "annualized_return_top5",
                      "annualized_excess_return_top10", "annualized_excess_return_top5",
                      "max_drawdown_top10", "max_drawdown_top5"]].to_string(index=False))


if __name__ == "__main__":
    main()
