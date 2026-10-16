"""Diagnose benchmark breadth and factor stability across market regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from src.config import PROJECT_ROOT, load_research_config
from src.factor_library import ENHANCED_FEATURE_NAMES


START_YEAR = 2021
END_YEAR = 2025
LABEL_HORIZON = 10
OUTPUT_DIR = PROJECT_ROOT / "reports" / "diagnostics" / "market_regime"


def calculate_market_summary(config: dict) -> pd.DataFrame:
    benchmark = D.features(
        [config["benchmark"]],
        ["$close"],
        start_time=f"{START_YEAR - 1}-12-01",
        end_time=f"{END_YEAR}-12-31",
        freq="day",
    )
    benchmark.columns = ["close"]
    benchmark = benchmark.reset_index().set_index("datetime").sort_index()

    constituent_returns = D.features(
        D.instruments(config["universe"]),
        ["$close / Ref($close, 1) - 1"],
        start_time=f"{START_YEAR}-01-01",
        end_time=f"{END_YEAR}-12-31",
        freq="day",
    )
    constituent_returns.columns = ["return_1d"]
    constituent_returns = constituent_returns.reset_index()

    records = []
    for year in range(START_YEAR, END_YEAR + 1):
        yearly_close = benchmark.loc[
            benchmark.index.year == year,
            "close",
        ]
        prior_close = benchmark.loc[
            benchmark.index < pd.Timestamp(f"{year}-01-01"),
            "close",
        ].iloc[-1]
        benchmark_return = yearly_close.iloc[-1] / prior_close - 1
        benchmark_daily = yearly_close.pct_change().dropna()
        benchmark_nav = yearly_close / yearly_close.iloc[0]
        benchmark_drawdown = (
            benchmark_nav / benchmark_nav.cummax() - 1
        )

        yearly_constituents = constituent_returns.loc[
            constituent_returns["datetime"].dt.year == year
        ]
        daily_cross_section = yearly_constituents.groupby("datetime")[
            "return_1d"
        ].agg(
            equal_weight_return="mean",
            cross_section_dispersion="std",
            positive_stock_ratio=lambda values: (values > 0).mean(),
        )
        equal_weight_return = (
            1 + daily_cross_section["equal_weight_return"]
        ).prod() - 1
        records.append(
            {
                "year": year,
                "benchmark_return": benchmark_return,
                "benchmark_volatility": (
                    benchmark_daily.std() * np.sqrt(252)
                ),
                "benchmark_max_drawdown": benchmark_drawdown.min(),
                "benchmark_positive_day_ratio": (
                    benchmark_daily > 0
                ).mean(),
                "equal_weight_constituent_return": equal_weight_return,
                "benchmark_minus_equal_weight": (
                    benchmark_return - equal_weight_return
                ),
                "average_cross_section_dispersion": (
                    daily_cross_section["cross_section_dispersion"].mean()
                ),
                "average_positive_stock_ratio": (
                    daily_cross_section["positive_stock_ratio"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def calculate_factor_rank_ic(config: dict) -> pd.DataFrame:
    feature_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ranked_enhanced_multifactor_dataset.parquet"
    )
    features = pd.read_parquet(
        feature_path,
        columns=ENHANCED_FEATURE_NAMES,
    )
    label = D.features(
        D.instruments(config["universe"]),
        [f"Ref($close, -{LABEL_HORIZON}) / Ref($open, -1) - 1"],
        start_time=f"{START_YEAR}-01-01",
        end_time=f"{END_YEAR}-12-31",
        freq="day",
    )
    label.columns = ["future_return"]
    panel = features.join(label, how="inner")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel["future_return_rank"] = (
        panel["future_return"]
        .groupby(level="datetime")
        .rank(pct=True)
        - 0.5
    )

    records = []
    for year in range(START_YEAR, END_YEAR + 1):
        dates = panel.index.get_level_values("datetime")
        yearly_panel = panel.loc[dates.year == year]
        signal_dates = pd.Index(
            yearly_panel.index.get_level_values("datetime").unique()
        ).sort_values()
        excluded_dates = signal_dates[-LABEL_HORIZON:]
        yearly_panel = yearly_panel.loc[
            ~yearly_panel.index.get_level_values("datetime").isin(
                excluded_dates
            )
        ]

        daily_values = []
        for _, day_panel in yearly_panel.groupby(
            level="datetime",
            sort=False,
        ):
            daily_values.append(
                day_panel[ENHANCED_FEATURE_NAMES].corrwith(
                    day_panel["future_return_rank"]
                )
            )
        yearly_rank_ic = pd.DataFrame(daily_values).mean()
        record = {"year": year}
        record.update(yearly_rank_ic.to_dict())
        record["mean_absolute_factor_rank_ic"] = yearly_rank_ic.abs().mean()
        record["positive_factor_count"] = int((yearly_rank_ic > 0).sum())
        record["negative_factor_count"] = int((yearly_rank_ic < 0).sum())
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    config = load_research_config()
    qlib.init(
        provider_uri=str(PROJECT_ROOT / config["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    market = calculate_market_summary(config)
    factor_ic = calculate_factor_rank_ic(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market.to_csv(
        OUTPUT_DIR / "market_yearly.csv",
        index=False,
        encoding="utf-8-sig",
    )
    factor_ic.to_csv(
        OUTPUT_DIR / "factor_rank_ic_yearly.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("市场状态：")
    print(market.to_string(index=False))
    print("\n年度因子Rank IC：")
    print(factor_ic.to_string(index=False))


if __name__ == "__main__":
    main()
