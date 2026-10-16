"""Compare LightGBM feature importance across annual rolling models."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src.config import PROJECT_ROOT


MODEL_NAME = "ranked_enhanced"
TEST_YEARS = (2022, 2023, 2024)


def load_one_year_importance(year: int) -> pd.DataFrame:
    model_path = (
        PROJECT_ROOT
        / "models"
        / "experiments"
        / "rolling_annual"
        / f"{MODEL_NAME}_test_{year}.txt"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    booster = lgb.Booster(model_file=str(model_path))
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    frame = pd.DataFrame(
        {
            "year": year,
            "factor": booster.feature_name(),
            "gain": gain,
            "split": split,
        }
    )
    total_gain = frame["gain"].sum()
    if total_gain <= 0:
        raise ValueError(f"Model {year} has no positive feature gain")

    frame["gain_ratio"] = frame["gain"] / total_gain
    frame["importance_rank"] = frame["gain_ratio"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    return frame.sort_values("importance_rank")


def main() -> None:
    detail = pd.concat(
        [load_one_year_importance(year) for year in TEST_YEARS],
        ignore_index=True,
    )

    summary = (
        detail.groupby("factor")
        .agg(
            mean_gain_ratio=("gain_ratio", "mean"),
            std_gain_ratio=("gain_ratio", "std"),
            mean_rank=("importance_rank", "mean"),
            best_rank=("importance_rank", "min"),
            worst_rank=("importance_rank", "max"),
            positive_years=("gain", lambda values: int((values > 0).sum())),
        )
        .sort_values(["mean_rank", "mean_gain_ratio"], ascending=[True, False])
        .reset_index()
    )

    rank_matrix = detail.pivot(
        index="factor",
        columns="year",
        values="importance_rank",
    )
    rank_correlation = rank_matrix.corr(method="spearman")

    output_dir = PROJECT_ROOT / "reports" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        output_dir / "feature_importance_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "feature_importance_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rank_correlation.to_csv(
        output_dir / "feature_rank_correlation.csv",
        encoding="utf-8-sig",
    )

    print("Feature importance stability:")
    print(summary.to_string(index=False))
    print("\nSpearman correlation between annual importance ranks:")
    print(rank_correlation.to_string())


if __name__ == "__main__":
    main()
