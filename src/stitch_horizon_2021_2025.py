"""Join new 2021 predictions with unchanged 2022-2025 research artifacts.

The joined parquet is an input to the continuous 2021-2025 account backtest.
The source prediction files are read-only; no model or Futu setting is changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


DESTINATION = PROJECT_ROOT / "reports/experiments/horizon_expanded_2021_2025"
SOURCES = {
    10: PROJECT_ROOT / "reports/experiments/horizon_grid_top10_2022_2025",
    20: PROJECT_ROOT / "reports/experiments/horizon_grid_top10_2022_2025",
    30: PROJECT_ROOT / "reports/experiments/horizon_grid_top10_2022_2025",
    60: PROJECT_ROOT / "reports/experiments/horizon_60_top10_2022_2025",
}
ARTIFACT_SUFFIXES = ("training.csv", "validation_folds.csv", "tree_selection.csv")


def check_prediction(frame: pd.DataFrame, expected_years: set[int], path: Path) -> None:
    if frame.index.names != ["instrument", "datetime"]:
        raise ValueError(f"Unexpected index in {path}: {frame.index.names}")
    if frame.empty or frame.index.has_duplicates or "prediction" not in frame.columns:
        raise ValueError(f"Missing or duplicate predictions in {path}")
    years = set(frame.index.get_level_values("datetime").year)
    if years != expected_years:
        raise ValueError(f"Unexpected years {sorted(years)} in {path}")
    if not np.isfinite(frame["prediction"].to_numpy()).all():
        raise ValueError(f"Non-finite predictions in {path}")
    counts = frame.groupby(level="datetime").size()
    if counts.min() < 100 or len(counts[counts.index.year == min(expected_years)]) < 200:
        raise ValueError(f"Incomplete daily predictions in {path}")


def main() -> None:
    for horizon, source_dir in SOURCES.items():
        stem = f"horizon_{horizon}d"
        destination = DESTINATION / f"{stem}_predictions.parquet"
        source = source_dir / destination.name
        current = pd.read_parquet(destination)
        historical = pd.read_parquet(source)
        check_prediction(current, {2021}, destination)
        check_prediction(historical, {2022, 2023, 2024, 2025}, source)
        if set(current.columns) != set(historical.columns):
            raise ValueError(f"Column mismatch for H{horizon}")
        joined = pd.concat([current, historical[current.columns]]).sort_index()
        check_prediction(joined, set(range(2021, 2026)), destination)
        joined.to_parquet(destination)

        for suffix in ARTIFACT_SUFFIXES:
            target_csv = DESTINATION / f"{stem}_{suffix}"
            old_csv = source_dir / target_csv.name
            new_rows = pd.read_csv(target_csv)
            old_rows = pd.read_csv(old_csv)
            merged = pd.concat([new_rows, old_rows], ignore_index=True)
            if "test_year" not in merged.columns or set(merged["test_year"]) != set(range(2021, 2026)):
                raise ValueError(f"Missing training provenance in {target_csv}")
            merged.to_csv(target_csv, index=False)
        print(f"H{horizon}: {len(joined):,} predictions, 2021-2025 -> {destination}")


if __name__ == "__main__":
    main()
