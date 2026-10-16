"""Audit and join the 2026 Top10 and Top5 horizon-grid experiments."""

from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT


OLD_DIR = PROJECT_ROOT / "reports/experiments/horizon_grid_top10_2026_jan_jul"
NEW_DIR = PROJECT_ROOT / "reports/experiments/horizon_expanded_2026_jan_jul"
KEY = ["label_horizon", "rebalance_every"]


def read_comparison(path: str, *, old: bool = False) -> pd.DataFrame:
    root = OLD_DIR if old else NEW_DIR
    frame = pd.read_csv(root / path)
    if frame.duplicated(KEY).any():
        raise ValueError(f"Duplicate strategy configuration in {root / path}")
    return frame


def main() -> None:
    top10 = pd.concat(
        [read_comparison("comparison.csv", old=True),
         read_comparison("comparison_added_top10.csv")],
        ignore_index=True,
    )
    top10_stress = pd.concat(
        [read_comparison("comparison_extra_10bp.csv", old=True),
         read_comparison("comparison_added_top10_extra_10bp.csv")],
        ignore_index=True,
    )
    top5 = read_comparison("comparison_expanded_top5.csv")
    top5_stress = read_comparison("comparison_expanded_top5_extra_10bp.csv")

    expected = 26
    for name, frame in [("Top10", top10), ("Top10 stress", top10_stress),
                        ("Top5", top5), ("Top5 stress", top5_stress)]:
        if len(frame) != expected or frame.duplicated(KEY).any():
            raise ValueError(f"{name} must contain {expected} unique configurations")
        if set(frame["trading_days"]) != {138}:
            raise ValueError(f"{name} does not use the same 138-day interval")

    columns = KEY + ["total_return", "total_excess_return", "max_drawdown",
                     "average_daily_turnover", "total_fees_cny", "signal_days",
                     "benchmark_total_return"]
    combined = top10[columns].merge(
        top5[columns], on=KEY, suffixes=("_top10", "_top5"), validate="one_to_one",
    )
    for name, frame in [("top10_stress", top10_stress), ("top5_stress", top5_stress)]:
        combined = combined.merge(
            frame[KEY + ["total_return"]].rename(
                columns={"total_return": f"total_return_{name}"}
            ),
            on=KEY, validate="one_to_one",
        )
    if len(combined) != expected:
        raise ValueError("Some Top10 and Top5 configurations did not match")
    if not (combined["benchmark_total_return_top10"] -
            combined["benchmark_total_return_top5"]).abs().lt(1e-12).all():
        raise ValueError("Benchmark returns differ between portfolio sizes")
    for size in ("top10", "top5"):
        if not (combined[f"total_return_{size}_stress"] <=
                combined[f"total_return_{size}" ] + 1e-10).all():
            raise ValueError(f"Extra costs unexpectedly improve {size} returns")

    combined["top5_minus_top10_percentage_points"] = 100 * (
        combined["total_return_top5"] - combined["total_return_top10"]
    )
    combined = combined.sort_values(KEY).reset_index(drop=True)
    output = NEW_DIR / "comparison_top10_vs_top5_2026.csv"
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(combined)} paired strategies: {output}")
    print(
        "Top5 higher return:", int((combined["total_return_top5"] >
                                      combined["total_return_top10"]).sum()),
        "/", expected,
    )
    print(
        "Top5 deeper drawdown:", int((combined["max_drawdown_top5"] <
                                       combined["max_drawdown_top10"]).sum()),
        "/", expected,
    )
    print(
        "Top5 higher fees:", int((combined["total_fees_cny_top5"] >
                                 combined["total_fees_cny_top10"]).sum()),
        "/", expected,
    )


if __name__ == "__main__":
    main()
