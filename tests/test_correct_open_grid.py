import pandas as pd

from paper_futu.backtest_correct_open_grid_2022_2026 import (
    END,
    KINDS,
    SCHEDULES,
    YEARS,
)
from paper_futu.backtest_label_grid_2022_2025 import expression
from src.factor_library import ENHANCED_FEATURE_NAMES


def test_requested_open_label_grid_is_exact():
    assert SCHEDULES == {
        5: (5,),
        10: (1, 5, 10, 30),
        20: (1, 5, 10, 30),
        30: (1, 5, 10, 30),
        40: (5, 10, 20, 30, 40),
        60: (1, 5, 10, 20, 30, 40, 50, 60),
    }
    assert sum(map(len, SCHEDULES.values())) == 26
    assert KINDS == ("open_to_open",)
    assert list(YEARS) == [2022, 2023, 2024, 2025, 2026]
    assert END == pd.Timestamp("2026-10-13")


def test_label_holding_period_and_original_feature_count():
    assert len(ENHANCED_FEATURE_NAMES) == 15
    assert all(not feature.startswith("KMID") for feature in ENHANCED_FEATURE_NAMES)
    for horizon in SCHEDULES:
        assert expression(horizon, "open_to_open") == (
            f"Ref($open, -{horizon + 1}) / Ref($open, -1) - 1"
        )
