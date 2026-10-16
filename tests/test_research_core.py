"""Small tests for the timing and cross-sectional rules that protect validity."""

import pandas as pd
import pytest

from src.build_ranked_dataset import cross_sectional_rank
from src.compare_models import build_rebalance_signal
from src.config import load_main_strategy_config
from src.prepare_dataset import purge_last_signal_dates
from src.run_horizon_10_experiment import build_label_expression
from src.train_walk_forward_cv import (
    build_validation_folds,
    latest_training_period,
)


def make_panel(values: list[float], dates: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [["SH600000", "SH600001"] * len(dates), pd.to_datetime(dates).repeat(2)],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame({"value": values}, index=index)


def test_forward_label_starts_at_next_open_and_ends_ten_sessions_later() -> None:
    assert build_label_expression(10) == "Ref($open, -11) / Ref($open, -1) - 1"
    assert build_label_expression(10, "legacy_open_to_close") == "Ref($close, -10) / Ref($open, -1) - 1"
    with pytest.raises(ValueError):
        build_label_expression(0)


def test_cross_sectional_rank_is_computed_independently_each_day() -> None:
    panel = make_panel([1.0, 3.0, 9.0, 2.0], ["2024-01-02", "2024-01-03"])
    ranked = cross_sectional_rank(panel, ["value"])["value"]
    assert ranked.loc[("SH600000", pd.Timestamp("2024-01-02"))] == pytest.approx(0.0)
    assert ranked.loc[("SH600001", pd.Timestamp("2024-01-02"))] == pytest.approx(0.5)
    assert ranked.loc[("SH600000", pd.Timestamp("2024-01-03"))] == pytest.approx(0.5)
    assert ranked.loc[("SH600001", pd.Timestamp("2024-01-03"))] == pytest.approx(0.0)


def test_purge_removes_last_horizon_signal_dates_for_every_stock() -> None:
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    panel = make_panel(list(range(10)), [date.strftime("%Y-%m-%d") for date in dates])
    cleaned, removed_dates = purge_last_signal_dates(panel, label_horizon=2)
    assert list(removed_dates) == list(dates[-2:])
    assert cleaned.index.get_level_values("datetime").max() == dates[-3]
    assert len(cleaned) == 6


def test_rebalance_signal_keeps_only_every_tenth_signal_date() -> None:
    dates = pd.bdate_range("2024-01-02", periods=21)
    panel = make_panel([0.1, 0.2] * len(dates), [date.strftime("%Y-%m-%d") for date in dates])
    panel = panel.rename(columns={"value": "prediction"})
    signal, signal_dates = build_rebalance_signal(panel, rebalance_every=10)
    assert list(signal_dates) == [dates[0], dates[10], dates[20]]
    assert signal.index.get_level_values("datetime").nunique() == 3


def test_frozen_main_strategy_is_expanding_ten_by_ten() -> None:
    config = load_main_strategy_config()
    assert config["model_variant"] == "ranked_enhanced"
    assert config["label_horizon"] == 10
    assert config["rebalance_every"] == 10
    assert config["validation_folds"] == 3
    assert config["iteration_selection_metric"] == "mean_rank_ic"
    assert config["iteration_candidates"] == [10, 20, 40, 80, 160, 320, 640]
    assert config["test_end_year"] == 2024


def test_walk_forward_folds_never_use_test_year() -> None:
    folds = build_validation_folds(
        test_year=2024,
        maximum_folds=3,
        data_start_year=2010,
    )
    assert [fold["validation_year"] for fold in folds] == ["2021", "2022", "2023"]
    assert folds[0]["train_start"] == "2010-01-01"
    assert folds[-1]["train_end"] == "2022-12-31"
    assert latest_training_period(2024, 2010) == (
        "2010-01-01",
        "2023-12-31",
    )
