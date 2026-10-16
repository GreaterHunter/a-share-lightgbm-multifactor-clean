import numpy as np
import pandas as pd
import pytest

from paper_futu.backtest_label_grid_2022_2025 import (
    daily_ic, mature, signal_execution_years, execution_limit_expressions, expression,
)


def test_vectorized_daily_ic_matches_pandas_with_ties_and_missing_values():
    index = pd.MultiIndex.from_product([["A", "B", "C", "D", "E"],
                                        pd.to_datetime(["2024-01-02", "2024-01-03"])],
                                       names=["instrument", "datetime"])
    prediction = pd.Series([1, 2, 1, 3, 3, 1, 4, np.nan, 5, 5], index=index)
    outcome = pd.Series([2, 2, 3, 1, 1, 4, 5, 3, 5, 5], index=index)
    result = daily_ic(prediction, outcome, minimum=3)
    for date, panel in pd.DataFrame({"p": prediction, "y": outcome}).groupby(level="datetime"):
        assert result.loc[date, "ic"] == pytest.approx(panel.p.corr(panel.y, method="pearson"))
        assert result.loc[date, "rank_ic"] == pytest.approx(panel.p.corr(panel.y, method="spearman"))


def test_constant_prediction_is_undefined_not_zero_ic():
    index = pd.MultiIndex.from_product([["A", "B", "C"], [pd.Timestamp("2024-01-02")]],
                                       names=["instrument", "datetime"])
    result = daily_ic(pd.Series(1.0, index=index), pd.Series([1, 2, 3], index=index), minimum=3)
    assert result[["ic", "rank_ic"]].isna().all().all()


def test_year_boundary_signal_uses_model_available_for_its_execution_year():
    calendar = pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-04",
                               "2022-12-29", "2022-12-30", "2023-01-03"])
    result = signal_execution_years(calendar, pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-03"))
    assert result.loc[pd.Timestamp("2021-12-31")] == 2022
    assert result.loc[pd.Timestamp("2022-12-30")] == 2023


def test_maturity_cutoff_excludes_known_later_outcomes_even_if_signal_is_earlier():
    index = pd.MultiIndex.from_tuples([("A", pd.Timestamp("2021-12-01")),
                                      ("B", pd.Timestamp("2021-12-20"))],
                                     names=["instrument", "datetime"])
    frame = pd.DataFrame({"label": [0.1, 0.2], "target": [0.1, 0.2],
                          "label_end_date": pd.to_datetime(["2021-12-31", "2022-01-10"])}, index=index)
    assert mature(frame, "2010-01-01", "2021-12-31").index.tolist() == [index[0]]


def test_morning_tradability_does_not_use_same_day_final_close():
    buy, sell = execution_limit_expressions("open")
    assert buy == "($open / Ref($close, 1) - 1) >= 0.095"
    assert sell == "($open / Ref($close, 1) - 1) <= -0.095"


@pytest.mark.parametrize("horizon", [5, 10, 15, 20, 30, 40, 60])
def test_label_exit_matches_holding_period(horizon):
    assert expression(horizon, "open_to_open") == f"Ref($open, -{horizon+1}) / Ref($open, -1) - 1"
    assert expression(horizon, "open_to_close") == f"Ref($close, -{horizon}) / Ref($open, -1) - 1"
