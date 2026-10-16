"""Protect exact label availability when refitting using a later full dataset."""

import pandas as pd

from paper_futu.retrain_cn_latest import LABEL, TARGET, label_end_dates, select_mature


def test_label_maturity_uses_trading_sessions_and_next_rebalance_open():
    calendar = pd.DatetimeIndex([
        "2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22",
        "2026-09-23", "2026-09-24", "2026-09-28", "2026-09-29",
        "2026-09-30", "2026-10-08", "2026-10-09", "2026-10-12",
    ])
    index = pd.MultiIndex.from_tuples(
        [("SH600000", calendar[0]), ("SZ000001", calendar[0]), ("SH600000", calendar[1])],
        names=["instrument", "datetime"],
    )
    result = label_end_dates(index, calendar)
    assert result.iloc[0] == pd.Timestamp("2026-10-12")
    assert result.iloc[1] == pd.Timestamp("2026-10-12")
    assert pd.isna(result.iloc[2])


def test_earlier_refit_ignores_labels_realized_after_its_information_cutoff():
    index = pd.MultiIndex.from_tuples([
        ("SH600000", pd.Timestamp("2026-06-01")),
        ("SH600000", pd.Timestamp("2026-06-16")),
        ("SH600000", pd.Timestamp("2026-06-17")),
        ("SH600000", pd.Timestamp("2026-06-18")),
    ], names=["instrument", "datetime"])
    dataset = pd.DataFrame({LABEL: [0.01, 0.02, 0.03, 0.04], TARGET: [0.1, 0.2, 0.3, 0.4],
                            "label_end_date": pd.to_datetime(["2026-06-16", "2026-06-30", "2026-07-01", None])}, index=index)
    retained = select_mature(dataset, "2010-01-01", "2026-06-30")
    assert retained.index.tolist() == index[:2].tolist()
    changed_future = dataset.copy()
    changed_future.loc[index[2:], [LABEL, TARGET]] = 999
    pd.testing.assert_frame_equal(retained, select_mature(changed_future, "2010-01-01", "2026-06-30"))


def test_no_extra_last_days_are_removed_after_maturity_filter():
    dates = pd.bdate_range("2026-01-01", periods=30)
    index = pd.MultiIndex.from_product([["SH600000"], dates], names=["instrument", "datetime"])
    dataset = pd.DataFrame({LABEL: 0.01, TARGET: 0.1, "label_end_date": label_end_dates(index, dates)}, index=index)
    retained = select_mature(dataset, dates[0], dates[-1])
    assert len(retained) == 19
    assert retained.index.get_level_values("datetime").max() == dates[18]
