"""Protect partial-year prediction timing and financial availability."""

from types import SimpleNamespace

import pandas as pd

from src import run_until_2026_07 as experiment
from src.train_walk_forward_cv import build_validation_folds, latest_training_period


def test_2026_model_selection_uses_only_earlier_years():
    folds = build_validation_folds(2026, 3, 2010)
    assert [fold["validation_year"] for fold in folds] == ["2023", "2024", "2025"]
    assert latest_training_period(2026, 2010) == ("2010-01-01", "2025-12-31")
    assert all(pd.Timestamp(fold["valid_end"]).year < 2026 for fold in folds)


def test_cutoff_hides_unmatured_labels_without_dropping_predictions(monkeypatch):
    calendar = pd.bdate_range("2026-07-01", periods=30)
    monkeypatch.setattr(experiment, "D", SimpleNamespace(calendar=lambda **kwargs: calendar))
    dates = calendar[:15]
    index = pd.MultiIndex.from_arrays(
        [["SH600000"] * len(dates), dates], names=["instrument", "datetime"]
    )
    frame = pd.DataFrame({"prediction": 0.1, experiment.RAW_LABEL: 0.02}, index=index)
    cutoff = calendar[20]
    censored = experiment.censor_evaluation_labels(frame, cutoff)
    assert censored.index.equals(frame.index)
    assert censored.prediction.equals(frame.prediction)
    assert censored.loc[("SH600000", calendar[10]), experiment.RAW_LABEL] == 0.02
    assert censored.loc[("SH600000", calendar[11]), experiment.RAW_LABEL] != 0.02
    assert censored[experiment.RAW_LABEL].notna().sum() == 11


def test_unannounced_half_year_report_is_not_available_in_july(monkeypatch):
    financials = pd.DataFrame(
        {
            "instrument": ["SH600000", "SH600000"],
            "report_date": pd.to_datetime(["2026-03-31", "2026-06-30"]),
            "book_value_per_share": [10.0, 11.0],
        }
    )
    disclosures = pd.DataFrame(
        {
            "instrument": ["SH600000", "SH600000"],
            "report_date": financials.report_date,
            "disclosure_date": pd.to_datetime(["2026-04-25", "2026-08-20"]),
        }
    )
    monkeypatch.setattr(experiment.pd, "read_parquet", lambda path: financials.copy())
    available = experiment.load_financial_events(disclosures)
    assert len(available) == 1
    assert available.iloc[0].report_date == pd.Timestamp("2026-03-31")
    assert available.iloc[0].available_date == pd.Timestamp("2026-04-26")
