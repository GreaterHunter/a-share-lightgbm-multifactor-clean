"""Contracts for information available immediately after the opening auction."""

import numpy as np
import pandas as pd
import pytest

from src.alpha158_morning import (
    alpha158_registry,
    feature_names,
    morning_end_dates,
    morning_registry,
    normalize_local_bar,
    specification,
)
from src.forward_labels import MORNING_OPEN_PROXY, label_contract, validate_open_label_metadata
from src.morning_observations import validated_opening_quotes


def test_complete_alpha158_is_shifted_as_a_whole_expression():
    official = alpha158_registry()
    morning = morning_registry()
    assert len(official) == 158
    assert len(set(official)) == 158
    assert len(feature_names()) == 162
    assert len(feature_names(False)) == 161
    for name, expression in official.items():
        assert morning[name] == f"Ref({expression}, 1)"
    assert morning["morning_open_gap"] == "$open / Ref($close, 1) - 1"
    assert specification(10)["feature_count"] == 162


def test_morning_label_is_current_observed_open_to_ten_sessions_later():
    contract = label_contract(10, MORNING_OPEN_PROXY)
    assert contract["label_expression"] == "Ref($open, -10) / $open - 1"
    assert contract["label_end_offset"] == 10
    assert contract["requires_intraday_execution_validation"] is True
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_open_label_metadata(contract)


def test_label_end_uses_sessions_and_preserves_missing_future():
    calendar = pd.bdate_range("2026-10-12", periods=12)
    index = pd.MultiIndex.from_tuples(
        [("SH600000", calendar[0]), ("SH600000", calendar[2])],
        names=["instrument", "datetime"],
    )
    end = morning_end_dates(index, calendar, 10)
    assert end.iloc[0] == calendar[10]
    assert pd.isna(end.iloc[1])


def test_local_quote_is_converted_to_verified_qlib_units():
    quote = {"suspension": False, "volume": 1_000_000, "turnover": 10_000_000,
             "open_price": 10, "last_price": 10, "high_price": 11, "low_price": 9}
    data = normalize_local_bar(quote, adjusted_factor=0.5)
    assert data["open"] == 5
    assert data["volume"] == 20_000
    assert data["amount"] == 10_000
    assert data["vwap"] == 5
    assert data["factor"] == 0.5
    suspended = normalize_local_bar({**quote, "suspension": True}, adjusted_factor=0.5)
    assert np.isnan(suspended["open"])
    assert suspended["factor"] == 0.5


def test_invalid_vwap_is_rejected():
    quote = {"suspension": False, "volume": 1_000_000, "turnover": 20_000_000,
             "open_price": 10, "last_price": 10, "high_price": 11, "low_price": 9}
    with pytest.raises(ValueError, match="VWAP"):
        normalize_local_bar(quote, 1)


def test_opening_quote_must_precede_the_decision_and_follow_the_auction():
    today = pd.Timestamp("2026-10-13")
    quote = pd.DataFrame({"code": ["SH.600000"], "update_time": ["2026-10-13 09:25:10"],
                          "open_price": [10.0], "prev_close_price": [9.9], "suspension": [False]})
    result = validated_opening_quotes(quote, today, pd.Timestamp("2026-10-13 09:31:00"))
    assert len(result) == 1
    with pytest.raises(ValueError, match="stale, future-dated, or arrived after"):
        validated_opening_quotes(quote.assign(update_time="2026-10-13 09:24:59"), today,
                                 pd.Timestamp("2026-10-13 09:31:00"))
    with pytest.raises(ValueError, match="stale, future-dated, or arrived after"):
        validated_opening_quotes(quote.assign(update_time="2026-10-13 09:31:01"), today,
                                 pd.Timestamp("2026-10-13 09:31:00"))
