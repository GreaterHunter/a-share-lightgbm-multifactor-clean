"""Ensure target timing is explicit and daily scaling cannot create labels."""

from __future__ import annotations

import pandas as pd
import pytest

from src.audit_label_alignment import (
    close_exit_expression,
    latest_mature_signal,
    open_exit_expression,
)


def test_open_exit_matches_next_rebalance_open() -> None:
    assert close_exit_expression(10) == "Ref($close, -10) / Ref($open, -1) - 1"
    assert open_exit_expression(10) == "Ref($open, -11) / Ref($open, -1) - 1"


def test_open_exit_requires_one_extra_trading_day_to_mature() -> None:
    calendar = pd.bdate_range("2026-01-01", periods=30)
    cutoff = calendar[-1]
    assert latest_mature_signal(calendar, cutoff, 10, open_exit=False) == calendar[-11]
    assert latest_mature_signal(calendar, cutoff, 10, open_exit=True) == calendar[-12]
    with pytest.raises(ValueError, match="Not enough"):
        latest_mature_signal(calendar, cutoff, 60, open_exit=True)


def test_dividing_returns_by_horizon_does_not_change_rank() -> None:
    raw = pd.Series([-0.1, -0.02, 0.0, 0.04, 0.18])
    assert raw.rank(method="average").equals((raw / 10).rank(method="average"))
