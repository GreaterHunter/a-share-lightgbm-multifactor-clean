"""Offline tests for the paper-vs-backtest execution audit."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from paper_futu.audit_execution_alignment import execution_gap, planned_lot_feasibility
from paper_futu.cn_common import STATE_DIR


def test_small_capital_audit_matches_original_plan() -> None:
    journal = STATE_DIR / "orders_2026-10-13.jsonl"
    if not journal.exists():
        pytest.skip("Local paper execution journal is not available")
    start = next(json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
                 if json.loads(line).get("event") == "execution_start")
    assert planned_lot_feasibility(start, 5000).buyable.sum() == 0
    assert planned_lot_feasibility(start, 10000).buyable.sum() == 3
    assert planned_lot_feasibility(start, 1_000_000).buyable.sum() == 10


def test_execution_gap_is_signed_relative_to_buy_or_sell() -> None:
    orders = pd.DataFrame([
        {"code": "SH.600000", "trd_side": "BUY", "order_status": "FILLED_ALL",
         "dealt_qty": 100, "dealt_avg_price": 10.1},
        {"code": "SH.600001", "trd_side": "SELL", "order_status": "FILLED_ALL",
         "dealt_qty": 100, "dealt_avg_price": 9.9},
    ])
    quotes = pd.DataFrame([
        {"code": "SH.600000", "open_price": 10.0},
        {"code": "SH.600001", "open_price": 10.0},
    ])
    result = execution_gap(orders, quotes)
    assert result.adverse_gap_pct.tolist() == pytest.approx([1, 1])


def test_execution_gap_rejects_unfilled_orders() -> None:
    orders = pd.DataFrame([{"code": "SH.600000", "trd_side": "BUY",
                            "order_status": "SUBMITTED", "dealt_qty": 0,
                            "dealt_avg_price": 0}])
    quotes = pd.DataFrame([{"code": "SH.600000", "open_price": 10.0}])
    with pytest.raises(ValueError, match="fully filled"):
        execution_gap(orders, quotes)
