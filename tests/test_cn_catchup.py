"""The recovery path must identify one missing fill without duplicating any order."""

import sys

import pandas as pd
import pytest

from paper_futu import cn_catchup
from paper_futu.cn_catchup import submit, validate_original_fills


def original_state():
    selected = ["SH.600000", "SH.600001", "SZ.300316"]
    events = [{"event": "execution_start", "selected": selected,
               "weights": {code: 0.1 for code in selected},
               "actions": [{"code": code, "side": "BUY", "held": 0, "target": 100}
                           for code in selected]}]
    events += [{"event": "submitted", "code": code, "qty": 100, "order_id": str(i)}
               for i, code in enumerate(selected[:2], start=1)]
    orders = pd.DataFrame({"order_id": ["1", "2"], "code": selected[:2],
                           "trd_side": ["BUY", "BUY"], "order_status": ["FILLED_ALL"] * 2,
                           "dealt_qty": [100, 100]})
    positions = pd.DataFrame({"code": selected[:2], "qty": [100, 100]})
    plan = {"ordered_candidates": selected, "top_k": 3}
    return events, orders, positions, plan


def test_exactly_one_unsubmitted_stock_can_be_recovered():
    assert validate_original_fills(*original_state()) == ("SZ.300316", 100, 0.1)


def test_partial_fill_requires_manual_reconciliation():
    events, orders, positions, plan = original_state()
    orders.loc[0, "dealt_qty"] = 50
    with pytest.raises(ValueError, match="未全额"):
        validate_original_fills(events, orders, positions, plan)


def test_unknown_order_requires_manual_reconciliation():
    events, orders, positions, plan = original_state()
    orders.loc[len(orders)] = ["3", "SZ.300316", "BUY", "FILLED_ALL", 100]
    with pytest.raises(ValueError, match="不完全一致"):
        validate_original_fills(events, orders, positions, plan)


def test_prior_submit_for_missing_stock_is_not_retried():
    events, orders, positions, plan = original_state()
    events.append({"event": "before_submit", "code": "SZ.300316"})
    with pytest.raises(ValueError, match="曾进入下单流程"):
        validate_original_fills(events, orders, positions, plan)


def test_cancelled_recovery_rejects_direct_submit_before_broker_access():
    with pytest.raises(ValueError, match="已暂停"):
        submit(None, None, None, {}, {"status": "cancelled"})


def test_cancelled_recovery_cannot_be_reprepared_accidentally(tmp_path, monkeypatch):
    plan = tmp_path / "catchup_plan.json"
    plan.write_text('{"status":"cancelled"}', encoding="utf-8")
    monkeypatch.setattr(cn_catchup, "CATCHUP_PLAN", plan)
    monkeypatch.setattr(sys, "argv", ["cn_catchup", "--prepare"])
    with pytest.raises(ValueError, match="已暂停"):
        cn_catchup.main()
