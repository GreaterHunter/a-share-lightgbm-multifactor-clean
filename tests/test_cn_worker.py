"""Exercise scheduling without opening a broker connection or placing any order."""

from datetime import datetime
import json

import pytest

from paper_futu import cn_worker as worker
from paper_futu.cn_common import CHINA_TZ


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    from paper_futu import cn_rank80_runtime
    monkeypatch.setattr(cn_rank80_runtime, "enabled", lambda: False)
    plan = {"signal_date": "2026-10-12", "execution_date": "2026-10-13",
            "next_rebalance_signal_date": "2026-10-26", "model_year": 2026,
            "top_k": 10, "n_drop": 3, "model_dir": "models/test", "model_sha256": "test-model"}
    (tmp_path / "active_plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(worker, "STATE_DIR", tmp_path)
    monkeypatch.setattr(worker, "load_deployment", lambda: (tmp_path, {}))
    monkeypatch.setattr(worker, "validate_plan_deployment", lambda *args: None)
    monkeypatch.setattr(worker, "verify_active_protocol", lambda: {"version": 8, "strategy": {"model_sha256": "test-model"}})
    monkeypatch.setattr(worker, "trading_dates", lambda: {"2026-10-12", "2026-10-13", "2026-10-14"})

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(worker.socket, "create_connection", lambda *args, **kwargs: FakeSocket())
    calls = []
    monkeypatch.setattr(worker, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    return tmp_path, calls


@pytest.mark.parametrize("stamp", ["2026-10-12T23:16:00", "2026-10-13T09:30:00",
                                    "2026-10-13T12:00:00", "2026-10-13T14:55:00"])
def test_no_submit_outside_planned_execution_window(scheduler, monkeypatch, stamp):
    _, calls = scheduler
    monkeypatch.setattr(worker, "china_now", lambda: datetime.fromisoformat(stamp).replace(tzinfo=CHINA_TZ))
    expected = "execution_window_missed" if stamp in {"2026-10-13T12:00:00", "2026-10-13T14:55:00"} else "waiting"
    assert worker.tick()["stage"] == expected
    assert calls == []


def test_planned_morning_executes_and_completion_prevents_duplicates(scheduler, monkeypatch):
    state_dir, calls = scheduler
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 13, 9, 31, tzinfo=CHINA_TZ))
    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        (state_dir / "execution_2026-10-13.json").write_text('{"status":"completed"}')
    monkeypatch.setattr(worker, "run", execute)
    assert worker.tick()["stage"] == "rebalance_completed"
    assert calls == [(("paper_futu.execute_cn_plan", "--submit"), {})]
    (state_dir / "execution_2026-10-13.json").write_text("{}")
    assert worker.tick()["stage"] == "waiting"
    assert len(calls) == 1


def test_missed_execution_date_does_not_chase_a_stale_signal(scheduler, monkeypatch):
    _, calls = scheduler
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 14, 9, 31, tzinfo=CHINA_TZ))
    with pytest.raises(RuntimeError, match="过期信号"):
        worker.tick()
    assert calls == []


@pytest.mark.parametrize("stamp,expected", [
    ("2026-10-13T09:19:59", "outside_session"),
    ("2026-10-13T09:20:00", None),
    ("2026-10-13T15:25:00", None),
    ("2026-10-13T16:30:00", "session_deadline_error"),
    ("2026-10-17T09:20:00", "non_trading_day"),
])
def test_daily_session_boundaries(scheduler, stamp, expected):
    now = datetime.fromisoformat(stamp).replace(tzinfo=CHINA_TZ)
    assert worker.session_exit_status(now) == expected


def test_exit_only_after_both_closing_outputs_exist(scheduler):
    state_dir, _ = scheduler
    now = datetime(2026, 10, 13, 15, 35, tzinfo=CHINA_TZ)
    daily = state_dir / "2026-10-13"
    daily.mkdir()
    (daily / "plan.json").write_text("{}")
    assert worker.session_exit_status(now) is None
    (state_dir / "nav_saved_2026-10-13.json").write_text("{}")
    assert worker.session_exit_status(now) == "session_completed"


def test_one_stock_catchup_runs_only_on_its_date_and_does_not_retry_submitted_order(scheduler, monkeypatch):
    state_dir, calls = scheduler
    (state_dir / "execution_2026-10-13.json").write_text('{"status":"partial_verified"}')
    (state_dir / "catchup_plan.json").write_text('{"execution_date":"2026-10-14"}')
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 14, 9, 31, tzinfo=CHINA_TZ))
    assert worker.tick()["stage"] == "catchup_completed"
    assert calls == [(("paper_futu.cn_catchup", "--submit"), {})]
    (state_dir / "catchup_orders_2026-10-14.jsonl").write_text('{"event":"before_submit"}\n')
    assert worker.tick()["stage"] == "catchup_reconciliation_required"
    assert len(calls) == 1
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 14, 10, 1, tzinfo=CHINA_TZ))
    assert worker.tick()["stage"] == "catchup_window_missed_or_requires_review"
    assert len(calls) == 1


def test_cancelled_catchup_never_submits_even_in_execution_window(scheduler, monkeypatch):
    state_dir, calls = scheduler
    (state_dir / "execution_2026-10-13.json").write_text('{"status":"partial_verified"}')
    (state_dir / "catchup_plan.json").write_text(
        '{"execution_date":"2026-10-14","status":"cancelled"}'
    )
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 14, 9, 31, tzinfo=CHINA_TZ))
    assert worker.tick()["stage"] == "single_stock_catchup_cancelled"
    assert calls == []


def test_worker_exits_before_0920_without_broker_calls(scheduler, monkeypatch):
    state_dir, calls = scheduler
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 13, 9, 19, tzinfo=CHINA_TZ))
    monkeypatch.setattr(worker.sys, "argv", ["cn_worker"])
    worker.main()
    assert calls == []
    assert json.loads((state_dir / "worker_status.json").read_text())["stage"] == "outside_session"


def test_data_update_precedes_broker_check_when_opend_is_offline(scheduler, monkeypatch):
    state_dir, calls = scheduler
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 13, 15, 40, tzinfo=CHINA_TZ))
    def disconnected(*args, **kwargs):
        assert calls == [(("paper_futu.update_cn_data", "--date", "2026-10-13"), {"model": True})]
        raise ConnectionRefusedError("OpenD intentionally offline")
    monkeypatch.setattr(worker.socket, "create_connection", disconnected)
    result = worker.tick()
    assert result["stage"] == "broker_offline"
    assert not (state_dir/"nav_saved_2026-10-13.json").exists()
    assert all("--submit" not in args for args, _ in calls)
