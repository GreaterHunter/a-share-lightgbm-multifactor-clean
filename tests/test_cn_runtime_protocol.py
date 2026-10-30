"""Version pinning and scheduling regressions, with no broker connection."""
import hashlib
import json
from datetime import datetime

import pytest

from paper_futu import cn_runtime_protocol as runtime
from paper_futu import cn_worker as worker
from paper_futu.cn_common import CHINA_TZ
from tests.test_cn_worker import scheduler


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    pointer = tmp_path / "active.json"
    monkeypatch.setattr(runtime, "BINDING_PATH", pointer)
    directory = tmp_path / "reports/paper_futu/v8"
    directory.mkdir(parents=True)
    source = tmp_path / "source.py"
    source.write_text("unchanged")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protocol = {"version": 8, "strategy": {"model_sha256": "m"},
                "immutable_bindings": {str(source): sha(source)}}
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "seal.json").write_text(json.dumps({"protocol_sha256": sha(directory / "protocol.json")}))
    binding = {"environment": "SIMULATE", "status": "active", "model_sha256": "m",
               "protocol_directory": "reports/paper_futu/v8", "protocol_sha256": sha(directory / "protocol.json")}
    pointer.write_text(json.dumps(binding))
    return pointer, directory, source, binding


def test_exact_active_protocol_is_shared(pinned):
    assert runtime.verify_active_protocol()["version"] == 8
    assert runtime.active_protocol_directory() == pinned[1]


def test_code_change_fails_closed(pinned):
    pinned[2].write_text("changed")
    with pytest.raises(ValueError, match="Frozen code"):
        runtime.verify_active_protocol()


@pytest.mark.parametrize("field,value", [("protocol_sha256", "0" * 64), ("model_sha256", "other"),
                                       ("environment", "REAL"), ("protocol_directory", "../outside")])
def test_bad_binding_is_rejected(pinned, field, value):
    pointer, _, _, binding = pinned
    binding[field] = value
    pointer.write_text(json.dumps(binding))
    with pytest.raises(ValueError):
        runtime.verify_active_protocol()


def test_preopen_detects_stale_seal_before_any_submit(scheduler, monkeypatch):
    state, calls = scheduler
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 13, 9, 20, tzinfo=CHINA_TZ))
    def invalid():
        raise ValueError("Frozen code/data/model changed")
    monkeypatch.setattr(worker, "verify_active_protocol", invalid)
    assert worker.tick()["stage"] == "execution_protocol_blocked"
    assert calls == []
    assert (state / "execution_error_2026-10-13.json").exists()


def test_child_failure_is_latched_and_survives_window_end(scheduler, monkeypatch):
    state, calls = scheduler
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 13, 9, 31, tzinfo=CHINA_TZ))
    def fail(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("intent failed; inspect before retry")
    monkeypatch.setattr(worker, "run", fail)
    assert worker.tick()["stage"] == "execution_failed_requires_review"
    assert worker.tick()["stage"] == "execution_failed_requires_review"
    assert len(calls) == 1
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 13, 11, 0, tzinfo=CHINA_TZ))
    result = worker.tick()
    assert result["stage"] == "execution_window_missed"
    assert "intent failed" in result["last_execution_error"]["message"]


def test_missing_receipt_never_means_completed(scheduler, monkeypatch):
    _, calls = scheduler
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 13, 9, 31, tzinfo=CHINA_TZ))
    assert worker.tick()["stage"] == "execution_receipt_missing"
    assert len(calls) == 1


def test_explicit_afternoon_window_is_respected(scheduler, monkeypatch):
    from paper_futu import cn_execution_session as session
    monkeypatch.setattr(session, "normal_bounds", lambda plan, now: (
        now.replace(hour=13, minute=0), now.replace(hour=13, minute=20)))
    monkeypatch.setattr(worker, "china_now", lambda: datetime(2026, 10, 13, 13, 0, tzinfo=CHINA_TZ))
    # A test-only time override does not create a real execution authorization.
    assert worker.tick()["stage"] == "execution_receipt_missing"
    assert len(scheduler[1]) == 1
