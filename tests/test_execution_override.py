from datetime import datetime
import json

import pytest

from paper_futu import execution_override as override
from paper_futu.cn_common import CHINA_TZ


@pytest.fixture
def authorized(tmp_path, monkeypatch):
    monkeypatch.setattr(override, "STATE_DIR", tmp_path)
    value=dict(status="authorized", environment="SIMULATE", signal_date="2026-10-19", execution_date="2026-10-20",
        not_before="2026-10-20T13:00:00+08:00", expires_at="2026-10-20T13:20:00+08:00",
        model_sha256="frozen", user_authorization="human requested 13:00")
    (tmp_path/"execution_override_2026-10-20.json").write_text(json.dumps(value))
    return value


@pytest.mark.parametrize("stamp,allowed", [("2026-10-20T09:31:00",False),("2026-10-20T12:59:59",False),
    ("2026-10-20T13:00:00",True),("2026-10-20T13:19:59",True),("2026-10-20T13:20:00",False)])
def test_one_time_window(authorized, stamp, allowed):
    assert override.execution_allowed(authorized, datetime.fromisoformat(stamp).replace(tzinfo=CHINA_TZ)) is allowed


def test_next_day_normal_schedule_is_not_changed(authorized):
    tomorrow=dict(authorized, signal_date="2026-10-20", execution_date="2026-10-21")
    assert override.execution_allowed(tomorrow,datetime(2026,10,21,9,31,tzinfo=CHINA_TZ))


def test_different_signal_model_is_blocked(authorized):
    with pytest.raises(ValueError,match="binding mismatch"):
        override.execution_allowed(dict(authorized,model_sha256="old_model"),datetime(2026,10,20,13,tzinfo=CHINA_TZ))


def test_late_activation_requires_unchanged_old_plan_and_no_submissions(authorized,tmp_path,monkeypatch):
    import hashlib
    from paper_futu import cn_common
    (tmp_path/"active_plan.json").write_text('{"unchanged":true}')
    value=dict(authorized,allow_delayed_activation=True,metadata_sha256="metadata",
        source_plan_sha256=hashlib.sha256((tmp_path/"active_plan.json").read_bytes()).hexdigest())
    (tmp_path/"execution_override_2026-10-20.json").write_text(json.dumps(value))
    deployment=tmp_path/"config.json"
    deployment.write_text(json.dumps(dict(pending_deployment=dict(effective_signal_date="2026-10-19",
        model_sha256="frozen",metadata_sha256="metadata"))))
    monkeypatch.setattr(cn_common,"DEPLOYMENT_PATH",deployment)
    now=datetime(2026,10,20,12,30,tzinfo=CHINA_TZ)
    assert override.late_activation_allowed("2026-10-19",now)
    (tmp_path/"orders_2026-10-20.jsonl").write_text('{"event":"before_submit"}\n')
    with pytest.raises(ValueError,match="reconcile"):
        override.late_activation_allowed("2026-10-19",now)
