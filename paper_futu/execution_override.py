"""Date-bound, model-bound single-session execution authorization; no orders."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json

from paper_futu.cn_common import STATE_DIR, CHINA_TZ, china_now


def read_override(now=None):
    now = now or china_now()
    path = STATE_DIR/("execution_override_"+now.date().isoformat()+".json")
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("environment") != "SIMULATE" or value.get("execution_date") != now.date().isoformat()
            or value.get("status") != "authorized" or not value.get("user_authorization")):
        raise ValueError("Invalid single-session paper execution authorization")
    start = datetime.fromisoformat(value["not_before"])
    end = datetime.fromisoformat(value["expires_at"])
    if (start.tzinfo is None or end.tzinfo is None or start.astimezone(CHINA_TZ).date() != now.date()
            or end.astimezone(CHINA_TZ).date() != now.date() or not start < end
            or end.hour*60+end.minute > 14*60+55 or start.hour*60+start.minute < 13*60):
        raise ValueError("Invalid single-session execution time bounds")
    return value


def execution_allowed(plan, now=None):
    now = now or china_now()
    value = read_override(now)
    if value is None or plan.get("execution_date") != value["execution_date"]:
        return True
    for name in ["signal_date", "execution_date", "model_sha256"]:
        if plan.get(name) != value[name]:
            raise ValueError("One-time execution/model/signal binding mismatch")
    return datetime.fromisoformat(value["not_before"]) <= now < datetime.fromisoformat(value["expires_at"])


def late_activation_allowed(signal_date, now=None):
    """Only an unchanged pending transition may be recovered on its trade date."""
    now = now or china_now()
    value = read_override(now)
    if value is None or signal_date != value["signal_date"]:
        return False
    if not value.get("allow_delayed_activation") or now >= datetime.fromisoformat(value["expires_at"]):
        return False
    active = STATE_DIR/"active_plan.json"
    if hashlib.sha256(active.read_bytes()).hexdigest() != value["source_plan_sha256"]:
        return False
    if (STATE_DIR/("execution_"+value["execution_date"]+".json")).exists():
        raise ValueError("Execution already completed; do not reactivate")
    journal = STATE_DIR/("orders_"+value["execution_date"]+".jsonl")
    if journal.exists() and any(json.loads(line).get("event")=="before_submit"
                              for line in journal.read_text(encoding="utf-8").splitlines()):
        raise ValueError("Orders already submitted; reconcile before delayed activation")
    from paper_futu.cn_common import DEPLOYMENT_PATH
    config = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    pending = config.get("pending_deployment", {})
    if (pending.get("effective_signal_date") != signal_date
            or pending.get("model_sha256") != value["model_sha256"]
            or pending.get("metadata_sha256") != value["metadata_sha256"]):
        raise ValueError("Delayed activation differs from the authorized pending model")
    return True
