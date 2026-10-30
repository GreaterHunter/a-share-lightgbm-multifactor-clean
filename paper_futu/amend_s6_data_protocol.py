"""Enroll an explicit data-ingestion revision before the first S6 execution.

The v1 seal/protocol are never rewritten. Their changed source files must have
been archived byte-for-byte before editing. No model or trading rule may change.
"""
from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

from paper_futu.cn_common import ROOT, STATE_DIR, china_now
from paper_futu.prospective_s6 import digest, read, verify_protocol, write_once

CHANGED = ["paper_futu/cn_worker.py", "paper_futu/build_cn_signal.py"]
ADDED = ["paper_futu/independent_cn_market.py", "paper_futu/update_cn_data.py",
         "paper_futu/amend_s6_data_protocol.py", "scripts/install_cn_data_schedule.ps1"]
ADDED += ["scripts/install_s6_observer_schedule.ps1"]


def amend(root=ROOT, now=None):
    root = Path(root).resolve()
    now = now or china_now()
    old = root/"reports/paper_futu/s6_forward_v1"
    output = root/"reports/paper_futu/s6_forward_v2_public_data"
    if (output/"protocol.json").exists():
        return verify_protocol(output)
    p = read(old/"protocol.json")
    old_hash = digest(old/"protocol.json")
    if old_hash != read(old/"seal.json")["protocol_sha256"]:
        raise ValueError("Old protocol/seal was altered")
    if list((old/"daily").glob("*.json")) or now.date().isoformat() >= p["first_execution_date"]:
        raise ValueError("Cannot reenroll an already started prospective period")
    if (now.date().isoformat() == p["signal_anchor"] and (now.hour, now.minute) >= (15, 35)):
        raise ValueError("Enroll the data revision before observing the signal-day close")
    if (root/"data/interim/futu_cn_paper"/p["signal_anchor"]/"plan.json").exists():
        raise ValueError("Signal already generated; do not retroactively change its source policy")
    before_after = {}
    bindings = dict(p["immutable_bindings"])
    for name, expected in list(bindings.items()):
        relative = Path(name).relative_to(root).as_posix()
        if relative in CHANGED:
            archived = old/"source_before_independent_data"/relative
            if digest(archived) != expected:
                raise ValueError("Original sealed source not preserved: " + relative)
            before_after[relative] = dict(previous_sha256=expected, new_sha256=digest(name),
                                         preserved_source=str(archived))
            bindings[name] = digest(name)
        elif digest(name) != expected:
            raise ValueError("Unapproved model/data/rule change: " + relative)
    if len(before_after) != len(CHANGED):
        raise ValueError("Unexpected original source bindings")
    for relative in ADDED:
        bindings[str(root/relative)] = digest(root/relative)
    revision = dict(parent_version=1, parent_protocol_sha256=old_hash,
        authorized_change="User requested automatic daily data and inference without OpenD; no model/portfolio/order change",
        effective_before_signal_close=p["signal_anchor"], changed_sources=before_after,
        historical_restatement=False, account_or_orders_changed=False)
    p.update(version=2, frozen_at_utc=now.astimezone(timezone.utc).isoformat(), immutable_bindings=bindings,
        data_source_amendment=revision,
        daily_data_policy=dict(primary="Sina raw daily OHLC/share-volume and dated cumulative adjustment factors",
            cross_check="Tencent raw daily OHLC and normalized share-volume, all active members",
            price_adjustment="Anchor to prior verified Qlib units using the dated factor ratio; never restate history",
            corporate_actions="Cash dividend factors checked against Tencent per-10-share events; unsupported actions block for review",
            membership="Official CSI constituent file; no future-dated universe for signal generation",
            cutoff="Use only completed sessions after 15:35 Asia/Shanghai",
            recovery="Up to 20 price-only sessions; never fabricate historical plans, fills or account NAV",
            retries="Daily 15:40/16:10/16:40/17:10/18:10 and next-morning 08:50; one file lock",
            failure="Block new signal on stale/inconsistent/missing data; preserve active plan",
            broker_boundary="OpenD only for actual quotes, account NAV and order submission"))
    write_once(output/"protocol.json", p)
    write_once(output/"seal.json", dict(protocol_sha256=digest(output/"protocol.json")))
    write_once(output/"migration.json", revision)
    verify_protocol(output)
    return p


if __name__ == "__main__":
    p = amend()
    print(json.dumps(dict(version=p["version"], first_execution_date=p["first_execution_date"],
        model_sha256=p["strategy"]["model_sha256"], data_only_revision=True), ensure_ascii=False))
