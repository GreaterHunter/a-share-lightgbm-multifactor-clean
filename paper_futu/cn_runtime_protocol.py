"""One explicitly pinned forward version for execution and read-only observation."""
from __future__ import annotations

import json
from pathlib import Path
import re

from paper_futu.cn_common import ROOT
from paper_futu.prospective_s6 import digest, verify_protocol

BINDING_PATH = ROOT / "configs/futu_cn_forward_protocol.json"


def active_binding():
    value = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    if value.get("environment") != "SIMULATE" or value.get("status") != "active":
        raise ValueError("Missing active SIMULATE forward protocol binding")
    if not re.fullmatch(r"[0-9a-f]{64}", value.get("protocol_sha256", "")):
        raise ValueError("Forward protocol must be pinned to an exact SHA256")
    relative = Path(value["protocol_directory"])
    directory = (ROOT / relative).resolve()
    if relative.is_absolute() or not directory.is_relative_to((ROOT / "reports/paper_futu").resolve()):
        raise ValueError("Forward protocol directory must be inside reports/paper_futu")
    return value, directory


def active_protocol_directory():
    return active_binding()[1]


def verify_active_protocol():
    binding, directory = active_binding()
    if digest(directory / "protocol.json") != binding["protocol_sha256"]:
        raise ValueError("Active forward protocol differs from the deployment binding")
    protocol = verify_protocol(directory)
    if protocol["strategy"]["model_sha256"] != binding["model_sha256"]:
        raise ValueError("Forward protocol model binding changed")
    return protocol
