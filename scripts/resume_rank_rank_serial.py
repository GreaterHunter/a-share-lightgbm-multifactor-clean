"""Resume only the rank/rank offset accounts in memory-bounded serial batches."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from paper_futu import final_ablation_core as core
from scripts.compare_all_full15_offsets import OUTPUT, SOURCE


ROOT = Path(__file__).resolve().parents[1]
STATUS = OUTPUT / "rank_only_serial_controller.json"
LIMITS = (600, 900, 1200, 1534)
MIN_START_GIB = 2.5


def save_status(value: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def active_account_workers() -> list[int]:
    active = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if ("-m" in command and "scripts.compare_all_full15_offsets" in command
                and "--stage" in command and "worker" in command):
            active.append(process.pid)
    return active


def validate_scope() -> dict:
    manifest = core.read_json(OUTPUT / "manifest.json")
    if core.code_hash() != manifest["scientific_code_sha256"]:
        raise ValueError("Scientific source differs from the prepared manifest")
    if core.digest(ROOT / "scripts/compare_all_full15_offsets.py") != manifest["comparison_script_sha256"]:
        raise ValueError("Comparison script differs from the prepared manifest")
    if any(core.digest(SOURCE / name) != digest for name, digest in manifest["source_sha256"].items()):
        raise ValueError("Frozen source result changed")
    spec = {candidate["candidate_key"]: candidate["spec"] for candidate in manifest["candidates"]}
    for shard in (0, 1):
        jobs = manifest["tasks"][shard::2]
        if len(jobs) < LIMITS[-1] + 1:
            raise ValueError("Unexpected worker partition")
        if not all(spec[job["candidate_key"]] == "lgb_15_rank_rank" for job in jobs[:LIMITS[-1]]):
            raise ValueError("Rank/rank jobs are not the expected worker prefix")
        if spec[jobs[LIMITS[-1]]["candidate_key"]] == "lgb_15_rank_rank":
            raise ValueError("Rank/rank scope extends beyond the configured stop")
    return manifest


def main() -> None:
    validate_scope()
    if active_account_workers():
        raise RuntimeError(f"Another account worker is active: {active_account_workers()}")
    started = time.time()
    completed = []
    save_status({"status": "running", "controller_pid": os.getpid(),
                 "started": started, "completed_batches": completed})
    try:
        for limit in LIMITS:
            for shard in (0, 1):
                free_gib = psutil.virtual_memory().available / 2**30
                if free_gib < MIN_START_GIB:
                    raise MemoryError(f"Only {free_gib:.2f} GiB free before batch; required {MIN_START_GIB:.1f} GiB")
                if active_account_workers():
                    raise RuntimeError("Another account worker appeared before a serial batch")
                command = [sys.executable, "-u", "-m", "scripts.compare_all_full15_offsets",
                           "--stage", "worker", "--shard", str(shard), "--shards", "2",
                           "--max-tasks", str(limit)]
                child = subprocess.Popen(command, cwd=ROOT)
                save_status({"status": "running", "controller_pid": os.getpid(),
                             "started": started, "active_shard": shard,
                             "active_limit": limit, "active_pid": child.pid,
                             "completed_batches": completed, "updated": time.time()})
                print(f"START rank/rank shard={shard} prefix={limit} pid={child.pid}", flush=True)
                code = child.wait()
                if code != 0:
                    raise RuntimeError(f"Rank/rank batch shard={shard} prefix={limit} exited {code}")
                completed.append({"shard": shard, "prefix": limit,
                                  "finished": time.time()})
                save_status({"status": "running", "controller_pid": os.getpid(),
                             "started": started, "active_pid": None,
                             "completed_batches": completed, "updated": time.time()})
        report = subprocess.run([sys.executable, "-m", "scripts.report_rank_rank_offsets"],
                                cwd=ROOT, check=True)
        if report.returncode != 0 or not (OUTPUT / "rank_rank_phase_ranking.csv").exists():
            raise RuntimeError("Rank/rank audited report was not created")
        save_status({"status": "complete", "controller_pid": os.getpid(),
                     "started": started, "finished": time.time(),
                     "completed_batches": completed,
                     "report": str(OUTPUT / "rank_rank_phase_ranking.csv")})
    except BaseException as error:
        save_status({"status": "failed", "controller_pid": os.getpid(),
                     "started": started, "failed": time.time(),
                     "completed_batches": completed, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
