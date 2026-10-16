"""Run small, serial rank/rank account batches without relaxing RAM safeguards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from paper_futu import final_ablation_core as core
from scripts.compare_all_full15_offsets import OUTPUT, SOURCE, isolate_output
from src.runtime_budget import require_memory


ROOT = Path(__file__).resolve().parents[1]
SPEC = "lgb_15_rank_rank"
RANK_TASK_COUNT = 3068
MAX_JOBS_PER_PROCESS = 20
CONTROLLER_STATUS = OUTPUT / "rank_rank_micro_controller.json"
MIN_FRESH_PROCESS_GIB = 3.6


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def checked_manifest() -> dict:
    manifest = core.read_json(OUTPUT / "manifest.json")
    if core.code_hash() != manifest["scientific_code_sha256"]:
        raise ValueError("Frozen scientific code changed")
    if core.digest(ROOT / "scripts/compare_all_full15_offsets.py") != manifest["comparison_script_sha256"]:
        raise ValueError("Prepared comparison code changed")
    if any(core.digest(SOURCE / name) != expected for name, expected in manifest["source_sha256"].items()):
        raise ValueError("Frozen source study changed")
    specs = {item["candidate_key"]: item["spec"] for item in manifest["candidates"]}
    tasks = manifest["tasks"]
    if not (len(tasks) > RANK_TASK_COUNT
            and all(specs[item["candidate_key"]] == SPEC for item in tasks[:RANK_TASK_COUNT])
            and specs[tasks[RANK_TASK_COUNT]["candidate_key"]] != SPEC):
        raise ValueError("Rank/rank prefix does not match the frozen comparison manifest")
    return manifest


def task_complete(case_id: str) -> bool:
    path = OUTPUT / "tasks" / f"{case_id}.json"
    if not path.exists():
        return False
    return core.read_json(path).get("status") == "complete"


def run_once(start: int, maximum: int, stop: int = RANK_TASK_COUNT) -> None:
    manifest = checked_manifest()
    if not (0 <= start < stop <= RANK_TASK_COUNT
            and 1 <= maximum <= MAX_JOBS_PER_PROCESS):
        raise ValueError("Invalid micro-batch range")
    isolate_output()
    from paper_futu.final_ablation_account import replay

    index = start
    performed = 0
    while index < stop and performed < maximum:
        job = manifest["tasks"][index]
        if task_complete(job["case_id"]):
            index += 1
            continue
        available = psutil.virtual_memory().available / 2**30
        if performed and available < 2.2:
            break
        require_memory(2.0, "完整15因子单进程分段回测")
        result = replay(job["prediction_path"], job["portfolio"], job["rebalance"],
                        job["offset"], 0.0)
        if result["rows"][0]["case_id"] != job["case_id"]:
            raise ValueError("Micro-batch account identity mismatch")
        performed += 1
        index += 1
    result_path = OUTPUT / f"rank_rank_micro_{start:04d}.json"
    write_json(result_path, {"start": start, "stop": stop, "next_index": index,
                             "new_accounts": performed,
                             "pid": os.getpid(), "finished": time.time(),
                             "available_gib": psutil.virtual_memory().available / 2**30})
    print(f"micro start={start} next={index} new={performed}", flush=True)


def active_replay_workers() -> list[int]:
    result = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if ("-m" in command and
                ("scripts.compare_all_full15_offsets" in command or
                 ("scripts.rank_rank_microbatch" in command and "--mode" in command and "once" in command))):
            result.append(process.pid)
    return result


def controller() -> None:
    manifest = checked_manifest()
    if active_replay_workers():
        raise RuntimeError(f"Another account replay is active: {active_replay_workers()}")
    started = time.time()
    index = next((i for i, job in enumerate(manifest["tasks"][:RANK_TASK_COUNT])
                  if not task_complete(job["case_id"])), RANK_TASK_COUNT)
    batches = 0
    write_json(CONTROLLER_STATUS, {"status": "running", "controller_pid": os.getpid(),
                                   "started": started, "next_index": index, "batches": batches})
    try:
        while index < RANK_TASK_COUNT:
            if task_complete(manifest["tasks"][index]["case_id"]):
                index += 1
                continue
            wait_started = time.time()
            while psutil.virtual_memory().available / 2**30 < MIN_FRESH_PROCESS_GIB:
                if time.time() - wait_started > 1800:
                    raise MemoryError("Could not reach 3.6 GiB available RAM for 30 minutes")
                time.sleep(15)
            if active_replay_workers():
                raise RuntimeError("Another replay worker appeared during serial control")
            command = [sys.executable, "-u", "-m", "scripts.rank_rank_microbatch",
                       "--mode", "once", "--start", str(index), "--max-jobs",
                       str(MAX_JOBS_PER_PROCESS)]
            child = subprocess.Popen(command, cwd=ROOT)
            write_json(CONTROLLER_STATUS, {"status": "running", "controller_pid": os.getpid(),
                                           "started": started, "next_index": index,
                                           "active_pid": child.pid, "batches": batches,
                                           "updated": time.time()})
            exit_code = child.wait()
            if exit_code != 0:
                raise RuntimeError(f"Micro-batch at index {index} exited {exit_code}")
            result = core.read_json(OUTPUT / f"rank_rank_micro_{index:04d}.json")
            next_index = int(result["next_index"])
            if not (index < next_index <= RANK_TASK_COUNT and result["new_accounts"] >= 1):
                raise RuntimeError(f"Micro-batch did not make account progress at index {index}")
            index = next_index
            batches += 1
            write_json(CONTROLLER_STATUS, {"status": "running", "controller_pid": os.getpid(),
                                           "started": started, "next_index": index,
                                           "active_pid": None, "batches": batches,
                                           "updated": time.time()})
            if batches % 25 == 0:
                print(f"rank/rank prefix={index}/{RANK_TASK_COUNT} batches={batches}", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.report_rank_rank_offsets"],
                       cwd=ROOT, check=True)
        if not (OUTPUT / "rank_rank_phase_ranking.csv").exists():
            raise RuntimeError("Audited rank/rank report was not created")
        write_json(CONTROLLER_STATUS, {"status": "complete", "controller_pid": os.getpid(),
                                       "started": started, "finished": time.time(),
                                       "next_index": index, "batches": batches,
                                       "report": str(OUTPUT / "rank_rank_phase_ranking.csv")})
    except BaseException as error:
        write_json(CONTROLLER_STATUS, {"status": "failed", "controller_pid": os.getpid(),
                                       "started": started, "failed": time.time(),
                                       "next_index": index, "batches": batches,
                                       "error": repr(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("once", "controller"), required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int, default=RANK_TASK_COUNT)
    parser.add_argument("--max-jobs", type=int, default=MAX_JOBS_PER_PROCESS)
    args = parser.parse_args()
    if args.mode == "once":
        if args.start is None:
            parser.error("--start is required with --mode once")
        run_once(args.start, args.max_jobs, args.stop)
    else:
        controller()


if __name__ == "__main__":
    main()
