"""Resume only the frozen rank/rank comparison with at most two short-lived workers.

The two lanes own disjoint manifest indices. A second process starts only when
there is enough free physical memory; each replay keeps the original 2 GiB guard.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from paper_futu import final_ablation_core as core
from scripts.compare_all_full15_offsets import OUTPUT
from scripts.rank_rank_microbatch import (
    RANK_TASK_COUNT,
    ROOT,
    active_replay_workers,
    checked_manifest,
    task_complete,
    write_json,
)


STATUS = OUTPUT / "rank_rank_parallel_micro_controller.json"
MIN_START_GIB = 3.6
MAX_JOBS = 10
DEFAULT_LANE_BOUNDS = ((0, RANK_TASK_COUNT // 2),
                       (RANK_TASK_COUNT // 2, RANK_TASK_COUNT))


def available_gib() -> float:
    return psutil.virtual_memory().available / 2**30


def first_incomplete(tasks: list[dict], start: int, stop: int) -> int:
    return next((index for index in range(start, stop)
                 if not task_complete(tasks[index]["case_id"])), stop)


def other_controllers() -> list[int]:
    result = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        command = process.info["cmdline"] or []
        if ("-m" in command and
                ("scripts.rank_rank_parallel_microbatch" in command or
                 ("scripts.rank_rank_microbatch" in command and
                  "--mode" in command and "controller" in command))):
            result.append(process.pid)
    return result


def controller(lane_bounds: tuple[tuple[int, int], tuple[int, int]]) -> None:
    if not (0 <= lane_bounds[0][0] < lane_bounds[0][1]
            == lane_bounds[1][0] < lane_bounds[1][1] == RANK_TASK_COUNT):
        raise ValueError("Invalid disjoint lane bounds")
    manifest = checked_manifest()
    workers = active_replay_workers()
    controllers = other_controllers()
    if workers or controllers:
        raise RuntimeError(f"Existing replay/controller process: {workers + controllers}")

    tasks = manifest["tasks"]
    lanes = [
        {"lane": number, "index": first_incomplete(tasks, start, stop),
         "stop": stop, "batches": 0, "active_pid": None}
        for number, (start, stop) in enumerate(lane_bounds)
    ]
    children: dict[int, tuple[subprocess.Popen, int, object, object]] = {}
    started = time.time()
    last_work = started
    failure: str | None = None

    def record(status: str, **extra: object) -> None:
        write_json(STATUS, {
            "status": status, "controller_pid": os.getpid(),
            "started": started, "updated": time.time(),
            "available_gib": available_gib(), "max_workers": 2,
            "max_jobs_per_child": MAX_JOBS,
            "lane_bounds": lane_bounds,
            "completed_new_accounts": sum(
                1 for job in tasks[:RANK_TASK_COUNT]
                if task_complete(job["case_id"])
            ),
            "lanes": lanes, **extra,
        })

    record("running")
    try:
        while True:
            for lane_number, (child, start_index, stdout, stderr) in list(children.items()):
                exit_code = child.poll()
                if exit_code is None:
                    continue
                stdout.close()
                stderr.close()
                del children[lane_number]
                lanes[lane_number]["active_pid"] = None
                if exit_code != 0:
                    failure = f"Lane {lane_number} batch {start_index} exited {exit_code}"
                    continue
                result = core.read_json(OUTPUT / f"rank_rank_micro_{start_index:04d}.json")
                next_index = int(result["next_index"])
                if not (start_index < next_index <= lanes[lane_number]["stop"]
                        and int(result["stop"]) == lanes[lane_number]["stop"]
                        and int(result["new_accounts"]) >= 1):
                    failure = f"Lane {lane_number} made no valid progress at {start_index}"
                    continue
                lanes[lane_number]["index"] = next_index
                lanes[lane_number]["batches"] += 1
                last_work = time.time()

            if failure:
                if not children:
                    raise RuntimeError(failure)
                record("draining_after_error", error=failure)
                time.sleep(10)
                continue

            if all(lane["index"] == lane["stop"] for lane in lanes):
                break

            for lane in lanes:
                if lane["active_pid"] is not None or lane["index"] == lane["stop"]:
                    continue
                lane["index"] = first_incomplete(tasks, lane["index"], lane["stop"])
                if lane["index"] == lane["stop"] or available_gib() < MIN_START_GIB:
                    continue
                if len(children) >= 2:
                    break
                index = lane["index"]
                command = [
                    sys.executable, "-u", "-m", "scripts.rank_rank_microbatch",
                    "--mode", "once", "--start", str(index),
                    "--stop", str(lane["stop"]), "--max-jobs", str(MAX_JOBS),
                ]
                stdout = (OUTPUT / f"rank_parallel_lane{lane['lane']}.stdout.log").open("a", encoding="utf-8")
                stderr = (OUTPUT / f"rank_parallel_lane{lane['lane']}.stderr.log").open("a", encoding="utf-8")
                try:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
                except BaseException:
                    stdout.close()
                    stderr.close()
                    raise
                children[lane["lane"]] = (child, index, stdout, stderr)
                lane["active_pid"] = child.pid
                record("running")
                # Allow the first child to allocate its Qlib cache before admitting another.
                time.sleep(15)

            if not children and available_gib() < MIN_START_GIB:
                if time.time() - last_work > 1800:
                    raise MemoryError("Free RAM stayed below 3.6 GiB for 30 minutes")
            record("running")
            time.sleep(10)

        subprocess.run([sys.executable, "-m", "scripts.report_rank_rank_offsets"],
                       cwd=ROOT, check=True)
        report = OUTPUT / "rank_rank_phase_ranking.csv"
        if not report.exists():
            raise RuntimeError("Audited rank/rank report was not created")
        record("complete", report=str(report), finished=time.time())
    except BaseException as error:
        record("failed", error=repr(error), failed=time.time())
        raise
    finally:
        for child, _, stdout, stderr in children.values():
            child.wait()
            stdout.close()
            stderr.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("controller", "status"), required=True)
    parser.add_argument("--start", type=int, default=DEFAULT_LANE_BOUNDS[0][0])
    parser.add_argument("--split", type=int, default=DEFAULT_LANE_BOUNDS[0][1])
    args = parser.parse_args()
    if args.mode == "controller":
        controller(((args.start, args.split), (args.split, RANK_TASK_COUNT)))
    else:
        if STATUS.exists():
            print(STATUS.read_text(encoding="utf-8"))
        else:
            print("No parallel controller status yet")


if __name__ == "__main__":
    main()
