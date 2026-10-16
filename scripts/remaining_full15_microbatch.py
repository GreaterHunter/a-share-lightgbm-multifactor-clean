"""Resume the remaining full-15-factor phase replays in bounded worker processes.

Only the already prepared manifest is used. The rank/rank prefix is never replayed,
and the frozen source study and trading deployment are not modified.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import psutil

from paper_futu import final_ablation_core as core
from scripts.compare_all_full15_offsets import OUTPUT, SOURCE, isolate_output
from scripts.rank_rank_microbatch import ROOT, RANK_TASK_COUNT, checked_manifest, task_complete, write_json
from src.runtime_budget import require_memory


STATUS = OUTPUT / "remaining_full15_controller.json"
MIN_START_GIB = 3.6
MAX_JOBS = 10


def available_gib() -> float:
    return psutil.virtual_memory().available / 2**30


def run_once(start: int, stop: int, maximum: int) -> None:
    manifest = checked_manifest()
    tasks = manifest["tasks"]
    if not (RANK_TASK_COUNT <= start < stop <= len(tasks) and 1 <= maximum <= MAX_JOBS):
        raise ValueError("Invalid remaining-task batch")
    isolate_output()
    from paper_futu.final_ablation_account import replay

    index = start
    performed = 0
    while index < stop and performed < maximum:
        job = tasks[index]
        if task_complete(job["case_id"]):
            index += 1
            continue
        if performed and available_gib() < 2.2:
            break
        require_memory(2.0, "完整15因子剩余模型调仓起点回测")
        result = replay(job["prediction_path"], job["portfolio"], job["rebalance"],
                        job["offset"], 0.0)
        if result["rows"][0]["case_id"] != job["case_id"]:
            raise ValueError("Replayed account case ID mismatch")
        performed += 1
        index += 1
    write_json(OUTPUT / f"remaining_micro_{start:04d}.json", {
        "start": start, "stop": stop, "next_index": index,
        "new_accounts": performed, "pid": os.getpid(),
        "finished": time.time(), "available_gib": available_gib(),
    })
    print(f"remaining micro start={start} next={index} new={performed}", flush=True)


def active_research_processes() -> list[int]:
    modules = {
        "scripts.compare_all_full15_offsets",
        "scripts.rank_rank_microbatch",
        "scripts.rank_rank_parallel_microbatch",
        "scripts.remaining_full15_microbatch",
        "paper_futu.run_final_ablation",
    }
    result = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        command = process.info["cmdline"] or []
        if any(command[i] == "-m" and command[i + 1] in modules
               for i in range(len(command) - 1)):
            result.append(process.pid)
    return result


def first_incomplete(tasks: list[dict], start: int, stop: int) -> int:
    return next((index for index in range(start, stop)
                 if not task_complete(tasks[index]["case_id"])), stop)


def controller() -> None:
    manifest = checked_manifest()
    if active_research_processes():
        raise RuntimeError(f"Another research process is active: {active_research_processes()}")
    tasks = manifest["tasks"]
    split = RANK_TASK_COUNT + (len(tasks) - RANK_TASK_COUNT) // 2
    bounds = ((RANK_TASK_COUNT, split), (split, len(tasks)))
    lanes = [{"lane": i, "index": first_incomplete(tasks, start, stop),
              "stop": stop, "batches": 0, "active_pid": None}
             for i, (start, stop) in enumerate(bounds)]
    children: dict[int, tuple[subprocess.Popen, int, object, object]] = {}
    started = time.time()
    last_work = started
    failure: str | None = None

    def record(status: str, **extra: object) -> None:
        write_json(STATUS, {
            "status": status, "controller_pid": os.getpid(),
            "started": started, "updated": time.time(),
            "available_gib": available_gib(), "max_workers": 2,
            "max_jobs_per_child": MAX_JOBS, "lane_bounds": bounds,
            "remaining_total": len(tasks) - RANK_TASK_COUNT,
            "remaining_complete": sum(task_complete(job["case_id"])
                                      for job in tasks[RANK_TASK_COUNT:]),
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
                result = core.read_json(OUTPUT / f"remaining_micro_{start_index:04d}.json")
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
                command = [sys.executable, "-u", "-m", "scripts.remaining_full15_microbatch",
                           "--mode", "once", "--start", str(index), "--stop",
                           str(lane["stop"]), "--max-jobs", str(MAX_JOBS)]
                stdout = (OUTPUT / f"remaining_lane{lane['lane']}.stdout.log").open("a", encoding="utf-8")
                stderr = (OUTPUT / f"remaining_lane{lane['lane']}.stderr.log").open("a", encoding="utf-8")
                try:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
                except BaseException:
                    stdout.close()
                    stderr.close()
                    raise
                children[lane["lane"]] = (child, index, stdout, stderr)
                lane["active_pid"] = child.pid
                record("running")
                time.sleep(15)

            if not children and available_gib() < MIN_START_GIB:
                if time.time() - last_work > 1800:
                    raise MemoryError("Free RAM stayed below 3.6 GiB for 30 minutes")
            record("running")
            time.sleep(10)

        subprocess.run([sys.executable, "-m", "scripts.audit_all_full15_offsets"],
                       cwd=ROOT, check=True)
        report = OUTPUT / "all_full15_phase_ranking.csv"
        if not report.exists():
            raise RuntimeError("Audited full-15 report was not created")
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
    parser.add_argument("--mode", required=True, choices=("once", "controller", "status"))
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--max-jobs", type=int, default=MAX_JOBS)
    args = parser.parse_args()
    if args.mode == "once":
        if args.start is None or args.stop is None:
            parser.error("--start and --stop are required with --mode once")
        run_once(args.start, args.stop, args.max_jobs)
    elif args.mode == "controller":
        controller()
    else:
        if STATUS.exists():
            print(STATUS.read_text(encoding="utf-8"))
        else:
            print("No remaining-model controller status yet")


if __name__ == "__main__":
    main()
