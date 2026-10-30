"""Independent closing data; OpenD is required only for account/order operations."""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta

from paper_futu.cn_common import HISTORICAL_PROVIDER, ROOT, STATE_DIR, china_now, load_deployment, validate_plan_deployment
from paper_futu.cn_runtime_protocol import verify_active_protocol

SDK_PYTHON = ROOT / "paper_futu/.venv/Scripts/python.exe"
MODEL_PYTHON = "D:/Anaconda/envs/quant-lightgbm/python.exe"
SESSION_START = 9 * 60 + 20
SESSION_DEADLINE = 16 * 60 + 30


def run(module, *args, model=False):
    timeout = 1800 if module == "paper_futu.update_cn_data" else 1500 if module in {
        "paper_futu.execute_cn_plan", "paper_futu.resume_cn_rebalance", "paper_futu.correct_cn_position"} else 900
    subprocess.run([MODEL_PYTHON if model else str(SDK_PYTHON), "-m", module, *args],
                   cwd=ROOT, check=True, timeout=timeout,
                   stdout=sys.stdout, stderr=sys.stderr,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def trading_dates():
    return set((HISTORICAL_PROVIDER / "calendars/day_future.txt").read_text().splitlines())


def session_exit_status(now):
    """Return an exit reason before any broker check; None means poll every 30s."""
    today = now.date().isoformat()
    minute = now.hour * 60 + now.minute
    if today not in trading_dates():
        return "non_trading_day"
    if minute < SESSION_START:
        return "outside_session"
    if minute >= 935 and (STATE_DIR / today / "plan.json").exists() and (STATE_DIR / f"nav_saved_{today}.json").exists():
        return "session_completed"
    if minute >= SESSION_DEADLINE:
        return "session_deadline_error"
    return None


def tick():
    now = china_now()
    today = now.date().isoformat()
    plan = json.loads((STATE_DIR / "active_plan.json").read_text(encoding="utf-8"))
    model_dir, metadata = load_deployment()
    validate_plan_deployment(plan, model_dir, metadata)
    status = {"pid": os.getpid(), "time": now.isoformat(), "mode": "CN_SIMULATE_ONLY",
              "signal_date": plan["signal_date"], "execution_date": plan["execution_date"],
              "next_rebalance_signal_date": plan["next_rebalance_signal_date"], "stage": "waiting",
              "top_k": plan["top_k"], "n_drop": plan["n_drop"], "model_dir": plan["model_dir"]}
    if now.year != plan["model_year"]:
        raise RuntimeError("模型年份已过期，需年度重训后再启动")
    if today not in trading_dates():
        status["stage"] = "non_trading_day"
        return status
    minute = now.hour * 60 + now.minute
    # This branch must precede every broker check. The separate scheduled data
    # job calls the same locked entry point, so OpenD can be closed all evening.
    if 935 <= minute < 1380 and not (STATE_DIR / today / "independent_data_ready.json").exists():
        retry_path=STATE_DIR/'worker_data_retry.json'
        retry=json.loads(retry_path.read_text(encoding='utf-8')) if retry_path.exists() else {}
        attempted=retry.get('last_attempt')
        from datetime import datetime
        if not attempted or (now-datetime.fromisoformat(attempted)).total_seconds()>=600:
            retry_path.write_text(json.dumps(dict(last_attempt=now.isoformat())),encoding='utf-8')
            try:
                run("paper_futu.update_cn_data", "--date", today, model=True)
                activated = json.loads((STATE_DIR / "active_plan.json").read_text(encoding="utf-8"))
                model_dir, metadata = load_deployment()
                validate_plan_deployment(activated, model_dir, metadata)
                plan = activated
                status.update({name: plan[name] for name in
                    ["signal_date", "execution_date", "next_rebalance_signal_date", "top_k", "n_drop", "model_dir"]})
                status["data_stage"] = "ready" if (STATE_DIR / today / "plan.json").exists() else "update_in_progress"
            except Exception as error:
                status.update(data_stage='error',data_error=str(error))
                # Continue to an independent account observation; no stale data is published.
        else:status['data_stage']='retry_backoff'
    execution_done = STATE_DIR / f"execution_{plan['execution_date']}.json"
    failure_path = STATE_DIR / f"execution_error_{plan['execution_date']}.json"
    market_window = False
    if today == plan["execution_date"] and not execution_done.exists():
        # Check the exact active code/model seal before the opening window, not
        # only inside a child that fails after 09:31. No broker/API calls here.
        try:
            protocol = verify_active_protocol()
            if protocol["strategy"]["model_sha256"] != plan["model_sha256"]:
                raise ValueError("Plan model differs from active forward protocol")
            status["protocol_version"] = protocol["version"]
        except Exception as error:
            status.update(stage="execution_protocol_blocked", message=str(error))
            if not failure_path.exists():
                failure_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            return status
        from paper_futu.cn_execution_session import normal_bounds
        start, end = normal_bounds(plan, now)
        if now >= end - timedelta(seconds=30):
            status.update(stage="execution_window_missed", message="No completion receipt; explicit recovery authorization required",
                          execution_window_end=end.isoformat())
            if failure_path.exists():
                status["last_execution_error"] = json.loads(failure_path.read_text(encoding="utf-8"))
            # Keep independent closing/account work; never chase expired orders.
        elif failure_path.exists():
            status.update(stage="execution_failed_requires_review",
                          last_execution_error=json.loads(failure_path.read_text(encoding="utf-8")))
            return status
        else:
            market_window = start <= now < end - timedelta(seconds=30)
    try:
        with socket.create_connection(("127.0.0.1", 11111), timeout=3):
            pass
    except OSError:
        status.update(stage="broker_offline", message="Public daily data is independent; account NAV/orders await OpenD")
        return status
    from paper_futu.execution_override import execution_allowed
    if today == plan["execution_date"] and not execution_allowed(plan, now) and not (STATE_DIR / f"execution_{today}.json").exists():
        market_window = False
        status["stage"] = "waiting_for_authorized_one_time_window"
    if today == plan["execution_date"] and market_window and not execution_done.exists():
        journal = STATE_DIR / f"orders_{today}.jsonl"
        journal_v5 = STATE_DIR / f"orders_v5_{today}.jsonl"
        if journal_v5.exists() or (journal.exists() and any(
            json.loads(line).get("event") == "before_submit"
            for line in journal.read_text(encoding="utf-8").splitlines()
        )):
            status["stage"] = "order_reconciliation_required"
        else:
            status["stage"] = "executing_paper_rebalance"
            (STATE_DIR / "worker_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            try:
                run("paper_futu.execute_cn_plan", "--submit")
            except Exception as error:
                status.update(stage="execution_failed_requires_review", message=str(error))
                failure_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                return status
            if execution_done.exists():
                receipt = json.loads(execution_done.read_text(encoding="utf-8"))
                status["stage"] = "rebalance_partial_constrained" if receipt.get("status") == "partial_constrained" else "rebalance_completed"
            else:
                status.update(stage="execution_receipt_missing", message="Executor returned without a completion receipt")
                failure_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    catchup_path = STATE_DIR / "catchup_plan.json"
    if catchup_path.exists():
        catchup = json.loads(catchup_path.read_text(encoding="utf-8"))
        catchup_window = 571 <= minute < 600  # next session 09:31-10:00 only
        if today == catchup["execution_date"] and catchup.get("status") == "cancelled":
            status["stage"] = "single_stock_catchup_cancelled"
        elif today == catchup["execution_date"] and catchup_window and not (STATE_DIR / f"catchup_complete_{today}.json").exists():
            journal = STATE_DIR / f"catchup_orders_{today}.jsonl"
            if journal.exists() and any(
                json.loads(line).get("event") == "before_submit"
                for line in journal.read_text(encoding="utf-8").splitlines()
            ):
                status["stage"] = "catchup_reconciliation_required"
            else:
                status["stage"] = "executing_single_stock_catchup"
                (STATE_DIR / "worker_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                run("paper_futu.cn_catchup", "--submit")
                status["stage"] = "catchup_completed"
        elif today == catchup["execution_date"] and minute >= 600 and not (STATE_DIR / f"catchup_complete_{today}.json").exists():
            status["stage"] = "catchup_window_missed_or_requires_review"
    if today > plan["execution_date"] and not execution_done.exists():
        raise RuntimeError("已错过计划交易日，不使用过期信号追买；请生成新的交易计划")
    # Wait until 15:35: post-close fixed-price trades can still change daily
    # volume through 15:30, even though the closing price itself is fixed.
    if 935 <= minute < 1380:
        directory = STATE_DIR / today
        # One independently saved closing account observation per session.
        marker = STATE_DIR / f"nav_saved_{today}.json"
        if not marker.exists():
            from paper_futu.record_cn_account import observe
            observed=observe()
            status['account_stage']=observed['stage']
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run a single readiness/scheduling cycle")
    parser.add_argument("--stop", action="store_true", help="Request the running worker to stop; does not sell holdings")
    args = parser.parse_args()
    from paper_futu.cn_rank80_runtime import enabled
    if enabled() and not args.stop:
        from paper_futu.cn_rank80_runtime import main as rank_main
        return rank_main(["--once"] if args.once else [])
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # pythonw hides the window; keep parent and child diagnostics in local log files.
    if sys.stdout is None:
        sys.stdout = (STATE_DIR / "worker_scheduled_stdout.log").open("a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = (STATE_DIR / "worker_scheduled_stderr.log").open("a", encoding="utf-8", buffering=1)
    stop_path = STATE_DIR / "stop_requested"
    if args.stop:
        stop_path.touch()
        print("已请求停止自动提交订单；不会卖出当前持仓", flush=True)
        return
    if stop_path.exists():
        raise RuntimeError("已有停止请求；人工确认恢复后再启动")
    lock_path = STATE_DIR / "worker.lock"
    with lock_path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("已有模拟交易工作进程运行，不重复启动")
        print(f"CSI300 CN paper worker started; PID {os.getpid()}", flush=True)
        while not stop_path.exists():
            if not args.once:
                now = china_now()
                exit_stage = session_exit_status(now)
                if exit_stage:
                    status = {"pid": os.getpid(), "time": now.isoformat(), "stage": exit_stage,
                              "mode": "CN_SIMULATE_ONLY", "schedule": "09:20 daily launch; 30-second in-session checks"}
                    (STATE_DIR / "worker_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                    print(json.dumps(status, ensure_ascii=False), flush=True)
                    if exit_stage == "session_deadline_error":
                        raise RuntimeError("收盘更新未完成，已到16:30；停止本日检查，请查看错误日志")
                    return
            try:
                status = tick()
            except Exception as error:
                status = {"pid": os.getpid(), "time": china_now().isoformat(), "stage": "error",
                          "mode": "CN_SIMULATE_ONLY", "message": str(error)}
                print(json.dumps(status, ensure_ascii=False), flush=True)
                if args.once:
                    (STATE_DIR / "worker_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                    raise
            (STATE_DIR / "worker_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            if args.once:
                print(json.dumps(status, ensure_ascii=False), flush=True)
                return
            # Leave as soon as closing feature and NAV persistence has succeeded.
            if session_exit_status(china_now()) == "session_completed":
                status["stage"] = "session_completed"
                status["time"] = china_now().isoformat()
                (STATE_DIR / "worker_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                return
            time.sleep(30)
        (STATE_DIR / "worker_status.json").write_text(json.dumps({"stage": "stopped", "time": china_now().isoformat()}), encoding="utf-8")


if __name__ == "__main__":
    main()
