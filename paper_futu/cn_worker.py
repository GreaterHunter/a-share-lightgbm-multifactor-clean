"""Run the authorized CSI300 paper strategy while this PC and OpenD stay online."""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import socket
import subprocess
import sys
import time

from paper_futu.cn_common import HISTORICAL_PROVIDER, ROOT, STATE_DIR, china_now, load_deployment, validate_plan_deployment

SDK_PYTHON = ROOT / "paper_futu/.venv/Scripts/python.exe"
MODEL_PYTHON = "D:/Anaconda/envs/quant-lightgbm/python.exe"
SESSION_START = 9 * 60 + 20
SESSION_DEADLINE = 16 * 60 + 30


def run(module, *args, model=False):
    subprocess.run([MODEL_PYTHON if model else str(SDK_PYTHON), "-m", module, *args],
                   cwd=ROOT, check=True, timeout=900, stdout=sys.stdout, stderr=sys.stderr,
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
    with socket.create_connection(("127.0.0.1", 11111), timeout=3):
        pass
    minute = now.hour * 60 + now.minute
    market_window = 571 <= minute < 690 or 780 <= minute < 895
    execution_done = STATE_DIR / f"execution_{plan['execution_date']}.json"
    if today == plan["execution_date"] and market_window and not execution_done.exists():
        journal = STATE_DIR / f"orders_{today}.jsonl"
        if journal.exists() and any(
            json.loads(line).get("event") == "before_submit"
            for line in journal.read_text(encoding="utf-8").splitlines()
        ):
            status["stage"] = "order_reconciliation_required"
        else:
            status["stage"] = "executing_paper_rebalance"
            (STATE_DIR / "worker_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            run("paper_futu.execute_cn_plan", "--submit")
            status["stage"] = "rebalance_completed"
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
        if not (directory / "plan.json").exists():
            run("paper_futu.fetch_cn_members", "--date", today, model=True)
            run("paper_futu.fetch_cn_market", "--date", today)
            options = [] if today >= plan["next_rebalance_signal_date"] else ["--daily-only"]
            run("paper_futu.build_cn_signal", "--date", today, *options, model=True)
            if not options:
                activated = json.loads((STATE_DIR / "active_plan.json").read_text(encoding="utf-8"))
                status.update({name:activated[name] for name in
                    ["signal_date","execution_date","next_rebalance_signal_date","top_k","n_drop","model_dir"]})
            status["stage"] = "closing_features_updated"
        # One independently saved closing account observation per session.
        marker = STATE_DIR / f"nav_saved_{today}.json"
        if not marker.exists():
            from paper_futu.execute_cn_plan import account_context, paper_account_id, save_account_snapshot
            trade = account_context()
            try:
                account_id = paper_account_id(trade)
                assets, positions = save_account_snapshot(trade, account_id)
                marker.write_text(json.dumps({"time": china_now().isoformat(), "assets": float(assets.total_assets),
                                               "cash": float(assets.cash), "holdings": len(positions)}, indent=2), encoding="utf-8")
            finally:
                trade.close()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run a single readiness/scheduling cycle")
    parser.add_argument("--stop", action="store_true", help="Request the running worker to stop; does not sell holdings")
    args = parser.parse_args()
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
