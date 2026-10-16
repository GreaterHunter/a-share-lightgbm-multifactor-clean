"""One-share SPY paper order; query the result and cancel any unfilled remainder."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

from paper_futu.check_connection import stock_paper_accounts

ROOT = Path(__file__).resolve().parents[1]
CODE = "US.SPY"
TERMINAL = {"FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED", "DISABLED", "DELETED"}


def require_ok(result, operation):
    status, data = result
    if status != 0:
        raise RuntimeError(f"{operation}失败：{data}")
    return data


def checked_price(price):
    price = float(price)
    if not math.isfinite(price) or not 0 < price <= 1000:
        raise ValueError("1 股测试订单限价必须大于 0 且不超过 1000 美元")
    return math.ceil(price * 100) / 100


def order_record(frame):
    if len(frame) != 1:
        raise RuntimeError("无法唯一识别测试订单，请在牛牛模拟账户检查订单")
    return frame.iloc[0][[
        name for name in ["code", "order_id", "trd_side", "order_status", "qty", "price", "dealt_qty", "dealt_avg_price", "last_err_msg"]
        if name in frame.columns
    ]].to_dict()


def append_event(journal, event, **values):
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **values}
    with journal.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Submit exactly one paper buy order")
    parser.add_argument("--fill-test", action="store_true", help="Use a 0.1% price cushion for a separate one-share fill test")
    args = parser.parse_args()
    date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    suffix = "_fill" if args.fill_test else ""
    journal = ROOT / "data/interim/futu_paper" / f"smoke_order_{date}{suffix}.jsonl"
    if args.submit and journal.exists():
        raise RuntimeError(f"本日测试日志已存在，不重复下单，请先检查：{journal}")

    from futu import (
        Currency, ModifyOrderOp, OpenQuoteContext, OpenSecTradeContext,
        OrderType, SecurityFirm, Session, TimeInForce, TrdEnv, TrdMarket, TrdSide,
    )

    quote = OpenQuoteContext(host="127.0.0.1", port=11111)
    trade = None
    order_id = None
    try:
        trade = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US, host="127.0.0.1", port=11111,
            security_firm=SecurityFirm.FUTUSECURITIES,
        )
        accounts = stock_paper_accounts(require_ok(trade.get_acc_list(), "查询账户"), "US")
        if len(accounts) != 1:
            raise RuntimeError("需要唯一的活跃美股股票模拟账户，不能自动猜测账户")
        account_id = int(accounts.iloc[0]["acc_id"])
        state = require_ok(quote.get_global_state(), "查询市场状态")
        snapshot = require_ok(quote.get_market_snapshot([CODE]), "查询 SPY 行情").iloc[0]
        if bool(snapshot["suspension"]):
            raise RuntimeError("SPY 停牌，停止测试")
        market = state["market_us"]
        sessions = {"MORNING": Session.RTH, "AFTERNOON": Session.RTH, "PRE_MARKET_BEGIN": Session.ETH, "AFTER_HOURS_BEGIN": Session.ETH}
        if market not in sessions:
            raise RuntimeError(f"当前市场状态 {market} 不适合即时交易测试")
        price_field = {"PRE_MARKET_BEGIN": "pre_price", "AFTER_HOURS_BEGIN": "after_price"}.get(market, "last_price")
        limit_price = checked_price(float(snapshot[price_field]) * (1.001 if args.fill_test else 1.0))
        assets = require_ok(trade.accinfo_query(
            trd_env=TrdEnv.SIMULATE, acc_id=account_id, currency=Currency.USD,
        ), "查询模拟资金")
        if assets.empty or float(assets.iloc[0]["cash"]) < limit_price + 10:
            raise RuntimeError("模拟现金不足，停止测试")
        positions_before = require_ok(trade.position_list_query(
            code=CODE, trd_env=TrdEnv.SIMULATE, acc_id=account_id,
        ), "查询初始持仓")
        initial_qty = float(positions_before["qty"].sum()) if not positions_before.empty else 0.0
        print(f"环境 SIMULATE；账户 ***{str(account_id)[-4:]}；市场 {market}；买入 {CODE} 1 股；限价 {limit_price:.2f} USD；初始持仓 {initial_qty}", flush=True)
        if not args.submit:
            print("预检查通过，没有提交订单。")
            return
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("x", encoding="utf-8"):
            pass
        append_event(journal, "before_submit", code=CODE, qty=1, limit_price=limit_price,
                     environment="SIMULATE", market_state=market, initial_qty=initial_qty,
                     account_suffix=str(account_id)[-4:], quote_time=str(snapshot["update_time"]))
        response = require_ok(trade.place_order(
            price=limit_price, qty=1, code=CODE, trd_side=TrdSide.BUY,
            order_type=OrderType.NORMAL, trd_env=TrdEnv.SIMULATE, acc_id=account_id,
            time_in_force=TimeInForce.DAY, session=sessions[market],
            remark=f"codex_smoke_{date}{suffix}",
        ), "提交模拟订单")
        order_id = str(response.iloc[0]["order_id"])
        append_event(journal, "submit_response", **order_record(response))
        current = None
        for _ in range(5):
            time.sleep(2)
            current = require_ok(trade.order_list_query(
                order_id=order_id, trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                refresh_cache=True,
            ), "查询订单状态")
            details = order_record(current)
            append_event(journal, "order_status", **details)
            if details["order_status"] in TERMINAL:
                break
        if current is not None and order_record(current)["order_status"] not in TERMINAL:
            result = trade.modify_order(
                ModifyOrderOp.CANCEL, order_id, 0, 0,
                trd_env=TrdEnv.SIMULATE, acc_id=account_id,
            )
            append_event(journal, "cancel_response", status=result[0], response=str(result[1]))
            for _ in range(5):
                time.sleep(2)
                current = require_ok(trade.order_list_query(
                    order_id=order_id, trd_env=TrdEnv.SIMULATE, acc_id=account_id, refresh_cache=True,
                ), "查询撤单结果")
                details = order_record(current)
                append_event(journal, "final_order_status", **details)
                if details["order_status"] in TERMINAL:
                    break
        positions_after = require_ok(trade.position_list_query(
            code=CODE, trd_env=TrdEnv.SIMULATE, acc_id=account_id, refresh_cache=True,
        ), "查询最终持仓")
        final_qty = float(positions_after["qty"].sum()) if not positions_after.empty else 0.0
        details = order_record(current)
        append_event(journal, "result", final_order_status=details["order_status"],
                     dealt_qty=details["dealt_qty"], dealt_avg_price=details["dealt_avg_price"],
                     initial_qty=initial_qty, final_qty=final_qty)
        if details["order_status"] not in TERMINAL:
            raise RuntimeError("订单终态尚未确认，请立即在牛牛模拟订单页检查")
        if details["order_status"] in {"FAILED", "DISABLED", "DELETED"}:
            raise RuntimeError(f"订单未正常执行：{details}")
        print(f"测试日志：{journal}", flush=True)
    except Exception as error:
        if args.submit and journal.exists():
            append_event(journal, "error", message=str(error), order_id=order_id)
        if order_id is not None and trade is not None:
            try:
                recovery = require_ok(trade.order_list_query(
                    order_id=order_id, trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                    refresh_cache=True,
                ), "异常后查询订单")
                if order_record(recovery)["order_status"] not in TERMINAL:
                    cancelled = trade.modify_order(
                        ModifyOrderOp.CANCEL, order_id, 0, 0,
                        trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                    )
                    append_event(journal, "recovery_cancel", status=cancelled[0], response=str(cancelled[1]))
            except Exception as cleanup_error:
                print(f"无法确认撤单，请在模拟账户检查订单 {order_id}：{cleanup_error}", flush=True)
        raise
    finally:
        if trade is not None:
            trade.close()
        quote.close()


if __name__ == "__main__":
    main()
