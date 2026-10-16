"""Reconcile an interrupted CN paper rebalance and buy its one missing holding next session.

Preparation writes local recovery state only.  Submission is explicitly paper-only,
date-bound and limited to the single missing stock; it never rebalances the nine fills.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time

from paper_futu.cn_common import (
    HISTORICAL_PROVIDER, STATE_DIR, china_now, load_deployment,
    target_share_quantity, validate_execution_session, validate_plan_deployment,
)
from paper_futu.execute_cn_plan import (
    account_context, fresh_quotes, paper_account_id, save_account_snapshot,
)
from paper_futu.smoke_order import TERMINAL, append_event, require_ok

CATCHUP_PLAN = STATE_DIR / "catchup_plan.json"


def next_trading_date(source_date: str) -> str:
    calendar = (HISTORICAL_PROVIDER / "calendars/day_future.txt").read_text().splitlines()
    return next(date for date in calendar if date > source_date)


def journal_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_original_fills(events, orders, positions, plan):
    starts = [event for event in events if event["event"] == "execution_start"]
    submitted = [event for event in events if event["event"] == "submitted"]
    if len(starts) != 1 or not submitted:
        raise ValueError("原调仓日志不完整，不能自动准备补仓")
    selected = set(starts[0]["selected"])
    if selected != set(plan["ordered_candidates"][:plan["top_k"]]):
        raise ValueError("原调仓选股与当前计划不一致")
    if len(selected) != plan["top_k"] or len({e["code"] for e in submitted}) != len(submitted):
        raise ValueError("原调仓代码重复或数量异常")
    order_ids = {str(event["order_id"]) for event in submitted}
    if len(order_ids) != len(submitted) or set(orders.order_id.astype(str)) != order_ids:
        raise ValueError("券商订单与原始日志不完全一致，须人工核对")
    by_id = orders.set_index(orders.order_id.astype(str))
    for event in submitted:
        row = by_id.loc[str(event["order_id"])]
        if (row.code != event["code"] or row.trd_side != "BUY"
                or row.order_status != "FILLED_ALL"
                or int(row.dealt_qty) != int(event["qty"])):
            raise ValueError("有订单未全额买成或订单信息不一致，须人工核对")
    held = set(positions.loc[positions.qty > 0, "code"])
    filled = {event["code"] for event in submitted}
    if held != filled or not filled.issubset(selected) or len(selected - filled) != 1:
        raise ValueError("持仓与已成交订单不一致，不能自动确定唯一漏单")
    missing = (selected - filled).pop()
    if any(event.get("code") == missing and event["event"] == "before_submit" for event in events):
        raise ValueError("漏单股票曾进入下单流程，须人工核对")
    action = next(a for a in starts[0]["actions"] if a["code"] == missing)
    if action["side"] != "BUY" or action["held"] != 0 or action["target"] <= 0:
        raise ValueError("漏单不是初始买入仓位")
    return missing, int(action["target"]), float(starts[0]["weights"][missing])


def prepare(trade, account_id, plan):
    source_date = plan["execution_date"]
    if china_now().date().isoformat() != source_date:
        raise ValueError("只能在原调仓当日核对并准备次日补仓")
    if (STATE_DIR / f"execution_{source_date}.json").exists() or CATCHUP_PLAN.exists():
        raise ValueError("已有调仓完成标记或补仓计划，不覆盖")
    journal = STATE_DIR / f"orders_{source_date}.jsonl"
    events = journal_events(journal)
    from futu import TrdEnv
    orders = require_ok(trade.order_list_query(start=source_date, trd_env=TrdEnv.SIMULATE,
                                               acc_id=account_id, refresh_cache=True), "核对模拟订单")
    positions = require_ok(trade.position_list_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                                                     refresh_cache=True), "核对模拟持仓")
    missing, max_qty, weight = validate_original_fills(events, orders, positions, plan)
    catchup_date = next_trading_date(source_date)
    recovery = {"source_execution_date": source_date, "execution_date": catchup_date,
                "code": missing, "max_qty": max_qty, "weight": weight,
                "model_sha256": plan["model_sha256"], "model_dir": plan["model_dir"],
                "environment": "SIMULATE", "account_suffix": "0000"}
    marker = {"execution_date": source_date, "status": "partial_verified",
              "fully_filled_orders": len(orders), "missing_code": missing,
              "catchup_execution_date": catchup_date, "verified_at": china_now().isoformat()}
    CATCHUP_PLAN.write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    (STATE_DIR / f"execution_{source_date}.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return recovery


def submit(trade, account_id, quote, plan, recovery):
    if recovery.get("status") == "cancelled":
        raise ValueError("单股模拟补仓已暂停；禁止提交订单")
    from futu import Currency, ModifyOrderOp, OrderType, TimeInForce, TrdEnv, TrdSide
    today = china_now().date().isoformat()
    source_date = recovery["source_execution_date"]
    if (today != recovery["execution_date"]
            or recovery["execution_date"] != next_trading_date(source_date)
            or recovery["model_sha256"] != plan["model_sha256"]
            or recovery["model_dir"] != plan["model_dir"]):
        raise ValueError("补仓日期或模型版本不匹配")
    minute = china_now().hour * 60 + china_now().minute
    if not 571 <= minute < 600:
        raise ValueError("补仓只允许次交易日09:31至10:00，过时不追买")
    marker = json.loads((STATE_DIR / f"execution_{source_date}.json").read_text(encoding="utf-8"))
    if marker.get("status") != "partial_verified" or marker.get("missing_code") != recovery["code"]:
        raise ValueError("原调仓尚未完成严格核对")
    original_events = journal_events(STATE_DIR / f"orders_{source_date}.jsonl")
    starts = [event for event in original_events if event["event"] == "execution_start"]
    if len(starts) != 1 or recovery["code"] not in starts[0]["selected"]:
        raise ValueError("补仓股票不在原始信号中")
    original_action = next((action for action in starts[0]["actions"] if action["code"] == recovery["code"]), None)
    if (original_action is None or original_action["side"] != "BUY"
            or original_action["held"] != 0 or int(original_action["target"]) != recovery["max_qty"]
            or starts[0]["weights"][recovery["code"]] != recovery["weight"]):
        raise ValueError("补仓股数或权重与原始调仓计划不符")
    if (STATE_DIR / f"catchup_complete_{today}.json").exists():
        print("补仓已完成，不重复下单", flush=True)
        return
    journal = STATE_DIR / f"catchup_orders_{today}.jsonl"
    if journal.exists() and any(e["event"] == "before_submit" for e in journal_events(journal)):
        raise ValueError("补仓已有提交记录，须人工核对；禁止自动重试")
    state = require_ok(quote.get_global_state(), "查询市场状态")
    validate_execution_session(today, recovery["execution_date"], plan["model_year"],
                               state["market_sh"], state["market_sz"])
    positions = require_ok(trade.position_list_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                                                     refresh_cache=True), "核对现有持仓")
    code = recovery["code"]
    if code in set(positions.loc[positions.qty > 0, "code"]):
        raise ValueError("漏单股票已在持仓中，不重复买入")
    today_orders = require_ok(trade.order_list_query(start=today, trd_env=TrdEnv.SIMULATE,
                                                     acc_id=account_id, refresh_cache=True), "核对今日订单")
    if not today_orders.empty:
        raise ValueError("今日账户已有订单，先人工核对再补仓")
    latest = fresh_quotes(quote, [code], execution=True).loc[code]
    if latest.suspension:
        raise ValueError("漏单股票停牌，不提交补仓订单")
    last_price = float(latest.last_price)
    upper = round(float(latest.prev_close_price) * 1.20, 2)
    if not math.isfinite(last_price) or last_price <= 0 or last_price >= upper - 0.005:
        raise ValueError("价格无效或触及涨停，不提交补仓订单")
    limit_price = min(upper, math.ceil(last_price * 1.003 * 100) / 100)
    funds = require_ok(trade.accinfo_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                                           refresh_cache=True, currency=Currency.CNH), "查询补仓资金")
    assets = funds.iloc[0]
    nav = float(assets.total_assets)
    available = max(0.0, float(assets.cash) - nav * (1 - plan["risk_degree"]) - 10)
    desired = target_share_quantity(code, recovery["weight"], nav, last_price)
    affordable = target_share_quantity(code, 1.0, max(available, 0.01), limit_price * 1.005)
    qty = min(int(recovery["max_qty"]), desired, affordable)
    qty = qty // 100 * 100
    if qty < 100:
        raise ValueError("资金不足一手且需保留5%现金，不提交补仓订单")
    remark = f"catchup{today.replace('-', '')}_{code.replace('.', '')}"
    append_event(journal, "before_submit", code=code, qty=qty, price=limit_price,
                 remark=remark, environment="SIMULATE", account_suffix="0000")
    result = trade.place_order(price=limit_price, qty=qty, code=code, trd_side=TrdSide.BUY,
                               order_type=OrderType.NORMAL, trd_env=TrdEnv.SIMULATE,
                               acc_id=account_id, time_in_force=TimeInForce.DAY, remark=remark)
    if result[0] != 0:
        append_event(journal, "rejected", response=str(result[1]))
        raise RuntimeError("补仓订单被拒绝；不能自动重试")
    order_id = str(result[1].iloc[0].order_id)
    append_event(journal, "submitted", code=code, qty=qty, order_id=order_id)
    time.sleep(5)
    order = require_ok(trade.order_list_query(order_id=order_id, trd_env=TrdEnv.SIMULATE,
                                              acc_id=account_id, refresh_cache=True), "复核补仓订单")
    if len(order) != 1:
        raise ValueError("不能唯一识别补仓订单，须人工核对")
    row = order.iloc[0]
    if row.order_status not in TERMINAL:
        cancelled = trade.modify_order(ModifyOrderOp.CANCEL, order_id, 0, 0,
                                       trd_env=TrdEnv.SIMULATE, acc_id=account_id)
        append_event(journal, "cancel_unfilled", order_id=order_id, status=cancelled[0])
        time.sleep(2.2)
        order = require_ok(trade.order_list_query(order_id=order_id, trd_env=TrdEnv.SIMULATE,
                                                  acc_id=account_id, refresh_cache=True), "确认补仓订单终态")
        if len(order) != 1:
            raise ValueError("不能确认补仓订单终态，须人工核对")
        row = order.iloc[0]
    append_event(journal, "order_result", code=code, order_id=order_id,
                 status=row.order_status, dealt_qty=float(row.dealt_qty))
    if row.order_status != "FILLED_ALL" or int(row.dealt_qty) != qty:
        raise ValueError("补仓未全部成交；保留记录，不自动二次下单")
    save_account_snapshot(trade, account_id)
    complete = {"execution_date": today, "source_execution_date": source_date,
                "code": code, "qty": qty, "order_id": order_id,
                "status": "filled_all", "completed_at": china_now().isoformat()}
    (STATE_DIR / f"catchup_complete_{today}.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    append_event(journal, "catchup_complete", **complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Verify the nine fills and write one-stock recovery state")
    parser.add_argument("--submit", action="store_true", help="Submit the single date-bound CN paper buy")
    args = parser.parse_args()
    if args.prepare == args.submit:
        parser.error("choose exactly one of --prepare or --submit")
    if args.prepare and CATCHUP_PLAN.exists():
        existing = json.loads(CATCHUP_PLAN.read_text(encoding="utf-8"))
        if existing.get("status") == "cancelled":
            raise ValueError("单股模拟补仓已暂停；不得用 --prepare 自动恢复原计划")
    plan = json.loads((STATE_DIR / "active_plan.json").read_text(encoding="utf-8"))
    model_dir, metadata = load_deployment()
    validate_plan_deployment(plan, model_dir, metadata)
    if args.submit:
        recovery = json.loads(CATCHUP_PLAN.read_text(encoding="utf-8"))
        if recovery.get("status") == "cancelled":
            raise ValueError("单股模拟补仓已暂停；禁止提交订单")
    with socket.create_connection(("127.0.0.1", 11111), timeout=3):
        pass
    trade = account_context()
    quote = None
    try:
        account_id = paper_account_id(trade)
        if args.prepare:
            recovery = prepare(trade, account_id, plan)
            print(f"已核对9笔成交；{recovery['execution_date']}仅补{recovery['code']}，最多{recovery['max_qty']}股", flush=True)
        else:
            recovery = json.loads(CATCHUP_PLAN.read_text(encoding="utf-8"))
            if recovery["environment"] != "SIMULATE" or recovery["account_suffix"] != "0000":
                raise ValueError("补仓账户或环境异常")
            from futu import OpenQuoteContext
            quote = OpenQuoteContext(host="127.0.0.1", port=11111)
            submit(trade, account_id, quote, plan, recovery)
    finally:
        trade.close()
        if quote is not None:
            quote.close()


if __name__ == "__main__":
    main()
