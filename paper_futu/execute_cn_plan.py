"""Execute a saved CSI300 plan on the next session, using only CN paper funds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import time

import pandas as pd

from paper_futu.check_connection import stock_paper_accounts
from paper_futu.cn_common import CHINA_TZ, STATE_DIR, china_now, load_deployment, select_holdings, select_audited_holdings, sellable_quantity, target_share_quantity, tiered_weights, validate_execution_session, validate_plan_deployment
from paper_futu.smoke_order import TERMINAL, append_event, require_ok


def account_context():
    from futu import OpenSecTradeContext, SecurityFirm, TrdMarket
    return OpenSecTradeContext(filter_trdmarket=TrdMarket.CN, host="127.0.0.1", port=11111,
                               security_firm=SecurityFirm.FUTUSECURITIES)


def paper_account_id(trade):
    paper = stock_paper_accounts(require_ok(trade.get_acc_list(), "读取沪深模拟账户"), "CN")
    if len(paper) != 1 or str(paper.iloc[0].acc_id)[-4:] != "0000":
        raise ValueError("沪深股票模拟账户不唯一或不是展示代码配置的演示尾号")
    return int(paper.iloc[0].acc_id)


def save_account_snapshot(trade, account_id):
    from futu import Currency, OpenQuoteContext, TrdEnv
    assets = require_ok(trade.accinfo_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                                           refresh_cache=True, currency=Currency.CNH), "查询模拟资产")
    positions = require_ok(trade.position_list_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                                                     refresh_cache=True), "查询模拟持仓")
    now = china_now()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = STATE_DIR / "account_baseline.json"
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps({"time": now.isoformat(), "initial_assets": float(assets.iloc[0].total_assets),
                                             "account_suffix": "0000"}, indent=2), encoding="utf-8")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    benchmark_context = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        benchmark = require_ok(benchmark_context.get_market_snapshot(["SH.000300"]), "查询沪深300基准")
        benchmark_price = float(benchmark.iloc[0].last_price)
    finally:
        benchmark_context.close()
    if "benchmark_price" not in baseline:
        baseline["benchmark_price"] = benchmark_price
        baseline["benchmark_start_time"] = now.isoformat()
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    row = {"time": now.isoformat(), "total_assets": float(assets.iloc[0].total_assets),
           "cash": float(assets.iloc[0].cash), "market_val": float(assets.iloc[0].market_val),
           "holdings": len(positions), "return_since_start": float(assets.iloc[0].total_assets) / baseline["initial_assets"] - 1,
           "benchmark_price": benchmark_price,
           "benchmark_return": benchmark_price / baseline["benchmark_price"] - 1,
           "geometric_excess_return": (float(assets.iloc[0].total_assets) / baseline["initial_assets"])
                                      / (benchmark_price / baseline["benchmark_price"]) - 1}
    path = STATE_DIR / "account_snapshots.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        header = rows[0]
        columns = list(row)
        extended_header = header + [name for name in columns if name not in header]
        earlier = pd.DataFrame([dict(zip(extended_header, values)) for values in rows[1:]])
        pd.concat([earlier, pd.DataFrame([row])], ignore_index=True).reindex(columns=columns).to_csv(path, index=False)
    else:
        pd.DataFrame([row]).to_csv(path, index=False)
    positions.to_csv(STATE_DIR / "positions_latest.csv", index=False)
    return assets.iloc[0], positions


def fresh_quotes(quote, codes, execution=False):
    parts = []
    for begin in range(0, len(codes), 100):
        parts.append(require_ok(quote.get_market_snapshot(codes[begin:begin + 100]), "查询交易行情"))
    frame = pd.concat(parts).set_index("code", drop=False)
    if set(frame.index) != set(codes):
        raise ValueError("交易行情缺少代码")
    if execution:
        now = china_now()
        for row in frame.itertuples():
            if not row.suspension:
                timestamp = pd.Timestamp(row.update_time).tz_localize(CHINA_TZ)
                if timestamp.date() != now.date() or abs((pd.Timestamp(now) - timestamp).total_seconds()) > 120:
                    raise ValueError(f"交易行情过期：{row.code}, {row.update_time}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plan = json.loads((STATE_DIR / "active_plan.json").read_text(encoding="utf-8"))
    model_dir, metadata = load_deployment()
    validate_plan_deployment(plan, model_dir, metadata)
    today = china_now().date().isoformat()
    execution_date = plan["execution_date"]
    journal = STATE_DIR / f"orders_{execution_date}.jsonl"
    complete_path = STATE_DIR / f"execution_{execution_date}.json"
    if args.submit and complete_path.exists():
        print("该调仓已完成，不重复提交", flush=True)
        return
    from futu import Currency, ModifyOrderOp, OpenQuoteContext, OrderType, TimeInForce, TrdEnv, TrdSide
    with socket.create_connection(("127.0.0.1", 11111), timeout=3):
        pass
    trade = account_context()
    quote = None
    submitted_ids = []
    account_id = None
    try:
        account_id = paper_account_id(trade)
        assets, positions = save_account_snapshot(trade, account_id)
        if (positions.qty < 0).any():
            raise ValueError("模拟账户存在空头，不符合本策略")
        quote = OpenQuoteContext(host="127.0.0.1", port=11111)
        state = require_ok(quote.get_global_state(), "查询市场状态")
        if args.submit:
            validate_execution_session(today, execution_date, metadata["model_year"], state["market_sh"], state["market_sz"])
        predicted = plan["predictions"]
        current = positions.loc[positions.qty > 0, "code"].tolist()
        quotes = fresh_quotes(quote, sorted(set(plan["ordered_candidates"] + current)), execution=args.submit)
        eligible = [code for code in plan["ordered_candidates"] if not bool(quotes.loc[code].suspension)]
        audited = metadata.get("selection_rule") == "audited_limited_replacement_v1"
        if audited:
            # Match the research's conservative 9.5% direction filter, in addition
            # to the board-specific legal price limits checked before submission.
            def tradable(code, buying):
                row = quotes.loc[code]
                previous = float(row.prev_close_price)
                if row.suspension or not math.isfinite(previous) or previous <= 0:
                    return False
                change = float(row.last_price) / previous - 1
                return change < .095 if buying else change > -.095
            eligible = [code for code in eligible if tradable(code, True)]
            sellable = [row.code for row in positions.itertuples() if row.qty > 0
                        and row.can_sell_qty >= row.qty and tradable(row.code, False)]
            selected = select_audited_holdings(predicted, eligible, current,
                top_k=plan["top_k"], n_drop=plan["n_drop"], sellable=sellable)
        else:
            selected = select_holdings(predicted, eligible, current, top_k=plan["top_k"], n_drop=plan["n_drop"])
        weights = tiered_weights(selected, risk_degree=plan["risk_degree"], top_k=plan["top_k"],
                                top_tier_size=plan["top_tier_size"], top_tier_weight=plan["top_tier_weight"])
        nav = float(assets.total_assets)
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError("模拟账户净资产异常，停止调仓")
        holdings = positions.set_index("code")
        targets = {code: target_share_quantity(code, weights[code], nav, float(quotes.loc[code].last_price)) for code in selected}
        actions = []
        for code in sorted(set(targets).union(current)):
            held = int(holdings.loc[code].qty) if code in holdings.index else 0
            desired = targets.get(code, 0)
            if desired != held:
                actions.append({"code": code, "side": "SELL" if desired < held else "BUY",
                                "qty": abs(desired - held), "held": held, "target": desired,
                                "weight": weights.get(code, 0), "reference_price": float(quotes.loc[code].last_price)})
        actions.sort(key=lambda action: (action["side"] != "SELL", action["code"]))
        preview = {"signal_date": plan["signal_date"], "execution_date": execution_date,
                   "environment": "SIMULATE", "market": "CN", "nav": nav,
                   "cash_reserve": nav * (1 - plan["risk_degree"]), "selected": selected, "weights": weights,
                   "model_dir": plan["model_dir"], "top_k": plan["top_k"], "n_drop": plan["n_drop"],
                   "actions": actions, "current_market_state": state["market_sh"]}
        (STATE_DIR / f"preview_{execution_date}.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
        print(f"CN SIMULATE: {len(selected)} selected, {len(actions)} position-difference orders; NAV {nav:.2f}; market {state['market_sh']}", flush=True)
        if not args.submit:
            print("Preview only; no orders submitted", flush=True)
            return
        prior_events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
        if any(event["event"] == "before_submit" for event in prior_events):
            raise ValueError("已有调仓提交日志；为防止重复下单，须先核对已提交订单")
        append_event(journal, "execution_start", **preview)
        today_orders = require_ok(trade.order_list_query(start=today, trd_env=TrdEnv.SIMULATE,
                                                         acc_id=account_id, refresh_cache=True), "查询当日模拟订单")
        today_buys = today_orders.loc[today_orders.trd_side.eq("BUY")].groupby("code").dealt_qty.sum().to_dict()
        # Fetch available cash once after sells have been submitted.  The broker
        # limits account-funds queries; reserve each accepted buy locally at its
        # limit price so ten buys cannot spend the same cash twice.
        available_buying_cash = None
        reserved_new_names = set()
        for action in actions:
            if (STATE_DIR / "stop_requested").exists():
                append_event(journal, "stop_requested", message="停止提交新订单，继续核对及撤销已提交的未成交订单")
                break
            code, side, quantity = action["code"], action["side"], action["qty"]
            latest = fresh_quotes(quote, [code], execution=True).loc[code]
            if latest.suspension:
                append_event(journal, "skipped", code=code, reason="suspension")
                continue
            previous = float(latest.prev_close_price)
            daily_limit = 0.20 if code.startswith(("SH.688", "SZ.300", "SZ.301")) else 0.10
            upper = round(previous * (1 + daily_limit), 2)
            lower = round(previous * (1 - daily_limit), 2)
            price = float(latest.last_price)
            if (side == "BUY" and price >= upper - 0.005) or (side == "SELL" and price <= lower + 0.005):
                append_event(journal, "skipped", code=code, reason="price_limit")
                continue
            if side == "SELL":
                position = require_ok(trade.position_list_query(code=code, trd_env=TrdEnv.SIMULATE,
                                                                 acc_id=account_id, refresh_cache=True), "查询可卖数量")
                if position.empty:
                    continue
                quantity = min(quantity, sellable_quantity(float(position.iloc[0].qty),
                                float(position.iloc[0].can_sell_qty), float(today_buys.get(code, 0))))
                limit_price = max(lower, math.floor(price * 0.997 * 100) / 100)
            else:
                if audited and code not in current:
                    actual_positions = require_ok(trade.position_list_query(trd_env=TrdEnv.SIMULATE,
                        acc_id=account_id, refresh_cache=True), "核验持仓数量上限")
                    actual_names = set(actual_positions.loc[actual_positions.qty > 0, "code"])
                    if code not in actual_names | reserved_new_names and len(actual_names | reserved_new_names) >= plan["top_k"]:
                        append_event(journal, "skipped", code=code, reason="holding_limit_pending_sell")
                        continue
                limit_price = min(upper, math.ceil(price * 1.003 * 100) / 100)
                if available_buying_cash is None:
                    funds = require_ok(trade.accinfo_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                                                           refresh_cache=True, currency=Currency.CNH), "查询买入现金")
                    available_buying_cash = max(
                        0.0, float(funds.iloc[0].cash) - nav * (1 - plan["risk_degree"]) - 10
                    )
                # Leave 0.5% of each order's notional for fees/price rounding.
                maximum = target_share_quantity(code, 1.0, max(available_buying_cash, 0.01),
                                                limit_price * 1.005)
                quantity = min(quantity, maximum)
                if not code.startswith("SH.688"):
                    quantity = quantity // 100 * 100
                elif quantity < 200:
                    quantity = 0
            if quantity <= 0:
                append_event(journal, "skipped", code=code, reason="lot_cash_or_t_plus_one")
                continue
            remark = f"cn{today.replace('-', '')}_{code.replace('.', '')}_{side}"
            append_event(journal, "before_submit", code=code, side=side, qty=quantity, price=limit_price,
                         remark=remark, environment="SIMULATE")
            result = trade.place_order(price=limit_price, qty=quantity, code=code,
                trd_side=TrdSide.BUY if side == "BUY" else TrdSide.SELL,
                order_type=OrderType.NORMAL, trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                time_in_force=TimeInForce.DAY, remark=remark)
            if result[0] != 0:
                append_event(journal, "rejected", code=code, response=str(result[1]))
            else:
                order_id = str(result[1].iloc[0].order_id)
                submitted_ids.append(order_id)
                append_event(journal, "submitted", code=code, side=side, qty=quantity, order_id=order_id)
                if side == "BUY":
                    available_buying_cash -= quantity * limit_price * 1.005
                    reserved_new_names.add(code)
            time.sleep(2.2)
        time.sleep(5)
        for order_id in submitted_ids:
            response = require_ok(trade.order_list_query(order_id=order_id, trd_env=TrdEnv.SIMULATE,
                                                          acc_id=account_id, refresh_cache=True), "复核模拟订单")
            if len(response) != 1:
                raise ValueError("Cannot uniquely identify submitted order")
            row = response.iloc[0]
            if row.order_status not in TERMINAL:
                cancelled = trade.modify_order(ModifyOrderOp.CANCEL, order_id, 0, 0,
                                                trd_env=TrdEnv.SIMULATE, acc_id=account_id)
                append_event(journal, "cancel_unfilled", order_id=order_id, status=cancelled[0], response=str(cancelled[1]))
                time.sleep(2.2)
        time.sleep(3)
        all_orders = require_ok(trade.order_list_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                                                       refresh_cache=True), "确认订单终态")
        ours = all_orders.loc[all_orders.order_id.astype(str).isin(submitted_ids)]
        if len(ours) != len(submitted_ids) or not ours.order_status.isin(TERMINAL).all():
            raise ValueError("仍有订单终态未确认，不能标记调仓完成")
        ours.to_csv(STATE_DIR / f"order_results_{execution_date}.csv", index=False)
        assets, positions = save_account_snapshot(trade, account_id)
        summary = {"execution_date": execution_date, "signal_date": plan["signal_date"],
                   "environment": "SIMULATE", "submitted_orders": len(submitted_ids),
                   "fully_filled_orders": int(ours.order_status.eq("FILLED_ALL").sum()),
                   "positions": len(positions), "total_assets": float(assets.total_assets),
                   "cash": float(assets.cash), "completed_at": china_now().isoformat()}
        if not submitted_ids and actions:
            raise ValueError("全部仓位差订单被跳过，不能声称调仓已执行")
        complete_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        append_event(journal, "execution_complete", **summary)
    except Exception as error:
        if args.submit and journal.exists():
            append_event(journal, "error", message=str(error))
        if trade is not None and account_id is not None:
            for order_id in submitted_ids:
                try:
                    result = trade.modify_order(ModifyOrderOp.CANCEL, order_id, 0, 0,
                                                 trd_env=TrdEnv.SIMULATE, acc_id=account_id)
                    if journal.exists():
                        append_event(journal, "recovery_cancel", order_id=order_id, status=result[0])
                    time.sleep(2.2)
                except Exception:
                    pass
        raise
    finally:
        trade.close()
        if quote is not None:
            quote.close()


if __name__ == "__main__":
    main()
