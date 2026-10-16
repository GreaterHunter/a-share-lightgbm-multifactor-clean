"""Submit exactly one small CN simulated order at the user-requested test time."""

import argparse
from datetime import datetime
import json
import math
import sys
import time

from paper_futu.cn_common import STATE_DIR, china_now
from paper_futu.execute_cn_plan import account_context, paper_account_id
from paper_futu.smoke_order import TERMINAL, append_event, order_record, require_ok

CODE = "SH.601669"
QUANTITY = 100
SCHEDULED = "2026-10-12T21:55:00+08:00"


def validate_time(now, scheduled):
    delay = (now - datetime.fromisoformat(scheduled)).total_seconds()
    if not 0 <= delay <= 180:
        raise ValueError("Test must start within three minutes of the authorized time")


def test_price(value):
    price = round(float(value), 2)
    if not math.isfinite(price) or price <= 0 or price * QUANTITY > 500:
        raise ValueError("Test buy must not exceed 500 CNY")
    return price


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--keep-overnight", action="store_true")
    parser.add_argument("--scheduled-for", default=SCHEDULED,
                        choices=[SCHEDULED, "2026-10-12T22:00:00+08:00"])
    args = parser.parse_args()
    test_key = datetime.fromisoformat(args.scheduled_for).strftime("%Y%m%d_%H%M")
    directory = STATE_DIR / f"overnight_probe_{test_key}"
    directory.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:
        sys.stdout = (directory / "stdout.log").open("a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = (directory / "stderr.log").open("a", encoding="utf-8", buffering=1)
    journal = directory / "orders.jsonl"
    if args.submit:
        validate_time(china_now(), args.scheduled_for)
        # Exclusive journal prevents resubmission after an uncertain network outcome.
        with journal.open("x", encoding="utf-8"):
            pass
    from futu import Currency, ModifyOrderOp, OpenQuoteContext, OrderType, TimeInForce, TrdEnv, TrdSide
    quote = OpenQuoteContext(host="127.0.0.1", port=11111)
    trade = None
    account_id = None
    order_id = None
    try:
        trade = account_context()
        account_id = paper_account_id(trade)
        plan = json.loads((STATE_DIR / "active_plan.json").read_text(encoding="utf-8"))
        if CODE not in plan["predictions"]:
            raise ValueError("Test symbol is not in the current strategy's candidate universe")
        snapshot = require_ok(quote.get_market_snapshot([CODE]), "Read test quote").iloc[0]
        if bool(snapshot.suspension) or str(snapshot.update_time)[:10] != china_now().date().isoformat():
            raise ValueError("Test quote is suspended or from a different date")
        price = test_price(snapshot.last_price)
        assets = require_ok(trade.accinfo_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                           refresh_cache=True, currency=Currency.CNH), "Read simulated cash").iloc[0]
        if float(assets.cash) < price * QUANTITY + 30:
            raise ValueError("Insufficient simulated cash")
        if not args.submit:
            print(f"PREVIEW ONLY: CN SIMULATE, {CODE}, {QUANTITY} shares, limit {price:.2f}, max 500 CNY", flush=True)
            return
        append_event(journal, "before_submit", code=CODE, qty=QUANTITY, price=price,
                     environment="SIMULATE", keep_overnight=args.keep_overnight, scheduled=args.scheduled_for)
        response = require_ok(trade.place_order(price=price, qty=QUANTITY, code=CODE,
            trd_side=TrdSide.BUY, order_type=OrderType.NORMAL, trd_env=TrdEnv.SIMULATE,
            acc_id=account_id, time_in_force=TimeInForce.DAY, remark=f"cn_probe_{test_key}"), "Submit single simulated probe")
        order_id = str(response.iloc[0].order_id)
        append_event(journal, "accepted", **order_record(response))
        time.sleep(3)
        frame = require_ok(trade.order_list_query(order_id=order_id, trd_env=TrdEnv.SIMULATE,
                           acc_id=account_id, refresh_cache=True), "Read probe status")
        details = order_record(frame)
        append_event(journal, "order_status", **details)
        if not args.keep_overnight and details["order_status"] not in TERMINAL:
            require_ok(trade.modify_order(ModifyOrderOp.CANCEL, order_id, 0, 0,
                       trd_env=TrdEnv.SIMULATE, acc_id=account_id), "Cancel probe")
            for _ in range(5):
                time.sleep(2)
                details = order_record(require_ok(trade.order_list_query(order_id=order_id,
                    trd_env=TrdEnv.SIMULATE, acc_id=account_id, refresh_cache=True), "Confirm cancellation"))
                if details["order_status"] in TERMINAL:
                    break
            if details["order_status"] not in TERMINAL:
                raise ValueError("Cancellation is not confirmed; check the simulated account")
        result = {"time": china_now().isoformat(), "environment": "SIMULATE", "market": "CN",
                  "account_suffix": "0000", "keep_overnight": args.keep_overnight, **details}
        (directory / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        append_event(journal, "result", **details)
        if details["order_status"] in {"FAILED", "DISABLED", "DELETED"}:
            raise ValueError("Probe was not accepted as a working order")
    except Exception as error:
        if args.submit and journal.exists():
            append_event(journal, "error", message=str(error), order_id=order_id)
        if order_id and trade and not args.keep_overnight:
            try:
                trade.modify_order(ModifyOrderOp.CANCEL, order_id, 0, 0,
                                   trd_env=TrdEnv.SIMULATE, acc_id=account_id)
            except Exception:
                pass
        raise
    finally:
        if trade:
            trade.close()
        quote.close()


if __name__ == "__main__":
    main()
