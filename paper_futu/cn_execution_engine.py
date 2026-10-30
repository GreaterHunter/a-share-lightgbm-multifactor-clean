"""One account, frozen targets, durable intents, bounded cancel/reprice.

The engine has no broker imports; deterministic fake brokers exercise all races.
It submits at most ONE order at a time (stricter than the five-order ceiling).
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

from paper_futu.cn_integer_targets import affordable_quantity, legal_order_quantity, board
from paper_futu.smoke_order import append_event

TERMINAL = {"FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED", "SUBMIT_FAILED", "DISABLED", "DELETED"}
MAX_ATTEMPTS = 3
ADVERSE_CAP = .005


def buy_cash_bound(quantity,price):
    value = quantity*price
    return value+max(5.,value*.005) if quantity else 0.


def affordable_buy_quantity(code,cash,price):
    quantity = affordable_quantity(code,cash,price)
    while quantity and buy_cash_bound(quantity,price) > cash+1e-8:
        if board(code) == "star":
            quantity = quantity-1 if quantity > 200 else 0
        else:
            quantity -= 100
    return quantity


class ReconciliationRequired(RuntimeError):
    pass


class ReceiptNotVisible(ReconciliationRequired):
    pass


class SubmissionNotSent(RuntimeError):
    """Definite local stop BEFORE invoking the broker's order API."""


def checked_quote(code, quote, now):
    stamp = datetime.fromisoformat(str(quote["update_time"]))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=now.tzinfo)
    if stamp.date() != now.date() or abs((now-stamp).total_seconds()) > 120:
        raise ValueError("stale_quote")
    price, previous = float(quote["last_price"]), float(quote["prev_close_price"])
    if not all(math.isfinite(p) and p > 0 for p in (price, previous)) or bool(quote.get("suspension")):
        raise ValueError("suspension_or_invalid_quote")
    board(code)
    if "ST" in str(quote.get("name", "")).upper():
        raise ValueError("unsupported_special_price_limit")
    tick = float(quote.get("price_spread", .01))
    if not math.isfinite(tick) or tick <= 0:
        raise ValueError("unknown_price_tick")
    if board(code) == "regular" and float(quote.get("lot_size", 100)) != 100:
        raise ValueError("unknown_order_unit")
    return price, previous, tick


def limit_price(code, side, quote, reference, attempt, now):
    price, previous, tick = checked_quote(code, quote, now)
    if not math.isfinite(reference) or reference <= 0 or attempt not in (1,2,3):
        raise ValueError("Invalid frozen reference or attempt")
    buying = side == "BUY"
    change = price/previous-1
    if (buying and change >= .095) or (not buying and change <= -.095):
        raise ValueError("conservative_direction_limit")
    daily = .2 if code.startswith(("SH.688", "SZ.300", "SZ.301")) else .1
    upper, lower = round(previous*(1+daily),2), round(previous*(1-daily),2)
    cap_high = math.floor((min(upper,reference*(1+ADVERSE_CAP))+1e-9)/tick)*tick
    cap_low = math.ceil((max(lower,reference*(1-ADVERSE_CAP))-1e-9)/tick)*tick
    try:
        opposite = float(quote.get("ask_price" if buying else "bid_price", float("nan")))
    except (ValueError,TypeError):
        opposite = float("nan")
    book = math.isfinite(opposite) and opposite > 0
    proposed = opposite + (attempt-1)*tick*(1 if buying else -1) if book else price*(1+(.001*attempt)*(1 if buying else -1))
    if buying:
        value = math.ceil((proposed-1e-9)/tick)*tick
        if value > cap_high+1e-8:
            raise ValueError("outside_frozen_price_cap")
    else:
        value = math.floor((proposed+1e-9)/tick)*tick
        if value < cap_low-1e-8:
            raise ValueError("outside_frozen_price_cap")
    if not lower-1e-8 <= value <= upper+1e-8:
        raise ValueError("outside_legal_price_limit")
    return round(value, 8)


class ExecutionEngine:
    def __init__(self, broker, manifest, journal, now, sleep, stop=lambda:False):
        self.broker, self.m, self.journal = broker, manifest, Path(journal)
        self.now, self.sleep, self.stop = now, sleep, stop
        self.end = datetime.fromisoformat(manifest["expires_at"])
        self.start = datetime.fromisoformat(manifest["not_before"])
        self.cleanup_end = max(self.end,self.now())+timedelta(seconds=120)
        self.observed, self.unknown, self.results = {}, False, []
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.attempts = []
        self.events = [json.loads(s) for s in self.journal.read_text(encoding="utf-8").splitlines()] if self.journal.exists() else []
        starts = [e for e in self.events if e.get("event") == "engine_started_v5"]
        if len(starts) > 1 or (starts and starts[0]["execution_manifest"] != manifest):
            raise ReconciliationRequired("Frozen manifest differs from the original durable batch")
        not_sent = {e["remark"] for e in self.events if e.get("event") == "not_sent_v5"}
        for event in self.events:
            if event.get("event") == "before_submit_v5" and event["remark"] not in not_sent:
                self.attempts.append(dict(event))
        if self.attempts and not starts:
            raise ReconciliationRequired("Order intents lack an immutable batch start")
        accepted = {e["remark"]:str(e["order_id"]) for e in self.events if e.get("event") == "accepted_v5"}
        for attempt in self.attempts:
            attempt["order_id"] = accepted.get(attempt["remark"])

    def event(self, event, **values):
        append_event(self.journal, event, batch_id=self.m["batch_id"], **values)

    def can_submit(self):
        now = self.now()
        return (now.date().isoformat() == self.m["execution_date"] and self.start <= now < self.end-timedelta(seconds=30)
                and not self.stop())

    def _order(self, attempt, force=False):
        orders = self.broker.orders(force=force)
        oid = attempt.get("order_id")
        found = [o for o in orders if (str(o["order_id"]) == str(oid) if oid else o.get("remark") == attempt["remark"])]
        if not found:
            raise ReceiptNotVisible("Receipt not yet visible: " + attempt["remark"])
        if len(found) != 1:
            raise ReconciliationRequired("Cannot uniquely reconcile accepted/ambiguous submission: " + attempt["remark"])
        row = dict(found[0])
        if row["code"] != attempt["code"] or row["trd_side"] != attempt["side"] or int(row["qty"]) != attempt["qty"]:
            raise ReconciliationRequired("Broker receipt differs from durable intent")
        filled = float(row["dealt_qty"])
        if not math.isfinite(filled) or filled < 0 or filled > attempt["qty"] or filled != int(filled):
            raise ReconciliationRequired("Invalid cumulative filled quantity")
        oid = str(row["order_id"])
        if not attempt.get("order_id"):
            attempt["order_id"] = oid
            self.event("accepted_v5", remark=attempt["remark"], order_id=oid, recovered=True)
        previous = self.observed.get(oid)
        if previous and (filled < previous["dealt_qty"] or (previous["order_status"] in TERMINAL and row["order_status"] != previous["order_status"])):
            # Out-of-order cache/push; require an authoritative refresh, never regress.
            if not force:
                return self._order(attempt, True)
            raise ReconciliationRequired("Authoritative receipt regressed")
        if row["order_status"] == "FILLED_ALL" and filled != attempt["qty"]:
            raise ReconciliationRequired("Filled status/quantity inconsistent")
        if filled and (not math.isfinite(float(row["dealt_avg_price"])) or float(row["dealt_avg_price"]) <= 0):
            raise ReconciliationRequired("Missing actual fill price")
        self.observed[oid] = row
        if previous != row:
            self.event("order_observed_v5", order=row)
        return row

    def _terminal(self, attempt, cancel=True):
        row = self._order(attempt, True)
        if row["order_status"] in TERMINAL:
            return row
        if cancel:
            try:
                self.broker.cancel(attempt["order_id"])
                self.event("cancel_requested_v5", order_id=attempt["order_id"])
            except Exception as exc:
                self.event("cancel_request_error_v5", order_id=attempt["order_id"], error=str(exc))
        until = min(self.now()+timedelta(seconds=30),self.cleanup_end)
        while self.now() < until:
            self.sleep(1.)
            row = self._order(attempt, False)
            if row["order_status"] in TERMINAL:
                final = self._order(attempt, True)
                if final["order_status"] in TERMINAL:
                    return final
        raise ReconciliationRequired("Cancel terminal state unconfirmed; no replacement allowed")

    def _wait(self, attempt):
        until = min(self.now()+timedelta(seconds=20),self.end)
        while self.now() < until and not self.stop():
            try:
                row = self._order(attempt, False)
            except ReceiptNotVisible:
                self.sleep(1.)
                continue
            if row["order_status"] in TERMINAL:
                return self._terminal(attempt, False)
            self.sleep(1.)
        return self._terminal(attempt)

    def _reconcile_start(self):
        orders = self.broker.orders(force=True)
        owned_ids = {a.get("order_id") for a in self.attempts if a.get("order_id")}
        remarks = {a["remark"] for a in self.attempts}
        for remark in remarks:
            if sum(o.get("remark") == remark for o in orders) > 1:
                raise ReconciliationRequired("Duplicate broker orders match one durable intent")
        foreign = [o for o in orders if o["order_status"] not in TERMINAL
                   and str(o["order_id"]) not in owned_ids and o.get("remark") not in remarks]
        if foreign:
            raise ReconciliationRequired("Other live account orders exist; do not cancel or compete with them")
        for attempt in self.attempts:
            self._terminal(attempt)
        positions = self.broker.positions(force=True)
        expected = dict(self.m["initial_quantities"])
        for attempt in self.attempts:
            filled = int(self.observed[str(attempt["order_id"])]["dealt_qty"])
            expected[attempt["code"]] = expected.get(attempt["code"],0)+filled*(1 if attempt["side"] == "BUY" else -1)
        actual = {c:int(r["qty"]) for c,r in positions.items() if r["qty"] > 0}
        if {c:q for c,q in expected.items() if q > 0} != actual:
            raise ReconciliationRequired("Actual holdings changed outside this frozen batch")
        return positions

    def _buy_quantity(self, action, quantity, price, positions, funds):
        code = action["code"]
        nav = self.m["nav"]
        held = int(positions.get(code,{}).get("qty",0))
        names = {c for c,r in positions.items() if r["qty"] > 0}
        if code not in names and len(names) >= self.m["top_k"]:
            return 0, "holding_limit_pending_sell"
        usable = max(0.,float(funds["cash"])-float(funds.get("frozen_cash",0))-nav*(1-self.m["risk_degree"])-10)
        quantity = min(quantity, affordable_buy_quantity(code,usable,price))
        per_stock_value = max(0.,(action["weight"]+.005)*nav-held*self.m["reference_prices"][code])
        quantity = min(quantity, max(0,math.floor((per_stock_value+1e-8)/price)))
        exposure = sum(float(r["qty"])*self.m["reference_prices"][c] for c,r in positions.items() if r["qty"] > 0)
        portfolio_room = max(0.,nav*self.m["risk_degree"]-exposure)
        quantity = min(quantity,max(0,math.floor((portfolio_room+1e-8)/price)))
        quantity = legal_order_quantity(code,quantity,"BUY")
        return quantity, "cash_lot_or_risk_cap"

    def run(self):
        try:
            positions = self._reconcile_start()
            funds = self.broker.funds(force=True)
            if not any(e.get("event") == "engine_started_v5" for e in self.events):
                self.event("engine_started_v5", execution_manifest=self.m)
            buying_phase = False
            for action in self.m["actions"]:
                code, side = action["code"],action["side"]
                if side == "BUY" and not buying_phase:
                    # Only settled/confirmed sell proceeds can finance the buy phase.
                    positions = self.broker.positions(force=True)
                    funds = self.broker.funds(force=True)
                    buying_phase = True
                own = [a for a in self.attempts if a["code"] == code]
                while self.can_submit() and len(own) < MAX_ATTEMPTS:
                    h = int(positions.get(code,{}).get("qty",0))
                    gap = action["target"]-h
                    quantity = max(0,gap if side == "BUY" else -gap)
                    if not quantity:
                        break
                    attempt_number = len(own)+1
                    try:
                        price = limit_price(code,side,self.broker.quote(code),action["reference_price"],attempt_number,self.now())
                    except ValueError as exc:
                        reason = str(exc)
                        if reason == "outside_frozen_price_cap":
                            self.sleep(min(5.,max(0.,(self.end-timedelta(seconds=30)-self.now()).total_seconds())))
                            continue
                        self.event("action_constrained_v5",code=code,reason=reason)
                        break
                    if side == "BUY":
                        quantity,reason = self._buy_quantity(action,quantity,price,positions,funds)
                    else:
                        quantity = legal_order_quantity(code,quantity,"SELL",h,int(positions.get(code,{}).get("can_sell_qty",0)))
                        reason = "t_plus_one_or_order_unit"
                    if quantity <= 0:
                        self.event("action_constrained_v5",code=code,reason=reason)
                        break
                    if not self.can_submit():
                        break
                    intent = hashlib.sha256((self.m["batch_id"]+code+side+str(attempt_number)).encode()).hexdigest()[:24]
                    record = dict(code=code,side=side,qty=quantity,price=price,attempt=attempt_number,remark="cn5_"+intent)
                    self.event("before_submit_v5",**record)
                    self.attempts.append(record)
                    own.append(record)
                    # Any transport error or missing receipt is AMBIGUOUS, not a retry.
                    try:
                        oid = self.broker.submit(code,side,quantity,price,record["remark"],self.can_submit)
                    except SubmissionNotSent:
                        self.event("not_sent_v5",remark=record["remark"],reason="deadline_before_order_api")
                        self.attempts.remove(record)
                        own.remove(record)
                        break
                    if not oid:
                        raise ReconciliationRequired("Submission returned no order id")
                    record["order_id"] = str(oid)
                    self.event("accepted_v5",remark=record["remark"],order_id=str(oid))
                    row = self._wait(record)
                    filled = int(row["dealt_qty"])
                    positions.setdefault(code,dict(qty=0,can_sell_qty=0))
                    positions[code]["qty"] += filled*(1 if side == "BUY" else -1)
                    if side == "SELL":
                        positions[code]["can_sell_qty"] = max(0,positions[code]["can_sell_qty"]-filled)
                    else:
                        funds["cash"] -= buy_cash_bound(filled,float(row["dealt_avg_price"]))
                    self.results.append(dict(code=code,side=side,attempt=attempt_number,order_id=str(oid),
                                             dealt_qty=filled,dealt_avg_price=float(row["dealt_avg_price"]),order_status=row["order_status"]))
            return self._summary()
        except Exception as exc:
            self.unknown = True
            self.event("reconciliation_required_v5",error=str(exc))
            # Only known owned IDs may be cancelled. Unknown acceptance is never resent.
            for attempt in self.attempts:
                try:
                    # A uniquely matching durable remark can recover an otherwise lost ID.
                    self._terminal(attempt)
                except Exception:
                    pass
            return self._summary(error=str(exc))

    def _summary(self,error=None):
        try:
            positions = self.broker.positions(force=True)
            funds = self.broker.funds(force=True)
            actual = {c:int(r["qty"]) for c,r in positions.items() if r["qty"] > 0}
            differences = {c:int(q)-actual.get(c,0) for c,q in self.m["targets"].items() if int(q) != actual.get(c,0)}
            if set(actual)-set(self.m["reference_prices"]):
                raise ReconciliationRequired("Unexpected final holding")
            if any(o["order_status"] not in TERMINAL for o in self.broker.orders(force=True)
                   if o.get("remark") in {a["remark"] for a in self.attempts}):
                self.unknown = True
            if len(actual) > self.m["top_k"]:
                self.unknown = True
            cash_ok = float(funds["cash"])-float(funds.get("frozen_cash",0)) >= self.m["nav"]*(1-self.m["risk_degree"])-1e-6
            if not cash_ok and any(a["side"] == "BUY" for a in self.attempts):
                self.unknown = True
            status = "reconciliation_required" if self.unknown else "completed" if not differences else "partial_constrained"
            value = dict(status=status,environment="SIMULATE",execution_date=self.m["execution_date"],
                batch_id=self.m["batch_id"],completed_at=self.now().isoformat(),positions=len(actual),
                quantities=actual,remaining_differences=differences,cash=float(funds["cash"]),
                total_assets=float(funds["total_assets"]),cash_reserve_ok=cash_ok,
                submitted_orders=len(self.attempts),fully_filled_orders=sum(r["order_status"] == "FILLED_ALL" for r in self.observed.values()),
                order_ids=[a.get("order_id") for a in self.attempts],results=self.results,error=error)
        except Exception as exc:
            value = dict(status="reconciliation_required",environment="SIMULATE",batch_id=self.m["batch_id"],
                         execution_date=self.m["execution_date"],error=error or str(exc),completed_at=self.now().isoformat())
        self.event("engine_finished_v5",summary=value)
        return value
