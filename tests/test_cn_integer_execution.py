"""Offline fault regressions; no OpenD connection and no real orders."""
from copy import deepcopy
from datetime import datetime, timedelta
import json

import pytest

from paper_futu.cn_common import CHINA_TZ, tiered_weights
from paper_futu.cn_integer_targets import allocate_targets, affordable_quantity, legal_order_quantity
from paper_futu.cn_execution_engine import (ExecutionEngine, limit_price, SubmissionNotSent,
    ReconciliationRequired, affordable_buy_quantity, buy_cash_bound)

A, B = "SH.600000", "SZ.000001"


class Clock:
    def __init__(self):
        self.value = datetime(2026,10,20,13,30,tzinfo=CHINA_TZ)
    def now(self):
        return self.value
    def sleep(self,seconds):
        self.value += timedelta(seconds=seconds)


def quote(clock,price=10.,**kwargs):
    return dict(last_price=price,prev_close_price=price,ask_price=price,bid_price=price,
                price_spread=.01,lot_size=100,suspension=False,update_time=clock.now().isoformat(),**kwargs)


def manifest(clock,target=100):
    held = 900-target
    return dict(batch_id="testbatch",environment="SIMULATE",execution_date="2026-10-20",nav=10000.,
        not_before=clock.now().isoformat(),expires_at=(clock.now()+timedelta(minutes=20)).isoformat(),
        top_k=20,risk_degree=.95,initial_quantities={A:held},reference_prices={A:10.,B:10.},
        targets={A:held,B:target},actions=[dict(code=B,side="BUY",target=target,held=0,qty=target,
            weight=target/1000,reference_price=10.)])


class Broker:
    def __init__(self,clock,m,scenarios=("fill",)):
        self.clock,self.m = clock,m
        self.pos = {A:dict(qty=m["initial_quantities"][A],can_sell_qty=m["initial_quantities"][A])}
        self.money = 10000.-self.pos[A]["qty"]*10
        self.rows,self.calls,self.cancelled = [],[],[]
        self.scenarios = list(scenarios)
    def positions(self,force=False):
        return deepcopy(self.pos)
    def funds(self,force=False):
        return dict(cash=self.money,total_assets=10000.,frozen_cash=0.)
    def orders(self,force=False):
        return deepcopy(self.rows)
    def quote(self,code):
        return quote(self.clock)
    def _fill(self,row,quantity):
        if quantity <= 0:
            return
        row["dealt_qty"] += quantity
        row["dealt_avg_price"] = 10.
        self.pos.setdefault(row["code"],dict(qty=0,can_sell_qty=0))
        if row["trd_side"] == "BUY":
            self.pos[row["code"]]["qty"] += quantity
            self.money -= quantity*10
        else:
            self.pos[row["code"]]["qty"] -= quantity
            self.pos[row["code"]]["can_sell_qty"] -= quantity
            self.money += quantity*10
    def submit(self,code,side,quantity,price,remark,allowed):
        assert allowed()
        self.calls.append((code,side,quantity,price,remark))
        scenario = self.scenarios.pop(0) if self.scenarios else "fill"
        row = dict(order_id=str(len(self.rows)+1),code=code,trd_side=side,qty=quantity,
                   dealt_qty=0,dealt_avg_price=0.,order_status="SUBMITTED",remark=remark,scenario=scenario)
        self.rows.append(row)
        if scenario in {"fill","timeout","missingid"}:
            self._fill(row,quantity)
            row["order_status"] = "FILLED_ALL"
        elif scenario == "partial":
            self._fill(row,quantity//2)
            row["order_status"] = "FILLED_PART"
        if scenario == "timeout":
            raise TimeoutError("accepted by server but response lost")
        return None if scenario == "missingid" else row["order_id"]
    def cancel(self,oid):
        self.cancelled.append(oid)
        row = next(r for r in self.rows if r["order_id"] == oid)
        if row["scenario"] == "race":
            self._fill(row,row["qty"]-row["dealt_qty"])
            row["order_status"] = "FILLED_ALL"
        elif row["scenario"] == "stuck":
            return
        else:
            row["order_status"] = "CANCELLED_PART" if row["dealt_qty"] else "CANCELLED_ALL"


def run(tmp_path,scenarios=("fill",),target=100):
    c = Clock(); m = manifest(c,target); b = Broker(c,m,scenarios)
    engine = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep)
    return c,m,b,engine.run()


def test_zhongji_keeps_one_lot_instead_of_liquidating():
    codes = ["SZ.300308"]+[f"SH.{600000+i}" for i in range(19)]
    weights = tiered_weights(codes,risk_degree=.95,top_k=20,top_tier_size=6,top_tier_weight=.4)
    prices = {c:dict(last_price=654.47 if c == codes[0] else 10.,prev_close_price=654.47 if c == codes[0] else 10.,suspension=False) for c in codes}
    r = allocate_targets(codes,weights,1009718.671,{codes[0]:dict(qty=100,can_sell_qty=100)},prices,944271.671)
    assert r["targets"][codes[0]] == 100
    assert not any(a["code"] == codes[0] for a in r["actions"])
    assert r["target_exposure"] <= 1009718.671*.95
    assert r["expected_post_cash"] >= 1009718.671*.05


def test_affordability_never_rounds_up():
    assert affordable_quantity(B,999.,10.) == 0
    assert affordable_quantity(B,1000.,10.) == 100
    assert affordable_quantity("SH.688521",1999.,10.) == 0
    assert affordable_quantity("SH.688521",2000.,10.) == 200


def test_small_buy_always_reserves_minimum_five_yuan_fee():
    assert buy_cash_bound(100,2.) == 205.
    assert affordable_buy_quantity(B,204.99,2.) == 0
    assert affordable_buy_quantity(B,205.,2.) == 100
    assert affordable_buy_quantity(B,1005.,10.) == 100


def test_hard_overweight_limit_and_deselection():
    r = allocate_targets([B],{B:.95},10000.,{A:dict(qty=100,can_sell_qty=100)},
                         {A:dict(last_price=10.,prev_close_price=10.),B:dict(last_price=100.,prev_close_price=100.)},10000.)
    assert r["targets"][A] == 0
    assert r["targets"][B] == 0  # 100 shares would exceed .955 of NAV


def test_t_plus_one_and_locked_risk_never_finance_new_buys():
    r = allocate_targets([B],{B:.95},10000.,{A:dict(qty=1000,can_sell_qty=0)},
                         {A:dict(last_price=10.,prev_close_price=10.),B:dict(last_price=10.,prev_close_price=10.)},0.)
    assert r["targets"][A] == 1000
    assert r["targets"][B] == 0
    assert not any(a["side"] == "BUY" for a in r["actions"])


@pytest.mark.parametrize("code,quantity,held,expected",[(B,50,150,0),(B,150,150,150),
    ("SH.688521",91,491,0),("SH.688521",191,191,191),("SH.688521",201,491,201)])
def test_board_order_units_and_full_odd_lot_exit(code,quantity,held,expected):
    assert legal_order_quantity(code,quantity,"SELL",held,held) == expected


def test_full_fill_and_reload_does_not_rebuy(tmp_path):
    c,m,b,result = run(tmp_path)
    assert result["status"] == "completed"
    assert len(b.calls) == 1 and b.pos[B]["qty"] == 100
    again = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert again["status"] == "completed" and len(b.calls) == 1


def test_partial_cancel_then_only_buy_remaining_legal_lot(tmp_path):
    _,_,b,r = run(tmp_path,("partial","fill"),200)
    assert r["status"] == "completed"
    assert [c[2] for c in b.calls] == [200,100]
    assert b.pos[B]["qty"] == 200 and b.cancelled == ["1"]


def test_partial_residual_below_lot_is_not_replaced_with_full_lot(tmp_path):
    _,_,b,r = run(tmp_path,("partial",),100)
    assert r["status"] == "partial_constrained"
    assert len(b.calls) == 1 and b.pos[B]["qty"] == 50


def test_cancel_fill_race_does_not_duplicate(tmp_path):
    _,_,b,r = run(tmp_path,("race",))
    assert r["status"] == "completed"
    assert len(b.calls) == 1 and b.pos[B]["qty"] == 100


def test_maximum_three_attempts(tmp_path):
    _,_,b,r = run(tmp_path,("open","open","open"))
    assert r["status"] == "partial_constrained"
    assert len(b.calls) == 3 and len(b.cancelled) == 3


def test_cancel_ack_without_terminal_forbids_replacement(tmp_path):
    _,_,b,r = run(tmp_path,("stuck",))
    assert r["status"] == "reconciliation_required" and len(b.calls) == 1


@pytest.mark.parametrize("scenario",["timeout","missingid"])
def test_ambiguous_acceptance_only_recovered_by_unique_remark(tmp_path,scenario):
    c,m,b,r = run(tmp_path,(scenario,))
    assert r["status"] == "reconciliation_required" and len(b.calls) == 1
    resumed = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert resumed["status"] == "completed" and len(b.calls) == 1


def test_duplicate_matching_ambiguous_orders_stop(tmp_path):
    c,m,b,r = run(tmp_path,("timeout",))
    b.rows.append(dict(b.rows[0],order_id="duplicate"))
    resumed = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert resumed["status"] == "reconciliation_required" and len(b.calls) == 1


def test_foreign_live_order_is_not_cancelled(tmp_path):
    c = Clock();m = manifest(c);b = Broker(c,m)
    b.rows.append(dict(order_id="manual",code=A,trd_side="BUY",qty=100,dealt_qty=0,
                       dealt_avg_price=0.,order_status="SUBMITTED",remark="manual"))
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert r["status"] == "reconciliation_required" and not b.calls and not b.cancelled


def test_frozen_price_cap_tick_and_fallback():
    c = Clock();q = quote(c)
    assert limit_price(B,"BUY",q,10.,1,c.now()) == 10.
    assert limit_price(B,"BUY",q,10.,3,c.now()) == 10.02
    q["ask_price"] = float("nan")
    assert limit_price(B,"BUY",q,10.,1,c.now()) == 10.01
    q["ask_price"] = 10.06
    with pytest.raises(ValueError,match="frozen_price_cap"):
        limit_price(B,"BUY",q,10.,1,c.now())
    q["update_time"] = (c.now()-timedelta(minutes=3)).isoformat()
    with pytest.raises(ValueError,match="stale_quote"):
        limit_price(B,"BUY",q,10.,1,c.now())


def test_deadline_stop_and_holding_limit(tmp_path):
    c = Clock();m = manifest(c);b = Broker(c,m)
    m["expires_at"] = (c.now()+timedelta(seconds=29)).isoformat()
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert r["status"] == "partial_constrained" and not b.calls
    m["expires_at"] = (c.now()+timedelta(minutes=20)).isoformat();m["top_k"] = 1
    r = ExecutionEngine(b,m,tmp_path/"orders2.jsonl",c.now,c.sleep).run()
    assert r["status"] == "partial_constrained" and not b.calls


def test_stop_request_blocks_all_new_orders(tmp_path):
    c = Clock();m = manifest(c);b = Broker(c,m)
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep,stop=lambda:True).run()
    assert r["status"] == "partial_constrained" and not b.calls


def test_actual_holding_change_is_not_mistaken_for_own_fill(tmp_path):
    c = Clock();m = manifest(c);b = Broker(c,m)
    b.pos[B] = dict(qty=100,can_sell_qty=0)
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert r["status"] == "reconciliation_required" and not b.calls


def test_three_attempt_budget_survives_restart(tmp_path):
    c,m,b,r = run(tmp_path,("open","open","open"))
    assert len(b.calls) == 3
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert r["status"] == "partial_constrained" and len(b.calls) == 3


def test_local_deadline_after_rate_wait_is_not_ambiguous_submission(tmp_path):
    c = Clock();m = manifest(c);b = Broker(c,m)
    def blocked(*args):
        c.sleep(1201)
        raise SubmissionNotSent()
    b.submit = blocked
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert r["status"] == "partial_constrained" and r["submitted_orders"] == 0


def test_altered_frozen_target_is_rejected_on_reload(tmp_path):
    c,m,b,r = run(tmp_path)
    m["targets"][B] = 200
    with pytest.raises(ReconciliationRequired,match="manifest"):
        ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep)


def test_delayed_order_visibility_is_not_blindly_resubmitted(tmp_path):
    c = Clock();m = manifest(c);b = Broker(c,m)
    actual_orders = b.orders
    hide = [2]
    def delayed(force=False):
        if b.rows and not force and hide[0]:
            hide[0] -= 1
            return []
        return actual_orders(force)
    b.orders = delayed
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert r["status"] == "completed" and len(b.calls) == 1


def test_failed_sell_keeps_its_slot_but_other_retained_buys_can_continue(tmp_path):
    c = Clock();m = manifest(c)
    m["actions"].insert(0,dict(code=A,side="SELL",qty=100,held=800,target=700,weight=.7,reference_price=10.))
    m["targets"][A] = 700
    b = Broker(c,m,("open","open","open","fill"))
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert r["status"] == "partial_constrained"
    assert [v[1] for v in b.calls] == ["SELL","SELL","SELL","BUY"]
    assert b.pos[A]["qty"] == 800 and b.pos[B]["qty"] == 100


def test_late_explicit_resume_can_cancel_but_never_buy(tmp_path):
    c,m,b,r = run(tmp_path,("stuck",))
    c.sleep(3600)
    b.rows[0]["scenario"] = "open"
    r = ExecutionEngine(b,m,tmp_path/"orders.jsonl",c.now,c.sleep).run()
    assert r["status"] == "partial_constrained"
    assert len(b.calls) == 1 and b.rows[0]["order_status"] == "CANCELLED_ALL"


def test_correction_supplement_fees_do_not_overwrite_original(tmp_path):
    from paper_futu.prospective_s6 import filled_costs
    import pandas as pd
    original = tmp_path/"order_results_2026-10-20.csv"
    pd.DataFrame([dict(order_id="old",trd_side="SELL",dealt_qty=100,dealt_avg_price=10.)]).to_csv(original,index=False)
    before = original.read_bytes()
    supplemental = tmp_path/"correction_v5_2026-10-20_SZ300308_100.orders.csv"
    pd.DataFrame([dict(order_id="new",trd_side="BUY",dealt_qty=100,dealt_avg_price=640.)]).to_csv(supplemental,index=False)
    result = filled_costs(original)
    assert result["filled_notional"] == 65000.
    assert result["modeled_fees_cny"] == 37.
    assert original.read_bytes() == before


@pytest.mark.parametrize("stamp",["2026-10-13T09:51:00","2026-10-13T10:00:00","2026-10-13T13:00:00"])
def test_default_window_does_not_extend_when_opend_login_is_late(stamp,monkeypatch):
    import paper_futu.execution_override as override
    from paper_futu.cn_execution_session import normal_bounds
    monkeypatch.setattr(override,"read_override",lambda now:None)
    now = datetime.fromisoformat(stamp).replace(tzinfo=CHINA_TZ)
    start,end = normal_bounds(dict(execution_date="2026-10-13"),now)
    assert start.hour == 9 and start.minute == 31
    assert end.hour == 9 and end.minute == 51
    assert now >= end
