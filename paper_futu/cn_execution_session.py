"""Paper-only broker adapter and shared locked execution entry points."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from paper_futu.cn_common import (ROOT, STATE_DIR, china_now, CHINA_TZ, load_deployment,
    validate_plan_deployment, validate_execution_session, select_audited_holdings, tiered_weights)
from paper_futu.cn_integer_targets import allocate_targets
from paper_futu.cn_broker_queries import (query_positions, PositionCache, bounded_read,
    FUNDS_BUDGET, ORDERS_BUDGET, SUBMIT_BUDGET, CANCEL_BUDGET)
from paper_futu.cn_execution_engine import ExecutionEngine, ReconciliationRequired, SubmissionNotSent
from paper_futu.prospective_s6 import write_once, digest
from paper_futu.cn_runtime_protocol import verify_active_protocol


@contextmanager
def account_lock():
    """Every mutating CN entry point takes the same OS-released account lock."""
    import msvcrt
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    with (STATE_DIR/"cn_account_execution.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:
            raise RuntimeError("Another account executor owns the lock; no duplicate launch") from exc
        try:
            yield
        finally:
            handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def save_result(path,value):
    """Derived current status may change; immutable manifests/journals never overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temporary,path)


class FutuBroker:
    def __init__(self):
        from futu import OpenQuoteContext, TradeOrderHandlerBase, RET_OK
        from paper_futu.execute_cn_plan import account_context, paper_account_id
        self.trade,self.quote_context = account_context(),None
        try:
            self.account_id = paper_account_id(self.trade)
            self.quote_context = OpenQuoteContext(host="127.0.0.1",port=11111)
            self.position_cache = PositionCache(self._positions)
            self.pushes,self.push_lock = {},threading.Lock()
            owner = self
            class Handler(TradeOrderHandlerBase):
                def on_recv_rsp(self, rsp_pb):
                    ret, frame = super().on_recv_rsp(rsp_pb)
                    if ret == RET_OK:
                        with owner.push_lock:
                            for row in frame.to_dict("records"):
                                if "acc_id" not in row or int(row["acc_id"]) == owner.account_id:
                                    key = str(row["order_id"])
                                    old = owner.pushes.get(key)
                                    if not old or float(row["dealt_qty"]) >= float(old["dealt_qty"]):
                                        owner.pushes[key] = row
                    return ret,frame
            self.trade.set_handler(Handler())
        except Exception:
            self.close()
            raise

    def close(self):
        if self.quote_context is not None:
            self.quote_context.close()
        self.trade.close()

    def _positions(self):
        return query_positions(self.trade,self.account_id)

    def positions(self,force=False):
        frame = self.position_cache.get(force)
        return {str(r["code"]):dict(qty=int(r["qty"]),can_sell_qty=int(r["can_sell_qty"]))
                for r in frame.to_dict("records") if float(r["qty"]) > 0}

    def funds(self,force=False):
        from futu import TrdEnv,Currency
        frame = bounded_read(lambda:self.trade.accinfo_query(trd_env=TrdEnv.SIMULATE,
            acc_id=self.account_id,refresh_cache=force,currency=Currency.CNH),FUNDS_BUDGET,"查询模拟资金",force)
        row = frame.iloc[0]
        result = dict(cash=float(row.cash),total_assets=float(row.total_assets),frozen_cash=float(row.get("frozen_cash",0)))
        if not all(math.isfinite(v) and v >= 0 for v in result.values()) or result["total_assets"] <= 0:
            raise ValueError("Invalid simulated account funds")
        return result

    def orders(self,force=False):
        from futu import TrdEnv
        frame = bounded_read(lambda:self.trade.order_list_query(trd_env=TrdEnv.SIMULATE,
            acc_id=self.account_id,refresh_cache=force),ORDERS_BUDGET,"查询模拟订单",force)
        # Forced query is authoritative. Pushes accelerate only non-forced observation.
        rows = {str(r["order_id"]):r for r in frame.to_dict("records")}
        if not force:
            with self.push_lock:
                for key,row in self.pushes.items():
                    old = rows.get(key)
                    if not old or (float(row["dealt_qty"]) >= float(old["dealt_qty"])
                                   and str(row.get("updated_time","")) >= str(old.get("updated_time",""))):
                        rows[key] = row
        columns = ["order_id","code","trd_side","qty","dealt_qty","dealt_avg_price","order_status","remark"]
        result = []
        for r in rows.values():
            value = {k:(str(r[k]) if k in {"order_id","code","trd_side","order_status","remark"} else float(r[k])) for k in columns}
            if value["dealt_qty"] == 0:
                value["dealt_avg_price"] = 0.
            result.append(value)
        return result

    def quotes(self,codes):
        from paper_futu.execute_cn_plan import fresh_quotes
        return fresh_quotes(self.quote_context,list(codes),execution=True).to_dict("index")

    def quote(self,code):
        return self.quotes([code])[code]

    def session(self,day,model_year):
        from paper_futu.smoke_order import require_ok
        state = require_ok(self.quote_context.get_global_state(),"查询市场时段")
        validate_execution_session(china_now().date().isoformat(),day,model_year,state["market_sh"],state["market_sz"])

    def submit(self,code,side,quantity,price,remark,allowed):
        from futu import TrdEnv,TrdSide,OrderType,TimeInForce
        SUBMIT_BUDGET.acquire()
        if not allowed():
            raise SubmissionNotSent("Window closed before actual API submission")
        ret,data = self.trade.place_order(price=price,qty=quantity,code=code,
            trd_side=TrdSide.BUY if side == "BUY" else TrdSide.SELL,
            order_type=OrderType.NORMAL,trd_env=TrdEnv.SIMULATE,acc_id=self.account_id,
            time_in_force=TimeInForce.DAY,remark=remark)
        if ret != 0 or len(data) != 1:
            raise ReconciliationRequired("Ambiguous or rejected submission, never blind retry: "+str(data))
        return str(data.iloc[0].order_id)

    def cancel(self,oid):
        from futu import ModifyOrderOp,TrdEnv
        from paper_futu.smoke_order import require_ok
        CANCEL_BUDGET.acquire()
        return require_ok(self.trade.modify_order(ModifyOrderOp.CANCEL,str(oid),0,0,
            trd_env=TrdEnv.SIMULATE,acc_id=self.account_id),"申请撤销本批订单")


def normal_bounds(plan,now):
    if now.date().isoformat() != plan["execution_date"]:
        raise ValueError("Execution date expired; no next-day catchup")
    start = now.replace(hour=9,minute=31,second=0,microsecond=0)
    end = now.replace(hour=9,minute=51,second=0,microsecond=0)
    from paper_futu.execution_override import read_override
    override = read_override(now)
    if override and override["execution_date"] == plan["execution_date"]:
        for key in ["signal_date","model_sha256"]:
            if override[key] != plan[key]:
                raise ValueError("Override binding mismatch")
        start,end = datetime.fromisoformat(override["not_before"]),datetime.fromisoformat(override["expires_at"])
    return start,end


def make_manifest(plan,positions,funds,quotes,selected,weights,start,end,kind="rebalance"):
    nav = funds["total_assets"]
    allocation = allocate_targets(selected,weights,nav,positions,quotes,funds["cash"]-funds.get("frozen_cash",0),plan["risk_degree"])
    batch_id = hashlib.sha256((kind+plan["execution_date"]+plan["model_sha256"]+start.isoformat()).encode()).hexdigest()[:24]
    return dict(version=5,kind=kind,environment="SIMULATE",execution_date=plan["execution_date"],
        signal_date=plan["signal_date"],model_sha256=plan["model_sha256"],batch_id=batch_id,
        active_plan_sha256=digest(STATE_DIR/"active_plan.json"),nav=nav,initial_cash=funds["cash"],
        initial_quantities={c:int(r["qty"]) for c,r in positions.items() if r["qty"] > 0},
        selected=selected,weights=weights,top_k=plan["top_k"],risk_degree=plan["risk_degree"],
        not_before=start.isoformat(),expires_at=end.isoformat(),**allocation)


def run_manifest(broker,manifest,journal,result_path,reconcile_only=False):
    result = ExecutionEngine(broker,manifest,journal,china_now,time.sleep,
        stop=lambda:reconcile_only or (STATE_DIR/"stop_requested").exists()).run()
    result["manifest_sha256"] = hashlib.sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
    result["journal_sha256"] = digest(journal)
    result["execution_rule"] = "integer_and_retry_v5"
    import csv
    try:
        receipts = [r for r in broker.orders(force=True) if str(r["order_id"]) in {str(v) for v in result.get("order_ids",[]) if v}]
    except Exception as exc:
        result.update(status="reconciliation_required",error="Final receipt export failed: "+str(exc))
        receipts = []
    csv_path = Path(result_path).with_name(Path(result_path).name.replace(".result.json",".orders.csv"))
    with csv_path.open("w",newline="",encoding="utf-8") as stream:
        columns = ["order_id","code","trd_side","qty","dealt_qty","dealt_avg_price","order_status","remark"]
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader(); writer.writerows(receipts)
    result["order_receipts_file"] = csv_path.name
    result["order_receipts_sha256"] = digest(csv_path)
    save_result(result_path,result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def rebalance_main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen integer-target CN simulated rebalance")
    parser.add_argument("--submit",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args(argv)
    from paper_futu.cn_rank80_runtime import enabled
    if enabled():
        if args.resume:
            raise ValueError("Rank80 uses its own durable batch; never resume a legacy S6 batch")
        from paper_futu.cn_rank80_runtime import execute
        return execute(preview=not args.submit)
    with account_lock():
        plan = json.loads((STATE_DIR/"active_plan.json").read_text(encoding="utf-8"))
        model,meta = load_deployment(); validate_plan_deployment(plan,model,meta)
        complete = STATE_DIR/f"execution_{plan['execution_date']}.json"
        if complete.exists() and not args.resume:
            print("Existing completion receipt retained; not executing the batch again")
            return
        manifest_path = STATE_DIR/f"execution_v5_{plan['execution_date']}.plan.json"
        journal = STATE_DIR/f"orders_v5_{plan['execution_date']}.jsonl"
        if args.resume and not manifest_path.exists():
            raise ValueError("No version-5 frozen batch; legacy batches cannot be silently reinterpreted")
        if args.submit:
            verify_active_protocol()
        broker = FutuBroker()
        try:
            now = china_now()
            start,end = normal_bounds(plan,now)
            within_window = start <= now < end-timedelta(seconds=30)
            if args.submit and not args.resume and not within_window:
                raise ValueError("Outside the fixed 20-minute rebalance window")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["model_sha256"] != plan["model_sha256"] or manifest["active_plan_sha256"] != digest(STATE_DIR/"active_plan.json"):
                    raise ValueError("Frozen batch binding changed")
                if journal.exists() and not args.resume:
                    raise ValueError("Existing intent journal; use explicit --resume after reconciliation")
            else:
                legacy = STATE_DIR/f"orders_{plan['execution_date']}.jsonl"
                if legacy.exists() and any(json.loads(s).get("event") == "before_submit" for s in legacy.read_text(encoding="utf-8").splitlines()):
                    raise ValueError("Legacy submitted batch needs manual reconciliation, not a new target calculation")
                positions,funds = broker.positions(True),broker.funds(True)
                quotes = broker.quotes(sorted(set(plan["ordered_candidates"])|set(positions)))
                values = plan["predictions"]
                def tradable(c,buying):
                    row = quotes[c]
                    previous = float(row["prev_close_price"])
                    return (not row["suspension"] and previous > 0
                        and (float(row["last_price"])/previous-1 < .095 if buying else float(row["last_price"])/previous-1 > -.095))
                eligible = [c for c in plan["ordered_candidates"] if tradable(c,True)]
                sellable = [c for c,r in positions.items() if r["can_sell_qty"] >= r["qty"] and tradable(c,False)]
                selected = select_audited_holdings(values,eligible,list(positions),plan["top_k"],plan["n_drop"],sellable)
                if len(selected) != plan["top_k"]:
                    raise ValueError("Insufficient eligible names; no implicit portfolio change")
                weights = tiered_weights(selected,**{k:plan[k] for k in ["risk_degree","top_k","top_tier_size","top_tier_weight"]})
                manifest = make_manifest(plan,positions,funds,quotes,selected,weights,start,end)
            if not args.submit:
                print(json.dumps(manifest,ensure_ascii=False,indent=2))
                print("Preview only; no orders submitted")
                return manifest
            reconcile_only = not within_window
            try:
                broker.session(plan["execution_date"],meta["model_year"])
            except ValueError:
                if not args.resume:
                    raise
                reconcile_only = True
            if not manifest_path.exists():
                write_once(manifest_path,manifest)
            result = run_manifest(broker,manifest,journal,STATE_DIR/f"execution_v5_{plan['execution_date']}.result.json",reconcile_only)
            if result["status"] != "reconciliation_required":
                import shutil
                conventional = STATE_DIR/f"order_results_{plan['execution_date']}.csv"
                if not complete.exists() and not conventional.exists():
                    shutil.copyfile(STATE_DIR/result["order_receipts_file"],conventional)
                # Both statuses are closed for automation; partial is NOT a successful fill claim.
                if complete.exists():
                    save_result(STATE_DIR/f"execution_v5_{plan['execution_date']}.resume_status.json",result)
                else:
                    write_once(complete,result)
            if result["status"] == "reconciliation_required":
                raise ReconciliationRequired(result.get("error") or "Unconfirmed execution state")
            return result
        finally:
            broker.close()
