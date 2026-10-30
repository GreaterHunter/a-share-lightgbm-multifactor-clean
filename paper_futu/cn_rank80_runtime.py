"""Logged-on Windows worker for rank80 paper trading; no Codex dependency."""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from paper_futu.cn_common import ROOT, STATE_DIR, HISTORICAL_PROVIDER, china_now, load_deployment, validate_serving_date
from paper_futu.cn_rank80_strategy import STRATEGY, RULES, bootstrap, decide, apply_fills
from paper_futu.cn_execution_session import account_lock, save_result, FutuBroker, run_manifest
from paper_futu.cn_execution_engine import TERMINAL, checked_quote, affordable_buy_quantity
from paper_futu.cn_runtime_protocol import verify_active_protocol
from paper_futu.prospective_s6 import digest, read, write_once

DIRECTORY = STATE_DIR/'rank80_full10_5'
MODEL_PYTHON = 'D:/Anaconda/envs/quant-lightgbm/python.exe'


def enabled():
    return read(ROOT/'configs/futu_cn_paper.json').get('execution_strategy') == STRATEGY


def dates():
    return (HISTORICAL_PROVIDER/'calendars/day_future.txt').read_text().splitlines()


def load_signal(day):
    """Only complete, immutable public-data packages can drive this executor."""
    import pandas as pd
    folder=STATE_DIR/day
    ready=read(folder/'independent_data_ready.json')
    if ready['date']!=day or digest(folder/'plan.json')!=ready['plan_sha256']:
        raise ValueError('Missing or modified completed daily signal')
    for name,h in ready['signal_artifacts'].items():
        if digest(folder/name)!=h:raise ValueError('Daily signal hash differs: '+name)
    manifest=read(folder/'market_manifest.json')
    if digest(folder/'market_manifest.json')!=ready['market_manifest_sha256']:
        raise ValueError('Market manifest hash differs')
    for name,h in manifest['artifacts'].items():
        if digest(folder/name)!=h:raise ValueError('Market artifact hash differs: '+name)
    plan=read(folder/'plan.json')
    model,meta=load_deployment()
    validate_serving_date(meta,day)
    if plan['signal_date']!=day or plan['model_sha256']!=meta['model_sha256']:
        raise ValueError('Daily signal model/date differs')
    bars=pd.read_parquet(folder/'bars.parquet')
    factors={c[:2]+'.'+c[2:]:float(row.factor) for (c,d),row in bars.iterrows()}
    prediction=pd.read_csv(folder/'predictions.csv')
    # Keep every scored stock for ranking; eligibility applies to new buys only.
    scores=dict(zip(prediction.code,prediction.prediction.astype(float)))
    closes=dict(zip(prediction.code,prediction.raw_close.astype(float)))
    eligible=set(prediction.loc[prediction.eligible.eq(True),'code'])
    return dict(day=day,predictions=scores,closes=closes,factors=factors,
                eligible=eligible,plan=plan,plan_sha256=digest(folder/'plan.json'))


def position_details(broker):
    frame=broker.position_cache.get(True)
    result={}
    for row in frame.to_dict('records'):
        if float(row['qty'])<=0:continue
        cost=row.get('average_cost',row.get('cost_price'))
        result[row['code']]=dict(qty=int(row['qty']),can_sell_qty=int(row['can_sell_qty']),
            average_cost=float(cost),cost_price_valid=bool(row['cost_price_valid']))
    return result


def build_manifest(broker,signal,previous,state,now,preview=False):
    from paper_futu.execute_cn_plan import fresh_quotes
    positions=position_details(broker);funds=broker.funds(True)
    if any(o['order_status'] not in TERMINAL for o in broker.orders(True)):
        raise ValueError('Other active orders exist; cannot migrate or compete')
    state=state or bootstrap(positions,funds['total_assets'],signal)
    ordered=sorted(signal['predictions'],key=lambda c:(-signal['predictions'][c],c))
    codes=sorted(set(ordered)|set(positions))
    quotes=fresh_quotes(broker.quote_context,codes,execution=False).to_dict('index')
    eligible=set();sellable=set()
    for c,q in quotes.items():
        try:
            if preview:
                price=float(q['last_price']);prev=float(q['prev_close_price'])
                if q['suspension'] or price<=0 or prev<=0:continue
            else:price,prev,_=checked_quote(c,q,now)
            if c in signal['eligible'] and price/prev-1<.095:eligible.add(c)
            if positions.get(c,{}).get('can_sell_qty',0)>=positions.get(c,{}).get('qty',1) and price/prev-1>-.095:
                sellable.add(c)
        except (ValueError,TypeError):continue
    decision=decide(state,signal,previous['predictions'],positions,eligible,sellable,now.date().isoformat())
    nav=funds['total_assets']
    references={c:float(quotes[c]['last_price']) for c in set(positions)|set(decision['new_codes'])}
    if any(not (0<v<float('inf')) for v in references.values()):raise ValueError('Invalid reference prices')
    targets={c:(0 if c in decision['reasons'] else int(r['qty'])) for c,r in positions.items()}
    for c in decision['new_codes']:
        targets[c]=affordable_buy_quantity(c,nav*.095,references[c])
    actions=[]
    for c,target in targets.items():
        held=positions.get(c,{}).get('qty',0)
        if target!=held:
            actions.append(dict(code=c,side='BUY' if target>held else 'SELL',qty=abs(target-held),
                target=target,held=held,weight=.095,reference_price=references[c]))
    actions.sort(key=lambda a:(a['side']!='SELL',decision['ranking'].get(a['code'],9999),a['code']))
    if decision['transition']:
        # One initial allocation establishes the new Top10 sizing. Subsequent
        # days never rebalance retained quantities. Inherited cost is not reset.
        from paper_futu.cn_integer_targets import allocate_targets
        if len(decision['selected'])!=10:
            raise ValueError('Initial transition cannot establish ten legal positions')
        allocation=allocate_targets(decision['selected'],{c:.095 for c in decision['selected']},nav,
            positions,quotes,funds['cash']-funds.get('frozen_cash',0),.95)
        targets,actions,references=allocation['targets'],allocation['actions'],allocation['reference_prices']
        decision['initial_allocation']=True
    day=now.date().isoformat(); start=now.replace(hour=9,minute=31,second=0,microsecond=0)
    end=start+timedelta(minutes=20)
    manifest=dict(version=1,kind='rank80_full10_5',environment='SIMULATE',execution_date=day,
        signal_date=signal['day'],model_sha256=signal['plan']['model_sha256'],
        signal_sha256=signal['plan_sha256'],previous_signal_sha256=previous['plan_sha256'],
        batch_id=hashlib.sha256((STRATEGY+day+signal['plan_sha256']).encode()).hexdigest()[:24],
        nav=nav,initial_cash=funds['cash'],initial_quantities={c:r['qty'] for c,r in positions.items()},
        top_k=10,risk_degree=.95,not_before=start.isoformat(),expires_at=end.isoformat(),
        targets=targets,actions=actions,reference_prices=references,decision=decision,
        factors={c:signal['factors'].get(c,state['positions'].get(c,{}).get('factor')) for c in references},
        strategy_id=STRATEGY,rules=RULES,
        active_protocol_sha256=read(ROOT/'configs/futu_cn_forward_protocol.json')['protocol_sha256'])
    return manifest


def execute(preview=False, tomorrow=False):
    import pandas as pd
    if not enabled():raise ValueError('Rank80 execution is not active')
    protocol=verify_active_protocol()
    if protocol.get('execution_rules')!=RULES:raise ValueError('Frozen strategy rules differ')
    now=china_now();cal=dates();today=now.date().isoformat()
    if tomorrow:
        if not preview:raise ValueError('Tomorrow is preview-only')
        day=next(d for d in cal if d>today)
        now=now.fromisoformat(day+'T09:31:00+08:00')
    else:day=today
    if day not in cal:raise ValueError('Not a verified trading session')
    i=cal.index(day);signal=load_signal(cal[i-1]);previous=load_signal(cal[i-2])
    if signal['plan']['execution_date']!=day:raise ValueError('Stale signal execution date')
    if not preview and not ((9,31)<=(now.hour,now.minute)<(9,51)):
        raise ValueError('Outside 09:31-09:51; never chase an expired signal')
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with account_lock():
        # An unfinished earlier batch must be resolved, never silently discarded.
        for p in DIRECTORY.glob('*/manifest.json'):
            if p.parent.name<day and not (p.parent/'committed.json').exists():
                raise ValueError('Prior unfinished batch needs reconciliation: '+p.parent.name)
        folder=DIRECTORY/day;done=folder/'committed.json'
        if done.exists():return dict(stage='already_completed',date=day,orders_submitted=0)
        result_path=folder/'execution.result.json'
        if result_path.exists() and read(result_path).get('status')=='reconciliation_required':
            raise ValueError('Ambiguous prior execution; manual reconciliation required')
        state_path=DIRECTORY/'state.json';state=read(state_path) if state_path.exists() else None
        broker=FutuBroker()
        try:
            manifest_path=folder/'manifest.json'
            if manifest_path.exists():
                manifest=read(manifest_path)
                if (manifest['signal_sha256']!=signal['plan_sha256'] or manifest['previous_signal_sha256']!=previous['plan_sha256']
                    or manifest['active_protocol_sha256']!=read(ROOT/'configs/futu_cn_forward_protocol.json')['protocol_sha256']):
                    raise ValueError('Frozen execution inputs changed')
            else:manifest=build_manifest(broker,signal,previous,state,now,preview)
            if preview:
                save_result(DIRECTORY/'preview.json',dict(preview_only=True,observed_at=china_now().isoformat(),manifest=manifest))
                return dict(stage='preview_only',execution_date=day,held=len(manifest['initial_quantities']),
                    sell_orders=sum(a['side']=='SELL' for a in manifest['actions']),
                    buy_orders=sum(a['side']=='BUY' for a in manifest['actions']),orders_submitted=0)
            broker.session(day,signal['plan']['model_year'])
            if not manifest_path.exists():write_once(manifest_path,manifest)
            if result_path.exists():result=read(result_path)
            else:result=run_manifest(broker,manifest,folder/'orders.jsonl',result_path)
            if result['status']=='reconciliation_required':raise ValueError(result.get('error') or 'Unresolved broker state')
            receipts_path=folder/result['order_receipts_file']
            if digest(receipts_path)!=result['order_receipts_sha256']:raise ValueError('Receipt export changed')
            receipts=pd.read_csv(receipts_path,dtype={'order_id':str}).to_dict('records')
            actual={c:r['qty'] for c,r in broker.positions(True).items()}
            after=apply_fills(manifest['decision'],receipts,actual,day,cal,manifest['factors'],manifest['nav'])
            save_result(state_path,after)
            write_once(done,dict(status=result['status'],time=china_now().isoformat(),
                manifest_sha256=digest(manifest_path),result_sha256=digest(result_path),state_after=after))
            return dict(stage=result['status'],date=day,orders_submitted=result.get('submitted_orders',0),holdings=len(actual))
        finally:broker.close()


def tick():
    now=china_now();today=now.date().isoformat();minute=now.hour*60+now.minute
    base=dict(time=now.isoformat(),pid=os.getpid(),strategy_id=STRATEGY,environment='SIMULATE')
    if not enabled():return dict(base,stage='not_active')
    cal=dates()
    if today>cal[-1]:raise ValueError('Trading calendar expired')
    if today not in cal or minute<560 or minute>=991:return dict(base,stage='outside_session')
    if (STATE_DIR/'stop_requested').exists():return dict(base,stage='stopped')
    # Existing independent close/logon tasks update data. Also catch up before
    # order evaluation after a late boot; update job itself has an OS lock.
    from paper_futu.update_cn_data import completed_date,calendar
    latest=completed_date(now,calendar())
    ready=STATE_DIR/latest/'independent_data_ready.json'
    if not ready.exists():
        retry=DIRECTORY/'data_retry.json';last=read(retry).get('time') if retry.exists() else None
        if not last or (now-now.fromisoformat(last)).total_seconds()>=600:
            save_result(retry,dict(time=now.isoformat()))
            log=DIRECTORY/'data_update.log'
            with log.open('a',encoding='utf-8') as f:
                subprocess.run([MODEL_PYTHON,'-X','utf8','-m','paper_futu.update_cn_data','--catch-up'],
                    cwd=ROOT,stdout=f,stderr=f,timeout=1800,creationflags=subprocess.CREATE_NO_WINDOW)
        return dict(base,stage='waiting_for_verified_data',signal_date=latest)
    if minute>=591:return dict(base,stage='window_closed_next_session_only',latest_data=latest)
    if minute<571:return dict(base,stage='waiting_for_0931',latest_data=latest)
    if (DIRECTORY/today/'committed.json').exists():return dict(base,stage='already_completed',latest_data=latest)
    blocked=DIRECTORY/today/'blocked.json'
    if blocked.exists():return dict(base,stage='requires_review',detail=read(blocked))
    try:
        with socket.create_connection(('127.0.0.1',11111),timeout=2):pass
    except OSError:return dict(base,stage='waiting_for_opend_login')
    try:return dict(base,**execute())
    except Exception as exc:
        # Login failure before any durable batch is retryable; any created batch
        # preserves intents and requires review after an explicit error.
        if (DIRECTORY/today/'manifest.json').exists():
            save_result(blocked,dict(time=now.isoformat(),error=str(exc)))
            return dict(base,stage='requires_review',error=str(exc))
        return dict(base,stage='readiness_blocked',error=str(exc))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--tomorrow',action='store_true')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args(argv)
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    if sys.stdout is None:sys.stdout=(DIRECTORY/'worker.stdout.log').open('a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=(DIRECTORY/'worker.stderr.log').open('a',encoding='utf-8',buffering=1)
    if args.preview or args.submit:
        print(json.dumps(execute(preview=not args.submit,tomorrow=args.tomorrow),ensure_ascii=False));return
    # Scheduler invokes --once at logon/every minute, or daily in-session loop.
    import msvcrt
    with (DIRECTORY/'worker.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:return
        while True:
            try:status=tick()
            except Exception as exc:status=dict(stage='error',time=china_now().isoformat(),pid=os.getpid(),error=str(exc))
            save_result(DIRECTORY/'status.json',status)
            if args.once or status['stage'] in ('outside_session','not_active','stopped'):return
            time.sleep(30)


if __name__=='__main__':main()
