"""Six fixed-budget input ablations and ranking cadence/delay checks.

python -m paper_futu.research_cn_minute_ablation --stage all
No production changes; no broker access or parameter grid search.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
from paper_futu.research_cn_intraday import (
    ROOT, RUN as BASE, SOURCE, END, TEST_START, read, save, digest, calendars,
    verify as verify_original, binary_series, ENHANCED_FEATURE_NAMES, FEATURE_NAMES,
)
from paper_futu.cn_intraday_ablation_account import AblationAccount

OUT=BASE.parent.parent/'cn_minute_factor_ablation_asof_20251231'
STOP=BASE.parent.parent/'cn_intraday_stop4_asof_20251231'
DAILY=list(ENHANCED_FEATURE_NAMES)
CLOCK=DAILY+['min_session_progress']
FEATURES={
    'M0_daily':DAILY,
    'M1_clock':CLOCK,
    'M2_return':CLOCK+[f'min_return_{n}' for n in (5,15,60)]+[f'min_relative_return_{n}' for n in (5,15,60)],
    'M3_volatility':CLOCK+[f'min_volatility_{n}' for n in (15,60)]+[f'min_amplitude_{n}' for n in (15,60)],
    'M4_position':CLOCK+['min_volume_ratio_5_30','min_vwap_deviation','min_return_since_open'],
    'M5_full':list(FEATURE_NAMES),
}
REUSED={'M0_daily':'daily15','M5_full':'minute29'}
NEW=[v for v in FEATURES if v not in REUSED]


def account_configs():
    rows=[dict(model=m,cadence='5m',lag=2,bp=bp) for m in FEATURES for bp in (0,10)]
    rows += [dict(model=m,cadence=c,lag=2,bp=bp) for m in ('M0_daily','M1_clock','M5_full') for c in ('30m','daily') for bp in (0,10)]
    rows += [dict(model=m,cadence='5m',lag=3,bp=bp) for m in ('M1_clock','M5_full') for bp in (0,10)]
    assert len(rows)==28
    return rows


def account_key(c):return f'{c["model"]}_{c["cadence"]}_lag{c["lag"]}_{c["bp"]}bp'


def model_dir(m,y,q):
    return (BASE/'models'/REUSED[m] if m in REUSED else OUT/'models'/m)/f'{y}Q{q}'


def implementation():
    return {str(ROOT/p):digest(ROOT/p) for p in ('paper_futu/research_cn_minute_ablation.py',
        'paper_futu/cn_intraday_ablation_account.py','scripts/report_cn_minute_ablation.py',
        'scripts/audit_cn_intraday_results.py','paper_futu/top5_buffer_report.py')}


def verify():
    verify_original();p=read(OUT/'protocol.json')
    if p['implementation']!=implementation():raise RuntimeError('Implementation changed')
    for path,h in p['inputs'].items():
        if digest(path)!=h:raise RuntimeError(f'Source changed: {path}')
    return p


def audit():
    from paper_futu.research_cn_intraday_stop4 import verify as verify_stop
    verify_stop()
    if read(STOP/'completion.json')['status']!='complete_and_audited':raise RuntimeError('Prior experiment not audited')
    inputs=dict(read(STOP/'protocol.json')['inputs'])
    for f in (BASE/'feature_checks').glob('*.json'):
        inputs.update(read(f)['artifacts'])
    for path,h in read(STOP/'completion.json')['artifacts'].items():inputs[path]=h
    inputs[str(STOP/'protocol.json')]=digest(STOP/'protocol.json')
    for f in (STOP/'accounts').glob('stop4_*/*'):
        if f.is_file():inputs[str(f)]=digest(f)
    for path,h in inputs.items():
        if digest(path)!=h:raise RuntimeError('Input hash mismatch')
    protocol=dict(start=TEST_START,end=END,training_start='2015-01-05',features=FEATURES,
        training='Quarterly expanding mature-label fit; exact same rows, target ranks, seed42 and annual S6 parameters/tree counts across inputs. Four fixed timestamps/day in fitting.',
        annual_parameter_bindings=read(BASE/'protocol.json')['annual_parameter_bindings'],
        tree_selection='No new tree selection or tuning. Conditional fixed-budget ablation; cannot conclude minute inputs fail under every re-tuned capacity or horizon.',
        accounts=account_configs(),new_fits=64,reused_fits=32,threads=4,
        rules=dict(top_k=10,rank_exit=80,strong_rank=1,tp=.10,stop_loss=.04,tp_cooldown=0,sl_cooldown=1,t_plus_one=True),
        cadence='Only rank exit / strong replacement and their previous-rank reference are throttled. TP/SL and vacant-slot entries still every five minutes; daily rank check at09:45;30m at10:00/10:30/11:00/11:30/13:30/14:00/14:30/15:00.',
        lag='lag2 is original: one completed five-minute bar between signal and execution bar open. lag3 adds one more five-minute bar.',
        diagnostics='Full-grid mature H10 Rank IC, daily aggregation, quarterly/yearly results, top10 consecutive overlap, paired circular block bootstrap lengths60/120 days,2000 replications seed20261030.',
        primary_reference='M1_clock removes session-progress confounding; M0_daily also reported; no model selected as a new champion.',
        inputs=inputs,implementation=implementation(),limitations=read(BASE/'protocol.json')['limitations'],
        no_production_changes=True)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if read(OUT/'protocol.json')!=protocol:raise RuntimeError('Protocol mismatch')
    else:save(OUT/'protocol.json',protocol)
    save(OUT/'audit_complete.json',dict(status='passed',inputs=len(inputs)))


def train():
    import lightgbm as lgb
    p=verify()
    if (OUT/'train_complete.json').exists():return
    train=pd.concat([pd.read_parquet(f) for f in sorted((BASE/'train_parts').glob('*.parquet'))])
    train=train.dropna(subset=['label','label_end']).sort_index()
    train['target']=train.label.groupby(level=0).rank(method='average',pct=True)-.5
    records=[]
    for year in range(2022,2026):
        for q in range(1,5):
            cutoff=pd.Timestamp(year,3*(q-1)+1,1)-pd.Timedelta(nanoseconds=1)
            selected=train.loc[train.label_end.le(cutoff)&(train.index<=cutoff)]
            binding=p['annual_parameter_bindings'][str(year)]
            for m,names in FEATURES.items():
                folder=model_dir(m,year,q);metadata=folder/'metadata.json'
                if metadata.exists():
                    r=read(metadata)
                    if digest(folder/'model.txt')!=r['model_sha256']:raise RuntimeError('Model cache changed')
                    if r['train_rows']!=len(selected) or r['features']!=names or pd.Timestamp(r['cutoff'])!=cutoff:raise RuntimeError('Unequal fitting sample or boundary')
                else:
                    folder.mkdir(parents=True,exist_ok=True)
                    params={**binding['parameters'],'n_estimators':binding['trees'],'random_state':42,'n_jobs':4}
                    print(f'TRAIN {m} {year}Q{q} rows={len(selected)} trees={binding["trees"]}',flush=True)
                    model=lgb.LGBMRegressor(**params).fit(selected[names],selected.target)
                    model.booster_.save_model(str(folder/'model.txt'))
                    loaded=lgb.Booster(model_file=str(folder/'model.txt'));sample=selected[names].tail(1000)
                    error=float(np.max(np.abs(model.predict(sample)-loaded.predict(sample,num_threads=4))))
                    if error>1e-10:raise RuntimeError('Reload mismatch')
                    r=dict(variant=m,year=year,quarter=q,cutoff=str(cutoff),label_end_max=str(selected.label_end.max()),
                        signal_max=str(selected.index.max()),train_rows=len(selected),features=names,parameters=params,
                        trees=binding['trees'],model_sha256=digest(folder/'model.txt'),reload_error=error)
                    save(metadata,r);del model,loaded,sample;gc.collect()
                if pd.Timestamp(r['label_end_max'])>cutoff:raise RuntimeError('Immature train label')
                records.append(dict(model=m,reused=m in REUSED,path=str(folder/'model.txt'),metadata=r))
                save(OUT/'progress.json',dict(stage='train',completed=len(records),total=96,new_completed=sum(not x['reused'] for x in records),new_total=64,time=time.time()))
    del train,selected;gc.collect()
    save(OUT/'train_complete.json',dict(status='complete',models=records))


def replay():
    import lightgbm as lgb
    verify()
    if (OUT/'backtest_complete.json').exists():return
    cfg=account_configs();accounts={account_key(c):AblationAccount(c['bp']/10000,rank_cadence=c['cadence']) for c in cfg}
    cal,days,grid=calendars();days=days[days>=TEST_START];grid=grid[grid>=TEST_START]
    codes=[p.stem for p in sorted((BASE/'market_parts').glob('*.parquet'))];official={};prior={}
    for code in codes:
        s=binary_series(SOURCE/'provider',code,'close',cal);official[code]=s.reindex(days).to_numpy();prior[code]=s.shift().reindex(days).to_numpy()
    date_index={d:i for i,d in enumerate(days)};carry=[];period=None;models={}
    (OUT/'predictions').mkdir(exist_ok=True)
    for month in pd.period_range(TEST_START,END,freq='M'):
        lo,hi=month.start_time,month.end_time;cache=OUT/'predictions'/f'{month}.parquet'
        prediction_parts=[];market_parts=[]
        for code in codes:
            if not cache.exists():
                f=pd.read_parquet(BASE/'predict_parts'/f'{code}.parquet',filters=[('datetime','>=',lo),('datetime','<=',hi)])
                if len(f):prediction_parts.append(f)
            m=pd.read_parquet(BASE/'market_parts'/f'{code}.parquet',filters=[('datetime','>=',lo),('datetime','<=',hi)])
            m['instrument']=code;market_parts.append(m)
        if not cache.exists():
            x=pd.concat(prediction_parts)
            if period!=(lo.year,lo.quarter):
                models={m:lgb.Booster(model_file=str(model_dir(m,lo.year,lo.quarter)/'model.txt')) for m in NEW};period=(lo.year,lo.quarter)
            predictions=pd.read_parquet(BASE/'predictions'/f'{month}.parquet').rename(columns={'daily15':'M0_daily','minute29':'M5_full'})
            if not x.index.equals(predictions.index) or not x.instrument.equals(predictions.instrument):raise RuntimeError('Prediction sample mismatch')
            for m in NEW:predictions[m]=models[m].predict(x[FEATURES[m]],num_threads=4)
            predictions.to_parquet(cache);save(cache.with_suffix('.json'),dict(sha256=digest(cache)))
            del x
        else:
            if read(cache.with_suffix('.json'))['sha256']!=digest(cache):raise RuntimeError('Prediction cache changed')
            predictions=pd.read_parquet(cache)
        if not np.isfinite(predictions[list(FEATURES)]).all().all():raise RuntimeError('Nonfinite score')
        market=pd.concat(market_parts);pg=dict(tuple(predictions.groupby(level=0)));mg=dict(tuple(market.groupby(level=0)))
        del prediction_parts,market_parts,market,predictions;gc.collect()
        for ts in grid[(grid>=lo)&(grid<=hi)]:
            m=mg[ts].set_index('instrument');quotes=m.to_dict('index');di=date_index[ts.normalize()]
            prev={c:prior[c][di] for c in codes}
            for conf in cfg:
                if len(carry)>=conf['lag']:
                    signal,scores,prices=carry[-conf['lag']]
                    accounts[account_key(conf)].step(signal,ts-pd.Timedelta(minutes=5),scores[conf['model']],prices,quotes,prev)
            p=pg[ts];sc={name:dict(zip(p.instrument,p[name])) for name in FEATURES}
            carry.append((ts,sc,dict(zip(m.index,m.close))));carry=carry[-3:]
            if ts.hour==15:
                close={c:official[c][di] for c in codes}
                for a in accounts.values():a.close_day(ts.normalize(),close)
        save(OUT/'progress.json',dict(stage='replay',month=str(month),completed=(month.year-2022)*12+month.month,total=48,time=time.time()))
        print(f'REPLAY {month} '+str({k:round(a.nav(),2) for k,a in accounts.items() if '_5m_lag2_0bp' in k}),flush=True)
        del pg,mg;gc.collect()
    rows=[]
    for conf in cfg:
        key=account_key(conf);a=accounts[key];folder=OUT/'accounts'/key;folder.mkdir(parents=True,exist_ok=True);check=a.reconcile();artifacts={}
        for name,values in [('daily',a.daily),('orders',a.orders),('holdings',a.holdings),('closed',a.closed)]:
            frame=pd.DataFrame(values)
            if conf['model']=='M5_full' and conf['cadence']=='5m' and conf['lag']==2:
                old=pd.read_parquet(STOP/'accounts'/f'stop4_{conf["bp"]}bp'/f'{name}.parquet')
                pd.testing.assert_frame_equal(frame,old,check_exact=True)
            frame.to_parquet(folder/f'{name}.parquet',index=False);artifacts[str(folder/f'{name}.parquet')]=digest(folder/f'{name}.parquet')
        save(folder/'reconciliation.json',check);rows.append(dict(key=key,**conf,artifacts=artifacts,**check))
    save(OUT/'backtest_complete.json',dict(status='complete',accounts=rows))


def diagnose():
    from scripts.report_cn_minute_ablation import diagnose as run
    run()


def report():
    from scripts.report_cn_minute_ablation import report as run
    run()


def main():
    import msvcrt
    p=argparse.ArgumentParser(__doc__);p.add_argument('--stage',choices=['audit','train','replay','diagnose','report','all'],default='all');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        stage='init'
        try:
            for stage in ('audit','train','replay','diagnose','report'):
                if args.stage in ('all',stage):
                    save(OUT/'status.json',dict(status='running',stage=stage,pid=os.getpid(),time=time.time()));globals()[stage]()
            save(OUT/'status.json',dict(status='complete' if (OUT/'completion.json').exists() else 'stage_complete',pid=os.getpid(),time=time.time()))
        except Exception as exc:
            save(OUT/'status.json',dict(status='failed',stage=stage,error=repr(exc),pid=os.getpid(),time=time.time()));raise


if __name__=='__main__':main()
