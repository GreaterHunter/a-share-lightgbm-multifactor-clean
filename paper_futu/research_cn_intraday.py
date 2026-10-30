"""Bounded A-share minute-feature experiment; isolated from V10 production.

python -m paper_futu.research_cn_intraday --stage all
Stages: audit/features/train/backtest/report. No broker or scheduler operations.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.prepare_cn_intraday_public import ROOT, OUT, SOURCE, START, END, save, digest
from scripts.review_cn_intraday_coverage import binary_series
from src.cn_intraday_features import FEATURE_NAMES, ENHANCED_FEATURE_NAMES, minute_features_batch, attach_previous_daily

V2=OUT/'normalization_v2'
RUN=OUT/'research'
CHECKPOINTS=(585,645,825,885)  # 09:45,10:45,13:45,14:45, selected before results
TEST_START='2022-01-04'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def calendars():
    cal=pd.DatetimeIndex(pd.to_datetime((SOURCE/'provider/calendars/day.txt').read_text().splitlines()))
    cal=cal[cal<=END]
    selected=cal[cal>=START]
    minutes=np.r_[np.arange(575,691,5),np.arange(785,901,5)]
    grid=pd.DatetimeIndex((selected.to_numpy()[:,None]+minutes.astype('timedelta64[m]')[None,:]).ravel(),name='datetime')
    return cal,selected,grid


def hashes():
    files=['paper_futu/research_cn_intraday.py','paper_futu/cn_intraday_account.py',
           'src/cn_intraday_features.py','src/cn_intraday_source.py','paper_futu/research_rank_buffer_futu_fees.py']
    return {p:digest(ROOT/p) for p in files}


def verify():
    p=read(RUN/'protocol.json')
    if p['implementation']!=hashes():raise RuntimeError('Implementation binding changed')
    for f,h in p['protected'].items():
        if digest(f)!=h:raise RuntimeError('Protected production artifact changed')
    return p


def audit():
    complete=read(V2/'completion.json')
    if len(complete['records'])!=658:raise RuntimeError('Incomplete normalized data')
    original=read(OUT/'data_protocol.json')
    if original['absent_source_instruments']:raise RuntimeError('Historical securities missing')
    cal,days,grid=calendars()
    members={}
    for line in (SOURCE/'provider/instruments/csi300.txt').read_text().splitlines():
        code,a,b=line.split();members.setdefault(code,[]).append((a,b))
    bindings={}
    for year in range(2022,2026):
        prior_cutoff=cal[cal<pd.Timestamp(year,1,1)].max().strftime('%Y-%m-%d')
        path=ROOT/f'models/experiments/s6_quarterly_l1l2_small_asof_20261013/{prior_cutoff}/metadata.json'
        m=read(path)
        if m['cutoff']>=f'{year}-01-01' or m['label_end_max']>m['cutoff']:raise RuntimeError('Prior model parameter cutoff invalid')
        bindings[str(year)]=dict(trees=m['trees'],parameters=m['parameters'],metadata_sha256=digest(path),
            note='Reuse only original prior-year-known parameter/tree setting; no new tree search for intraday target')
    p=dict(start=TEST_START,end=END,training_start=START,features=FEATURE_NAMES,
        checkpoint_minutes=list(CHECKPOINTS),prediction_every_minutes=5,horizon_sessions=10,
        label='adjusted open at execution bar + 10 trading sessions / adjusted open at execution bar - 1',
        execution='signal bar index + 2, fill at that bar open; one full bar intervenes; no same-bar fills',
        models=['daily15','minute29'],annual_parameter_bindings=bindings,quarterly_from_scratch=True,
        target='cross-sectional future-return percentile rank minus 0.5 at each sampled timestamp',
        rules=dict(top_k=10,rank_exit=80,strong_rank=1,tp=.10,sl=.05,tp_cooldown=0,sl_cooldown=1,
                   stock_budget=.95,t_plus_one=True,participation_cap=.05,retained_weights='drift'),
        frictions=[0.,.001],rf_annual=.01,threads=4,
        benchmark_intraday='510300 ETF proxy explicitly authorized; not CSI300 index intraday',
        benchmark_performance='H00300 total return',implementation=hashes(),protected=original['protected'],
        source_binding_sha256=digest(V2/'protocol.json'),
        limitations=['Public data upstream providers not fully disclosed; data audited but not exchange-certified.',
          'Zero-trade/missing OHLC cannot fill; unavailable quotes never filled using future prices.',
          'Historical financial revisions lack a full vintage archive.',
          'Adjusted inventory models corporate actions synthetically, not a raw-share dividend cash ledger.',
          'Uniform +/-9.5% directional filter retained; not full historical board/ST limit reconstruction.',
          'Bar-open executions capped by 5% of subsequent bar volume; queue priority and market impact unknown.',
          'Old pre-2018 Shanghai final trade may differ from official weighted closing price; official daily close used for daily NAV.',
          'Daily four-time sampling controls training size; serves every five minutes. No validation-driven new parameter search.',
          '2022-2025 repeatedly studied; not an untouched independent test.'])
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():
        if read(RUN/'protocol.json')!=p:raise RuntimeError('Research protocol changed')
    else:save(RUN/'protocol.json',p)
    for row in complete['records']:
        for f,h in row['artifacts'].items():
            if digest(f)!=h:raise RuntimeError('Normalized artifact changed')
    verify()
    save(RUN/'audit_complete.json',dict(status='passed_with_disclosed_limits',records=658,
        normalization_completion_sha256=digest(V2/'completion.json'),data_cutoff=END))


def feature_stock(code,daily,benchmark,cal,days,grid,intervals):
    bars=pd.read_parquet(V2/'bars_5m'/f'{code}.parquet')
    minute=minute_features_batch(bars,benchmark,bar_minutes=5).reindex(grid)
    # Only previous-session feature values; missing minute prices are not filled.
    f=attach_previous_daily(minute,daily,cal)
    factor=binary_series(SOURCE/'provider',code,'factor',cal).reindex(grid.normalize()).to_numpy()
    q=bars.reindex(grid)
    op=q.open.to_numpy()*factor;close=q.close.to_numpy()*factor
    positions=np.arange(len(grid));entry=positions+2;exit_=entry+10*48
    label=np.full(len(grid),np.nan);ends=np.full(len(grid),np.datetime64('NaT'),dtype='datetime64[ns]')
    valid=exit_<len(grid);i=np.flatnonzero(valid)
    label[i]=op[exit_[i]]/op[entry[i]]-1;ends[i]=grid.to_numpy()[exit_[i]]
    f['label']=label;f['label_end']=ends
    active=np.zeros(len(days),dtype=bool)
    for a,b in intervals:active|=(days>=a)&(days<=b)
    active=np.repeat(active,48)
    # Membership is from dated intervals, not the future label's availability.
    f=f.loc[active].copy()
    f['instrument']=code
    for c in FEATURE_NAMES+['label']:f[c]=f[c].astype('float32')
    sample=np.isin(f.index.hour*60+f.index.minute,CHECKPOINTS)
    training=f.loc[sample].replace([np.inf,-np.inf],np.nan)
    predict=f.loc[f.index>=TEST_START,FEATURE_NAMES+['instrument']]
    market=pd.DataFrame(dict(open=op,close=close,factor=factor,volume=q.volume.to_numpy()),index=grid)
    return training,predict,market.loc[grid>=TEST_START]


def features():
    verify()
    if (RUN/'features_complete.json').exists():return
    cal,days,grid=calendars()
    benchmark=pd.read_parquet(V2/'bars_5m/SH510300.parquet')[['open','close']]
    daily=pd.read_parquet(SOURCE/'features.parquet')
    groups={c:g.droplevel('instrument') for c,g in daily.groupby(level='instrument')};del daily
    members={}
    for line in (SOURCE/'provider/instruments/csi300.txt').read_text().splitlines():
        code,a,b=line.split();members.setdefault(code,[]).append((a,b))
    codes=[r['instrument'] for r in read(V2/'completion.json')['records'] if r['instrument']!='SH510300']
    for folder in ('train_parts','predict_parts','market_parts','feature_checks'):(RUN/folder).mkdir(exist_ok=True)
    for n,code in enumerate(codes):
        check=RUN/'feature_checks'/f'{code}.json'
        if check.exists():
            r=read(check)
            for path,h in r['artifacts'].items():
                if digest(path)!=h:raise RuntimeError('Feature cache changed')
            continue
        t,p,m=feature_stock(code,groups[code],benchmark,cal,days,grid,members[code])
        t.to_parquet(RUN/'train_parts'/f'{code}.parquet');p.to_parquet(RUN/'predict_parts'/f'{code}.parquet')
        m.to_parquet(RUN/'market_parts'/f'{code}.parquet')
        files=[RUN/k/f'{code}.parquet' for k in ('train_parts','predict_parts','market_parts')]
        save(check,dict(instrument=code,train_rows=len(t),prediction_rows=len(p),
            artifacts={str(path):digest(path) for path in files}))
        if n%10==0:print(f'FEATURES {n+1}/{len(codes)} {code}',flush=True)
        save(RUN/'progress.json',dict(stage='features',completed=n+1,total=len(codes),last=code,time=time.time()))
        del t,p,m;gc.collect()
    save(RUN/'features_complete.json',dict(status='complete',stocks=len(codes)))


def train():
    import lightgbm as lgb
    verify()
    if (RUN/'train_complete.json').exists():return
    train=pd.concat([pd.read_parquet(p) for p in sorted((RUN/'train_parts').glob('*.parquet'))])
    train=train.dropna(subset=['label','label_end']).sort_index()
    train['target']=train.label.groupby(level=0).rank(method='average',pct=True)-.5
    if (train.label_end<=train.index).any():raise RuntimeError('Label end before signal')
    protocol=verify();records=[]
    for year in range(2022,2026):
        for quarter in range(1,5):
            start=pd.Timestamp(year,3*(quarter-1)+1,1)
            end=min(start+pd.offsets.QuarterEnd(0),pd.Timestamp(END))
            cutoff=start-pd.Timedelta(nanoseconds=1)
            eligible=train.label_end.le(cutoff)&(train.index<=cutoff)
            selected=train.loc[eligible]
            binding=protocol['annual_parameter_bindings'][str(year)]
            for variant,names in [('daily15',ENHANCED_FEATURE_NAMES),('minute29',FEATURE_NAMES)]:
                directory=RUN/'models'/variant/f'{year}Q{quarter}';directory.mkdir(parents=True,exist_ok=True)
                metadata=directory/'metadata.json'
                if metadata.exists():
                    r=read(metadata)
                    if digest(directory/'model.txt')!=r['model_sha256']:raise RuntimeError('Model changed')
                else:
                    params={**binding['parameters'],'n_estimators':binding['trees'],'random_state':42,'n_jobs':4}
                    print(f'TRAIN {variant} {year}Q{quarter} rows={len(selected)} trees={binding["trees"]}',flush=True)
                    model=lgb.LGBMRegressor(**params)
                    model.fit(selected[names],selected.target)
                    model.booster_.save_model(str(directory/'model.txt'))
                    reload=lgb.Booster(model_file=str(directory/'model.txt'))
                    sample=selected[names].tail(1000)
                    err=float(np.max(np.abs(model.predict(sample)-reload.predict(sample,num_threads=4))))
                    if err>1e-10:raise RuntimeError('Model reload inconsistent')
                    r=dict(variant=variant,year=year,quarter=quarter,cutoff=str(cutoff),
                        label_end_max=str(selected.label_end.max()),signal_max=str(selected.index.max()),
                        train_rows=len(selected),trees=binding['trees'],parameters=params,
                        features=names,model_sha256=digest(directory/'model.txt'),reload_error=err)
                    save(metadata,r);del model,reload,sample;gc.collect()
                records.append(r)
                save(RUN/'progress.json',dict(stage='train',completed=len(records),total=32,time=time.time()))
    del train;gc.collect()
    save(RUN/'train_complete.json',dict(status='complete',models=records))


def backtest():
    import lightgbm as lgb
    from paper_futu.cn_intraday_account import Account
    verify()
    if (RUN/'backtest_complete.json').exists():return
    cal,days,grid=calendars();test_days=days[days>=TEST_START]
    testgrid=grid[grid>=TEST_START]
    accounts={(variant,bp):Account(bp/10000) for variant in ('daily15','minute29') for bp in (0,10)}
    codes=[p.stem for p in sorted((RUN/'market_parts').glob('*.parquet'))]
    official={};prior={}
    for c in codes:
        s=binary_series(SOURCE/'provider',c,'close',cal)
        official[c]=s.reindex(test_days).to_numpy()
        prior[c]=s.shift().reindex(test_days).to_numpy()
    # Month-wise materialization bounds RAM. Save signals before each account
    # execution; resuming accounts replays deterministic saved model predictions.
    carry=[];current_period=None;models={};index_of={d:i for i,d in enumerate(test_days)}
    for month in pd.period_range(TEST_START,END,freq='M'):
        lo=month.start_time;hi=month.end_time
        prediction_parts=[];market_parts=[]
        for c in codes:
            pred=pd.read_parquet(RUN/'predict_parts'/f'{c}.parquet',filters=[('datetime','>=',lo),('datetime','<=',hi)])
            if len(pred):prediction_parts.append(pred)
            m=pd.read_parquet(RUN/'market_parts'/f'{c}.parquet',filters=[('datetime','>=',lo),('datetime','<=',hi)])
            m['instrument']=c;market_parts.append(m)
        x=pd.concat(prediction_parts);market=pd.concat(market_parts)
        period=(lo.year,lo.quarter)
        if period!=current_period:
            models={v:lgb.Booster(model_file=str(RUN/'models'/v/f'{period[0]}Q{period[1]}'/'model.txt')) for v in ('daily15','minute29')}
            current_period=period
        predictions=x[['instrument']].copy()
        for v,names in [('daily15',ENHANCED_FEATURE_NAMES),('minute29',FEATURE_NAMES)]:
            predictions[v]=models[v].predict(x[names],num_threads=4)
            if not np.isfinite(predictions[v]).all():raise RuntimeError('Nonfinite predictions')
        (RUN/'predictions').mkdir(exist_ok=True)
        predictions.to_parquet(RUN/'predictions'/f'{month}.parquet')
        pgroup=dict(tuple(predictions.groupby(level=0)));mgroup=dict(tuple(market.groupby(level=0)))
        del x,prediction_parts,market_parts,market,predictions;gc.collect()
        times=testgrid[(testgrid>=lo)&(testgrid<=hi)]
        for ts in times:
            m=mgroup[ts].set_index('instrument');quotes=m.to_dict('index')
            # Signal index + 2: the intervening bar is not a fill opportunity.
            if len(carry)>=2:
                signal,sc,sp=carry.pop(0)
                execution=ts-pd.Timedelta(minutes=5)
                di=index_of[ts.normalize()];previous={c:prior[c][di] for c in codes}
                for (v,bp),account in accounts.items():account.step(signal,execution,sc[v],sp,quotes,previous)
            p=pgroup.get(ts)
            if p is None:raise RuntimeError(f'Missing cross-section {ts}')
            scores={v:dict(zip(p.instrument,p[v])) for v in ('daily15','minute29')}
            carry.append((ts,scores,dict(zip(m.index,m.close))))
            if ts.hour==15:
                di=index_of[ts.normalize()];closes={c:official[c][di] for c in codes}
                for account in accounts.values():account.close_day(ts.normalize(),closes)
        save(RUN/'progress.json',dict(stage='backtest',last_month=str(month),time=time.time(),
            nav={f'{v}_{bp}bp':a.nav() for (v,bp),a in accounts.items()}))
        print(f'BACKTEST {month} '+str({f'{v}_{bp}bp':round(a.nav(),2) for (v,bp),a in accounts.items()}),flush=True)
        del pgroup,mgroup;gc.collect()
    results=[]
    for (v,bp),a in accounts.items():
        directory=RUN/'accounts'/f'{v}_{bp}bp';directory.mkdir(parents=True,exist_ok=True)
        check=a.reconcile()
        for name,rows in [('daily',a.daily),('orders',a.orders),('holdings',a.holdings),('closed',a.closed)]:pd.DataFrame(rows).to_parquet(directory/f'{name}.parquet',index=False)
        save(directory/'reconciliation.json',check)
        results.append(dict(variant=v,friction_bp=bp,**check))
    save(RUN/'backtest_complete.json',dict(status='complete',accounts=results))


def report():
    from paper_futu.top5_buffer_report import metrics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify();rows=[];annual=[]
    b=pd.read_parquet(SOURCE/'benchmark.parquet').benchmark_return
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for variant in ('daily15','minute29'):
        for bp in (0,10):
            directory=RUN/'accounts'/f'{variant}_{bp}bp';d=pd.read_parquet(directory/'daily.parquet')
            d['date']=pd.to_datetime(d.date)
            if not pd.DatetimeIndex(d.date).equals(b.index):raise RuntimeError('Benchmark period mismatch')
            orders=pd.read_parquet(directory/'orders.parquet');closed=pd.read_parquet(directory/'closed.parquet')
            row=dict(variant=variant,friction_bp=bp,**metrics(d,b),fills=len(orders),closed_trades=len(closed),
                win_rate=float((closed.pnl>0).mean()) if len(closed) else None,
                average_win=float(closed.loc[closed.pnl>0,'return_on_cost'].mean()) if (closed.pnl>0).any() else None,
                average_loss=float(closed.loc[closed.pnl<0,'return_on_cost'].mean()) if (closed.pnl<0).any() else None)
            rows.append(row)
            for y,g in d.groupby(d.date.dt.year):annual.append(dict(variant=variant,friction_bp=bp,year=int(y),**metrics(g,b)))
            axes[0].plot(d.date,d.nav/1e6,label=f'{variant} +{bp}bp')
            nav=d.nav.to_numpy()/1e6;dd=nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1
            axes[1].plot(d.date,dd,label=f'{variant} +{bp}bp')
    axes[0].plot(b.index,(1+b).cumprod(),label='H00300',color='black',linestyle='--')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Net asset value');axes[1].set_ylabel('Drawdown')
    fig.tight_layout();fig.savefig(RUN/'curves.png',dpi=150);plt.close(fig)
    table=pd.DataFrame(rows);years=pd.DataFrame(annual)
    table.to_csv(RUN/'comparison.csv',index=False,encoding='utf-8-sig');years.to_csv(RUN/'yearly.csv',index=False,encoding='utf-8-sig')
    shown=table[['variant','friction_bp','total_return','cagr','max_drawdown','annual_geometric_excess','sharpe','information_ratio','fills','fees','win_rate']].copy()
    for c in ['total_return','cagr','max_drawdown','annual_geometric_excess','win_rate']:shown[c]=shown[c].map(lambda x:f'{x:.2%}')
    yearly=years[['variant','friction_bp','year','total_return','max_drawdown','geometric_excess']].copy()
    for c in ['total_return','max_drawdown','geometric_excess']:yearly[c]=yearly[c].map(lambda x:f'{x:.2%}')
    limitations=''.join(f'<li>{x}</li>' for x in verify()['limitations'])
    page='<html lang="zh"><meta charset="utf-8"><title>A股分钟特征离线实验</title><style>body{font:16px/1.7 system-ui;max-width:1300px;margin:30px auto}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}img{max-width:100%}</style><h1>A股分钟特征：15因子与29因子对照</h1><p>训练2015起，回放2022—2025；盘中参照510300 ETF，绩效基准H00300。季度重训，每日09:45、10:45、13:45、14:45训练抽样，每5分钟评估；用信号后隔一根5分钟线的开盘到10个交易日后相同时点的复权收益排名作标签。沿用各年原S6事前确定参数及树数，没有重新寻优。</p><p>Top10、排名80退出、新进第1名替换，止盈10%、止损5%，全部退出；止盈无额外冷却，止损次日可入。保留T+1，最多10股，初始每股账户9.5%，保留股不配平。零成交无法成交，单根成交量5%上限，额外10bp为每边压力。费用已计入净收益。</p>'+shown.to_html(index=False)+'<h2>逐年</h2>'+yearly.to_html(index=False)+'<img src="curves.png"><h2>限制</h2><ul>'+limitations+'</ul><p>这是历史探索实验，不切换V10，不宣称日内Alpha已经得到独立验证。四组只有分钟特征与摩擦不同，不代表原日线部署模型的复算。<a href="comparison.csv">完整指标</a> · <a href="yearly.csv">逐年结果</a> · <a href="protocol.json">固定口径</a></p></html>'
    (RUN/'report.html').write_text(page,encoding='utf8')
    save(RUN/'completion.json',dict(status='complete',accounts=4,models=32,protocol_sha256=digest(RUN/'protocol.json'),
        artifacts={str(p):digest(p) for p in [RUN/'report.html',RUN/'comparison.csv',RUN/'yearly.csv',RUN/'curves.png']}))
    print(shown.to_string(index=False),flush=True)


def main():
    import msvcrt
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--stage',choices=['audit','features','train','backtest','report','all'],default='all');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Research already running')
        started=time.time()
        try:
            for stage in ('audit','features','train','backtest','report'):
                if args.stage in ('all',stage):
                    save(RUN/'status.json',dict(status='running',stage=stage,pid=os.getpid(),started=started,updated=time.time()))
                    globals()[stage]()
            save(RUN/'status.json',dict(status='complete' if (RUN/'completion.json').exists() else 'stage_complete',stage=args.stage,pid=os.getpid(),elapsed_seconds=time.time()-started))
        except Exception as exc:
            save(RUN/'status.json',dict(status='failed',stage=stage,pid=os.getpid(),error_type=type(exc).__name__,error=str(exc)))
            raise


if __name__=='__main__':main()
