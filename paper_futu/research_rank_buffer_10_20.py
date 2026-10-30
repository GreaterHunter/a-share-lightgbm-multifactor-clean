"""Six isolated Top10/20 accounts using audited pre-2026 S6 predictions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import time
import traceback

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'

from paper_futu.top5_buffer_data import ROOT,START,END,read,save,digest
from paper_futu.rank_buffer_10_20_engine import Account,Rules

SOURCE = ROOT/'reports/experiments/top5_rank_buffer_asof_20251231'
OUT = ROOT/'reports/experiments/rank_buffer_top10_top20_asof_20251231'


def implementation():
    names = ('research_rank_buffer_10_20.py','rank_buffer_10_20_engine.py','rank_buffer_10_20_report.py',
             'top5_buffer_data.py','top5_buffer_report.py')
    return {str(ROOT/'paper_futu'/n):digest(ROOT/'paper_futu'/n) for n in names}


def verify(files):
    for p,h in files.items():
        if digest(p)!=h: raise ValueError(f'Changed artifact: {p}')


def protocol():
    p=read(OUT/'protocol.json')
    if p['implementation']!=implementation(): raise ValueError('Code changed; review cached artifacts before resume')
    verify(p['protected']);return p


@contextmanager
def lock():
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as e:raise RuntimeError('Duplicate experiment runner') from e
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def audit():
    import numpy as np
    import pandas as pd
    from dataclasses import asdict
    if (OUT/'audit_complete.json').exists():
        p=protocol();verify(p['sources']);return
    complete=read(SOURCE/'completion.json')
    if complete['status']!='complete' or complete['report_manifest_sha256']!=digest(SOURCE/'report_manifest.json'):
        raise ValueError('Source study completion manifest mismatch')
    verify(read(SOURCE/'report_manifest.json')['artifacts'])
    verify(read(SOURCE/'audit_complete.json')['artifacts'])
    source_p=read(SOURCE/'protocol.json')
    verify(source_p['implementation']);verify(source_p['protected'])
    model_audit=read(SOURCE/'models_audit.json')
    if len(model_audit['periods'])!=16:raise ValueError('Require 16 causal quarters')
    sources={str(SOURCE/n):digest(SOURCE/n) for n in (
        'completion.json','report_manifest.json','protocol.json','audit_complete.json','models_audit.json',
        'data_audit.json','review.json','data/predictions.parquet','data/market.parquet',
        'data/benchmark.parquet','data/provider/calendars/day.txt','comparison.csv','yearly.csv')}
    for m in model_audit['periods']:
        if not m['label_end_max']<=m['cutoff']<m['start'] or m['end']>END or not m['reload_equal']:
            raise ValueError('Source training boundary or reload failed')
        sources[m['model_path']]=m['model_sha256']
    verify(sources)
    pred=pd.read_parquet(SOURCE/'data/predictions.parquet')
    dates=pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    if dates.max()!=pd.Timestamp(END):raise ValueError('Unexpected calendar cutoff')
    expected=dates[(dates>=START)&(dates<=END)]
    signals=pred.index.get_level_values('datetime')
    pos=dates.get_indexer(signals)
    if pred.index.has_duplicates or (pos<0).any() or (pos+1>=len(dates)).any():raise ValueError('Invalid signal index')
    if not np.array_equal(dates.to_numpy()[pos+1],pred.execution_date.to_numpy()):raise ValueError('Not next-open prediction')
    if not (pred.model_cutoff<=signals).all() or not np.isfinite(pred.prediction).all():raise ValueError('Model unavailable at signal time')
    if set(pred.execution_date)!=set(expected):raise ValueError('Trading dates missing')
    benchmark=pd.read_parquet(SOURCE/'data/benchmark.parquet')
    if not benchmark.index.equals(expected):raise ValueError('Benchmark interval mismatch')
    market=pd.read_parquet(SOURCE/'data/market.parquet',columns=['open'])
    if market.index.get_level_values('datetime').max()>pd.Timestamp(END):raise ValueError('Future quotes')
    p=dict(start=START,end=END,initial_cash=1e6,top_k=[20,10],friction=[0.,.001,.002],
           rules={str(k):asdict(Rules(k)) for k in (20,10)},risk_degree=.95,
           stop_budget='min(entry NAV * .02, initial fee-inclusive cost * .05)',
           cooldown_sessions=3,strong_entry_top=2,max_strong_replacements=2,
           retained_weights='drift; no topups',signal='completed close',execution='next ideal open',
           memory_gate_enabled=False,max_workers=1,threads=4,
           implementation=implementation(),sources=sources,
           protected={**source_p['protected'],**source_p['implementation']},
           frozen_source_manifest=read(SOURCE/'report_manifest.json')['artifacts'],
           limitations=source_p['limitations'])
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=p:raise ValueError('Protocol already exists and differs')
    save(OUT/'protocol.json',p)
    save(OUT/'audit_complete.json',dict(status='passed',prediction_rows=len(pred),trading_days=len(expected),
         quarters=16,no_training=True,source_cutoff=END,protocol_sha256=digest(OUT/'protocol.json')))
    print('AUDIT passed; 16 quarters reused, no training, no 2026 rows',flush=True)


def replay(top_k,friction,market,predictions,dates,prior):
    import pandas as pd
    name=f'top{top_k}_{round(friction*10000):02d}bp';directory=OUT/'accounts'/name
    done=directory/'complete.json'
    if done.exists():
        c=read(done)
        if c['protocol_sha256']!=digest(OUT/'protocol.json'):raise ValueError('Mixed account protocol')
        verify(c['artifacts']);return
    a=Account(top_k,friction)
    for step,day in enumerate(dates):
        if step%100==0:
            save(OUT/'progress.json',dict(account=name,day=str(day.date()),step=step,total=len(dates)))
        rows=predictions[day];prev=prior[day]
        if not rows.datetime.eq(prev).all():raise ValueError('Invalid signal alignment')
        previous={c:q['close'] for c,q in market[prev].items()}
        a.step(str(day.date()),step,dict(zip(rows.instrument,rows.prediction)),market[day],previous,
               signal_day=str(prev.date()))
    checks=a.reconcile()
    records=dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,
                 episodes=a.episodes,open_episodes=a.open_episodes(str(dates[-1].date()),len(dates)-1))
    directory.mkdir(parents=True,exist_ok=True)
    for n,r in records.items():pd.DataFrame(r).to_parquet(directory/(n+'.parquet'),index=False)
    save(directory/'reconciliation.json',checks)
    save(done,dict(status='complete',top_k=top_k,friction=friction,
         protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in directory.iterdir()
                                                             if f.is_file() and f.name!='complete.json'}))
    print(f'COMPLETE {name} NAV={a.daily[-1]["nav"]:.2f} closed={len(a.episodes)} open={len(a.positions)}',flush=True)


def backtest():
    import pandas as pd
    p=protocol();verify(p['sources'])
    panel=pd.read_parquet(SOURCE/'data/market.parquet').reset_index()
    market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')}
    del panel
    frame=pd.read_parquet(SOURCE/'data/predictions.parquet').reset_index()
    predictions=dict(tuple(frame.groupby('execution_date')))
    calendar=pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    dates=calendar[(calendar>=START)&(calendar<=END)]
    prior={d:calendar[calendar.get_loc(d)-1] for d in dates}
    for k in (20,10):
        for friction in (0.,.001,.002):replay(k,friction,market,predictions,dates,prior)


def main():
    import psutil
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--as-of',choices=[END],default=END)
    parser.add_argument('--stage',choices=['audit','backtest','report','all'],default='all')
    args=parser.parse_args()
    with lock():
        status=dict(status='running',pid=os.getpid(),created=psutil.Process().create_time(),
                    started=time.time(),stage=args.stage,memory_gate_enabled=False)
        save(OUT/'status.json',status)
        try:
            if args.stage in ('audit','all'):audit()
            if args.stage in ('backtest','all'):backtest()
            if args.stage in ('report','all'):
                from paper_futu.rank_buffer_10_20_report import report
                report(OUT)
            status.update(status='complete',elapsed=time.time()-status['started'])
        except BaseException as e:
            status.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
        finally:save(OUT/'status.json',status)


if __name__=='__main__':main()
