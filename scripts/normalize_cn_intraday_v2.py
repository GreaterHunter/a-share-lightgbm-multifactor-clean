"""Rebuild cached public data with explicit no-trade prices, into a NEW version.

No source downloads, no overwrites of version-one results, no broker imports.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from scripts.prepare_cn_intraday_public import OUT, ROOT, START, END, digest, save
from src.cn_intraday_source import normalize_public_bars


def quarantine_volume_conflicts(raw):
    bad=(raw.volume.eq(0)&raw.turnover.ne(0))|(raw.volume.gt(0)&raw.turnover.le(0))
    return raw.loc[~bad].copy(),raw.loc[bad].copy()


def run():
    import msvcrt
    version=OUT/'normalization_v2';version.mkdir(exist_ok=True)
    for folder in ('bars_5m','daily','audit'): (version/folder).mkdir(exist_ok=True)
    with (version/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise RuntimeError('Normalization v2 already running')
        summary=json.loads((OUT/'data_summary.json').read_text(encoding='utf8'))
        if summary['errors'] or len(summary['results'])!=658: raise RuntimeError('Source acquisition incomplete')
        code_hash=digest(ROOT/'src/cn_intraday_source.py')
        protocol=dict(implementation_sha256=code_hash,source_summary_sha256=digest(OUT/'data_summary.json'),
            source_protocol_sha256=digest(OUT/'data_protocol.json'),
            change='Exclude zero-trade carried quotes from OHLC; retain missingness and all source-volume records',
            no_training=True,no_broker=True,runner_sha256=digest(__file__),
            quarantine_policy='Contradictory volume/turnover rows are saved separately then omitted, invalidating their entire 5-minute bucket; never impute a price')
        binding=version/'protocol.json'
        if binding.exists() and json.loads(binding.read_text(encoding='utf8'))!=protocol:
            old=json.loads(binding.read_text(encoding='utf8'))
            if any(protocol.get(k)!=v for k,v in old.items()) or 'runner_sha256' in old:
                raise RuntimeError('Existing version binding changed')
            save(version/'compatibility_before_quarantine.json',dict(previous=old,
                reason='Completed caches passed the original strict no-conflict validator. New preprocessing is identity on those inputs; source normalizer hash unchanged. Only previously failing inputs can differ.',
                completed_cache_count=len(list((version/'audit').glob('*.json'))),new_runner_sha256=digest(__file__)))
        save(binding,protocol)
        started=time.time();results=[]
        for record in summary['results']:
            code=record['instrument'];destination=version/'audit'/f'{code}.json'
            if destination.exists():
                r=json.loads(destination.read_text(encoding='utf8'))
                if r['implementation_sha256']!=code_hash: raise RuntimeError('Cache code changed')
                for path,h in r['artifacts'].items():
                    if digest(path)!=h: raise RuntimeError('Cache data changed')
            else:
                source=OUT/'source_files'/f'{code}.parquet'
                if digest(source)!=record['source_sha256']: raise RuntimeError('Source data changed')
                raw=pd.read_parquet(source,filters=[('timestamp','>=',pd.Timestamp(START)),
                    ('timestamp','<',pd.Timestamp(END)+pd.Timedelta(days=1))])
                raw,conflicts=quarantine_volume_conflicts(raw)
                if len(conflicts):
                    (version/'quarantine').mkdir(exist_ok=True)
                    conflicts.to_parquet(version/'quarantine'/f'{code}.parquet',index=False)
                bars,daily,audit=normalize_public_bars(raw,code,START,END)
                audit['quarantined_rows']=len(conflicts)
                bars.to_parquet(version/'bars_5m'/f'{code}.parquet')
                daily.to_parquet(version/'daily'/f'{code}.parquet')
                r=dict(instrument=code,source_sha256=record['source_sha256'],implementation_sha256=code_hash,
                    audit=audit,artifacts={str(p):digest(p) for p in [version/'bars_5m'/f'{code}.parquet',version/'daily'/f'{code}.parquet']})
                save(destination,r)
            results.append(r)
            save(version/'status.json',dict(status='running',pid=os.getpid(),completed=len(results),total=658,
                elapsed_seconds=time.time()-started,updated_at=time.time(),training_started=False,backtest_complete=False))
            if len(results)%25==0: print(f'NORMALIZED {len(results)}/658',flush=True)
        save(version/'completion.json',dict(status='normalization_complete_pending_data_acceptance',
            protocol_sha256=digest(binding),records=results,training_started=False,backtest_complete=False))
        save(version/'status.json',dict(status='normalization_complete_pending_data_acceptance',pid=os.getpid(),
            completed=len(results),total=658,elapsed_seconds=time.time()-started,training_started=False,backtest_complete=False))


if __name__=='__main__':
    try:run()
    except Exception as exc:
        save(OUT/'normalization_v2/status.json',dict(status='failed',pid=os.getpid(),
            error_type=type(exc).__name__,error=str(exc),training_started=False,backtest_complete=False))
        raise
