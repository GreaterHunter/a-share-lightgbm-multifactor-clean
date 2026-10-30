"""Download and qualify pinned public minute files. No training or broker calls.

python -m scripts.prepare_cn_intraday_public [--limit N]
Run without --limit to examine the complete historical membership universe.
Data qualification must pass before the separate model/backtest experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import requests

from src.cn_intraday_source import normalize_public_bars

ROOT=Path(__file__).resolve().parents[1]
PROBE=ROOT/'reports/experiments/cn_intraday_data_feasibility_20261029'
OUT=ROOT/'reports/experiments/cn_intraday_etfproxy_asof_20251231'
SOURCE=ROOT/'reports/experiments/top5_rank_buffer_asof_20251231/data'
START,END='2015-01-05','2025-12-31'
STOCK_REV='ba589a11534825044fe5a6b84838f50ba8d8d188'
ETF_REV='4b7e48c1e8265cc7d25baee55879dd5bbb24f74d'


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=True,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(path)


def download(code):
    target=OUT/'source_files'/f'{code}.parquet'
    etf=code=='SH510300'
    sample=PROBE/(f'hf_etf_{code[2:]}.parquet' if etf else f'hf_stock_{code[2:]}.parquet')
    if not target.exists():
        if shutil.disk_usage(OUT).free<5*2**30: raise RuntimeError('Less than 5 GiB free disk; preserve existing files')
        if sample.exists(): shutil.copyfile(sample,target)
        else:
            repo='china-etf-1min-ohlcv' if etf else 'china-a-share-1min-ohlcv'
            revision=ETF_REV if etf else STOCK_REV
            folder='etf_1m' if etf else 'stock_1m'
            url=f'https://huggingface.co/datasets/neigezhu/{repo}/resolve/{revision}/data/{folder}/{code[:2]}/{code[2:]}.parquet'
            tmp=target.with_suffix('.partial')
            with requests.get(url,stream=True,timeout=(15,60)) as response:
                response.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in response.iter_content(2**20): f.write(chunk)
            # Validate the file before promoting it; leave partial for diagnosis.
            import pyarrow.parquet as pq
            pq.read_metadata(tmp)
            tmp.replace(target)
    return target


def prepare_one(code, expected_days, reference):
    record=OUT/'audit'/f'{code}.json'
    if record.exists():
        old=json.loads(record.read_text(encoding='utf8'))
        if old.get('implementation_sha256')!=digest(ROOT/'src/cn_intraday_source.py'):
            raise RuntimeError('Normalization code changed; audit existing caches first')
        if old.get('status')=='complete':
            for name,h in old['artifacts'].items():
                if digest(Path(name))!=h: raise RuntimeError('Cached data changed')
            return old
    path=download(code)
    raw=pd.read_parquet(path,filters=[('timestamp','>=',pd.Timestamp(START)),
                                     ('timestamp','<',pd.Timestamp(END)+pd.Timedelta(days=1))])
    bars,daily,audit=normalize_public_bars(raw,code,START,END)
    del raw
    bars.to_parquet(OUT/'bars_5m'/f'{code}.parquet')
    daily.to_parquet(OUT/'daily'/f'{code}.parquet')
    missing=expected_days.difference(daily.index)
    price_audit={}
    if reference is not None:
        c=daily.join(reference[['open','close','factor']],lsuffix='_minute',rsuffix='_daily',how='inner')
        for field in ('open','close'):
            expected=c[f'{field}_daily']/c.factor
            delta=(c[f'{field}_minute']-expected).abs()
            valid=expected.gt(0)&expected.notna()
            mismatch=valid & delta.gt(.011) & (delta/expected).gt(.001)
            price_audit[field]=dict(compared=int(valid.sum()),mismatches=int(mismatch.sum()),
                mismatch_examples=[str(x.date()) for x in c.index[mismatch][:10]])
    result=dict(status='complete',qualification='candidate_only_not_approved_for_backtest',instrument=code,
        source_revision=ETF_REV if code=='SH510300' else STOCK_REV,source_sha256=digest(path),
        implementation_sha256=digest(ROOT/'src/cn_intraday_source.py'),normalization=audit,
        expected_pool_days=len(expected_days),missing_pool_days=len(missing),
        missing_pool_day_examples=[str(x.date()) for x in missing[:20]],reference_prices=price_audit,
        artifacts={str(p):digest(p) for p in [OUT/'bars_5m'/f'{code}.parquet',OUT/'daily'/f'{code}.parquet']})
    save(record,result)
    return result


def main():
    import msvcrt
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--limit',type=int);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ('source_files','bars_5m','daily','audit'): (OUT/folder).mkdir(exist_ok=True)
    with (OUT/'prepare.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise RuntimeError('Data preparation already running')
        protected={str(p):digest(p) for p in [ROOT/'configs/futu_cn_forward_protocol.json',
            ROOT/'paper_futu/cn_rank80_strategy.py',ROOT/'paper_futu/cn_rank80_runtime.py']}
        feats=pd.read_parquet(SOURCE/'features.parquet',columns=['book_to_price'],
            filters=[('datetime','>=',pd.Timestamp(START)),('datetime','<=',pd.Timestamp(END))])
        expected={code:g.index.get_level_values('datetime').unique() for code,g in feats.groupby(level='instrument')}
        calendar=pd.DatetimeIndex(sorted(feats.index.get_level_values('datetime').unique()))
        del feats
        panel=pd.read_parquet(SOURCE/'market.parquet')
        refs={code:g.droplevel('instrument') for code,g in panel.groupby(level='instrument')};del panel
        stock_codes=sorted(expected)
        info=json.loads((PROBE/'hf_stock_info.json').read_text(encoding='utf8'))
        available={x['rfilename'].split('/')[-2]+Path(x['rfilename']).stem for x in info['siblings'] if x['rfilename'].startswith('data/stock_1m/')}
        absent=[c for c in stock_codes if c not in available]
        eligible=[c for c in stock_codes if c in available]
        selected=eligible[:args.limit] if args.limit else eligible
        protocol=dict(training_data_start=START,end=END,test_start='2022-01-04',
            stock_revision=STOCK_REV,etf_revision=ETF_REV,benchmark_intraday='510300 ETF proxy (explicit user approval)',
            performance_benchmark='H00300 total return',bar_minutes=5,features=29,
            t_plus_one=True,take_profit_cooldown=0,stop_loss_cooldown=1,
            stock_universe_count=len(stock_codes),absent_source_instruments=absent,
            scope='Data qualification only; not a backtest or model approval',protected=protected)
        save(OUT/'data_protocol.json',protocol)
        results=[];errors=[];started=time.time()
        jobs=['SH510300']+selected
        def status(state):
            save(OUT/'data_status.json',dict(status=state,total_this_run=len(jobs),finished=len(results),
                errors=errors,elapsed_seconds=time.time()-started,backtest_complete=False,training_started=False,
                last_update=time.time(),full_historical_universe=len(stock_codes),absent_source_instruments=absent))
        status('running')
        # Network I/O only: two bounded public downloads, no parallel model fits.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(prepare_one,c,calendar if c=='SH510300' else expected[c],refs.get(c)):c for c in jobs}
            for future in as_completed(futures):
                code=futures[future]
                try: results.append(future.result())
                except Exception as exc: errors.append(dict(instrument=code,error_type=type(exc).__name__,message=str(exc)[:250]))
                status('running')
                print(json.dumps(dict(finished=len(results),errors=len(errors),total=len(jobs),last=code)),flush=True)
        for p,h in protected.items():
            if digest(p)!=h: raise RuntimeError('Protected production artifact changed during preparation')
        summary=dict(results=results,errors=errors,absent_source_instruments=absent,
            total_missing_pool_days=sum(r['missing_pool_days'] for r in results if r['instrument']!='SH510300'),
            status='downloaded_requires_coverage_review',production_files_unchanged=True,
            training_started=False,backtest_complete=False,limited_run=args.limit is not None)
        save(OUT/'data_summary.json',summary);status('downloaded_requires_coverage_review')


if __name__=='__main__': main()
