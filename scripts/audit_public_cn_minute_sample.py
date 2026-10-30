"""Audit downloaded public equity minute samples against BaoStock raw bars.

This is a data qualification check, NOT a strategy backtest. No broker imports.
"""
from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/experiments/cn_intraday_data_feasibility_20261029'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    wheel = OUT/'vendor/baostock-0.9.4-py3-none-any.whl'
    assert sha(wheel) == '0bf71c6069ab5890ff3596632f9c3f8f1fbc6bfcac582c2f9d6a5c11ab2cfaf8'
    sys.path.insert(0, str(wheel))
    import baostock as bs
    socket.setdefaulttimeout(15)
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError('BaoStock anonymous login failed')
    results = []
    try:
        for symbol, exchange in [('000001', 'sz'), ('600519', 'sh'), ('300750', 'sz')]:
            path = OUT/f'hf_stock_{symbol}.parquet'
            f = pd.read_parquet(path, filters=[('timestamp', '>=', pd.Timestamp('2022-01-01')),
                                               ('timestamp', '<', pd.Timestamp('2026-01-01'))])
            f = f.sort_values('timestamp')
            assert f.symbol.eq(symbol).all() and f.exchange.eq(exchange.upper()).all()
            assert not f.timestamp.duplicated().any()
            f['date'] = f.timestamp.dt.normalize()
            d = f.groupby('date').agg(open=('open', 'first'), high=('high', 'max'),
                                     low=('low', 'min'), close=('close', 'last'),
                                     volume=('volume', 'sum'), amount=('turnover', 'sum'),
                                     bars=('timestamp', 'size'))
            q = bs.query_history_k_data_plus(f'{exchange}.{symbol}',
                    'date,open,high,low,close,volume,amount,tradestatus',
                    start_date='2022-01-01', end_date='2025-12-31', frequency='d', adjustflag='3')
            raw = q.get_data()
            if q.error_code != '0' or raw.empty:
                raise RuntimeError(f'No independent daily bars for {symbol}')
            raw.to_csv(OUT/f'bs_daily_{symbol}.csv', index=False)
            raw['date'] = pd.to_datetime(raw.date)
            raw = raw.set_index('date').apply(pd.to_numeric)
            raw = raw[raw.volume.gt(0)]
            common = d.join(raw, how='inner', lsuffix='_minute', rsuffix='_daily')
            errors = {}
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                abs_error = (common[f'{col}_minute'] - common[f'{col}_daily']).abs()
                rel_error = abs_error/common[f'{col}_daily'].abs().replace(0, np.nan)
                tolerance = .011 if col in ['open', 'high', 'low', 'close'] else None
                errors[col] = dict(max_abs=float(abs_error.max()),
                                   median_rel=float(rel_error.median()), max_rel=float(rel_error.max()),
                                   matched=int((abs_error.le(tolerance) if tolerance else rel_error.le(.001)).sum()),
                                   tolerance='0.011 CNY' if tolerance else '0.1% relative')
            intraday_checks = []
            for day in ['2022-01-04', '2024-06-03', '2025-12-30']:
                q = bs.query_history_k_data_plus(f'{exchange}.{symbol}',
                    'date,time,open,high,low,close,volume,amount',
                    start_date=day, end_date=day, frequency='5', adjustflag='3')
                five = q.get_data()
                if q.error_code != '0' or five.empty:
                    intraday_checks.append(dict(day=day, status='no_reference'))
                    continue
                five.to_csv(OUT/f'bs_5m_{symbol}_{day}.csv', index=False)
                five.index = pd.to_datetime(five.time, format='%Y%m%d%H%M%S%f')
                five = five[['open', 'high', 'low', 'close', 'volume', 'amount']].astype(float)
                part = f[f.date.eq(pd.Timestamp(day))].copy()
                # Published 09:30 auction observation belongs to first bucket.
                ends = part.timestamp.dt.ceil('5min')
                ends = ends.mask(ends.dt.strftime('%H:%M').eq('09:30'), pd.Timestamp(day+' 09:35'))
                ends = ends.mask(ends.dt.strftime('%H:%M').eq('13:00'), pd.Timestamp(day+' 13:05'))
                a = part.groupby(ends).agg(open=('open','first'), high=('high','max'),
                    low=('low','min'), close=('close','last'), volume=('volume','sum'), amount=('turnover','sum'))
                a, b = a.align(five, join='inner')
                intraday_checks.append(dict(day=day, pairs=len(a),
                    close_matches=int((a.close-b.close).abs().le(.011).sum()),
                    volume_matches=int(((a.volume-b.volume).abs()/b.volume.replace(0,np.nan)).le(.001).sum()),
                    zero_volume_reference=int(b.volume.eq(0).sum()),
                    max_close_abs=float((a.close-b.close).abs().max())))
            row = dict(symbol=symbol, source_sha256=sha(path), rows=len(f),
                first=str(f.timestamp.min()), last=str(f.timestamp.max()), observed_days=len(d),
                independent_daily_days=len(raw), matched_days=len(common),
                missing_active_days=[str(t.date()) for t in raw.index.difference(d.index)],
                bar_counts={str(k):int(v) for k,v in d.bars.value_counts().sort_index().items()},
                daily_comparison=errors, intraday_comparison=intraday_checks,
                status='sample_audited_not_full_universe_qualified')
            results.append(row)
            print(json.dumps(row,ensure_ascii=True),flush=True)
    finally:
        bs.logout()
    output=dict(status='equity_samples_audited_index_unresolved', results=results,
                no_training=True, no_backtest=True, no_broker_connection=True,
                limitations=['Only three liquid stocks sampled',
                    'Source provenance does not identify upstream providers',
                    'Minute highs/lows can differ between providers',
                    '09:30 auction row needs explicit normalization before feature construction',
                    'CSI300 historical intraday series still missing'])
    (OUT/'public_sample_audit.json').write_text(json.dumps(output,ensure_ascii=True,indent=2),encoding='utf-8')


if __name__ == '__main__':
    run()
