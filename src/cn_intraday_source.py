"""Explicit normalization of the pinned public minute-data candidate.

The additional 09:30 observation is carried into the first continuous bar,
never discarded from turnover/VWAP. Incomplete five-minute buckets are absent,
not filled. Nothing here imports or operates the trading application.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_public_bars(frame, instrument, start, end):
    required = ['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    f = frame[required].copy()
    if not f.symbol.astype(str).eq(instrument[2:]).all() or not f.exchange.eq(instrument[:2]).all():
        raise ValueError('Instrument identity mismatch')
    f['timestamp'] = pd.to_datetime(f.timestamp)
    if f.timestamp.dt.tz is not None:
        f['timestamp'] = f.timestamp.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    f = f[f.timestamp.ge(pd.Timestamp(start)) & f.timestamp.lt(pd.Timestamp(end)+pd.Timedelta(days=1))]
    f = f.sort_values('timestamp').set_index('timestamp')
    if f.index.has_duplicates or f.index.hasnans:
        raise ValueError('Duplicate/invalid timestamps')
    numeric = ['open', 'high', 'low', 'close', 'volume', 'turnover']
    if not np.isfinite(f[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Missing/infinite source values')
    if (f[numeric[:4]] <= 0).any().any() or (f[numeric[4:]] < 0).any().any():
        raise ValueError('Invalid price/volume')
    if (f.high < f[numeric[:4]].max(axis=1)).any() or (f.low > f[numeric[:4]].min(axis=1)).any():
        raise ValueError('Invalid OHLC')
    clock = f.index.hour*60+f.index.minute
    legal = ((clock >= 570) & (clock <= 690)) | ((clock >= 781) & (clock <= 900))
    if not legal.all() or (f.index.second != 0).any():
        raise ValueError('Unexpected minute labels; cannot infer timestamp semantics')
    # Some vendors repeat the prior day's quote until the first actual trade.
    # It is not an executable opening price. Keep zero volume/turnover, but do
    # not allow these copied OHLC values into aggregated transaction prices.
    no_trade = f.volume.eq(0)
    if (no_trade & f.turnover.ne(0)).any() or (f.volume.gt(0) & f.turnover.le(0)).any():
        raise ValueError('Volume and turnover disagree about whether trades occurred')
    f.loc[no_trade, ['open', 'high', 'low', 'close']] = np.nan
    # 09:30 plus 09:31-09:35 all contribute to the first bucket. The auction
    # must be present for that bucket: otherwise first OPEN and VWAP unknown.
    bucket = f.index.ceil('5min')
    bucket = bucket.where(clock != 570, f.index.normalize()+pd.Timedelta(hours=9,minutes=35))
    result = f.groupby(bucket).agg(open=('open','first'), high=('high','max'), low=('low','min'),
        close=('close','last'), volume=('volume','sum'), amount=('turnover','sum'), source_bars=('volume','size'))
    expected = np.where((result.index.hour==9)&(result.index.minute==35),6,5)
    incomplete = result[result.source_bars.ne(expected)]
    result = result[result.source_bars.eq(expected)].drop(columns='source_bars')
    result.index.name = 'datetime'
    daily = f.groupby(f.index.normalize()).agg(open=('open','first'), close=('close','last'),
        high=('high','max'), low=('low','min'), volume=('volume','sum'), amount=('turnover','sum'), bars=('close','size'))
    daily.index.name = 'datetime'
    audit = dict(source_rows=len(f), days=len(daily), complete_5m_rows=len(result),
        incomplete_5m_buckets=len(incomplete), complete_days=int(daily.bars.eq(241).sum()),
        zero_trade_source_rows=int(no_trade.sum()), zero_trade_5m_bars=int(result.volume.eq(0).sum()),
        zero_trade_policy='OHLC missing when no trades; first/last positive-volume observation used in each bucket; never execute at carried quotes',
        timestamp_policy='09:30 included in 09:35 bucket; remaining minute labels treated as ends; incomplete buckets omitted',
        no_fill=True, no_post_cutoff_rows=bool(f.empty or f.index.max()<pd.Timestamp(end)+pd.Timedelta(days=1)))
    return result, daily, audit
