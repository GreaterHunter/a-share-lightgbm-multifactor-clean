"""Causal A-share bar-close features, isolated from production trading.

Inputs: unadjusted OHLC, volume in shares, amount in CNY; bar END timestamps
in Asia/Shanghai (naive local timestamps accepted). 1/5-minute bars only.
Windows count trading minutes across lunch, but never cross trading days.
Missing bars invalidate affected windows; they are never forward-filled.
No training, broker access, or same-bar fill assumption lives in this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor_library import ENHANCED_FEATURE_NAMES

MINUTE_FEATURE_NAMES = (
    [f'min_return_{n}' for n in (5, 15, 60)]
    + [f'min_relative_return_{n}' for n in (5, 15, 60)]
    + [f'min_volatility_{n}' for n in (15, 60)]
    + [f'min_amplitude_{n}' for n in (15, 60)]
    + ['min_volume_ratio_5_30', 'min_vwap_deviation',
       'min_return_since_open', 'min_session_progress']
)
FEATURE_NAMES = ENHANCED_FEATURE_NAMES + MINUTE_FEATURE_NAMES


def _local_index(index):
    index = pd.DatetimeIndex(pd.to_datetime(index))
    return index.tz_convert('Asia/Shanghai').tz_localize(None) if index.tz else index


def _grid(day, bar_minutes):
    return pd.date_range(day + pd.Timedelta(hours=9, minutes=30+bar_minutes),
                         day + pd.Timedelta(hours=11, minutes=30), freq=f'{bar_minutes}min').append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=bar_minutes),
                      day + pd.Timedelta(hours=15), freq=f'{bar_minutes}min'))


def _checked(frame, columns, bar_minutes):
    result = frame[columns].copy()
    result.index = _local_index(result.index)
    if result.index.has_duplicates or result.index.hasnans:
        raise ValueError('Duplicate or invalid bar timestamps')
    result = result.sort_index().astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Infinite bar values')
    for day, group in result.groupby(result.index.normalize()):
        if not group.index.isin(_grid(day, bar_minutes)).all():
            raise ValueError('Expected continuous-session bar END timestamps')
    return result


def minute_features(stock, benchmark, *, bar_minutes=1):
    """One stock and CSI300 close series; outputs at supplied stock bar ends.

    Volatility is population std of bar returns (not annualized). Amplitude
    is rolling high/low - 1. Session return uses today's first bar OPEN.
    Different bar resolutions must not be pooled in one model.
    """
    if bar_minutes not in (1, 5):
        raise ValueError('Only 1/5 minute source bars supported')
    columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    s = _checked(stock, columns, bar_minutes)
    b = _checked(benchmark, ['open', 'close'], bar_minutes)
    prices = s[['open', 'high', 'low', 'close']]
    if (prices <= 0).any().any() or (s[['volume', 'amount']] < 0).any().any():
        raise ValueError('Invalid prices, shares or turnover')
    if ((s.high < prices.max(axis=1)) | (s.low > prices.min(axis=1))).any():
        raise ValueError('Inconsistent OHLC')
    if (b <= 0).any().any():
        raise ValueError('Invalid benchmark prices')
    chunks = []
    for day, original in s.groupby(s.index.normalize()):
        # Never examine later bars when constructing earlier outputs.
        grid = _grid(day, bar_minutes)
        q, z = original.reindex(grid), b.reindex(grid)
        close, ref = q.close, z.close
        valid = q[columns].notna().all(axis=1)
        ref_valid = z.notna().all(axis=1)
        out = pd.DataFrame(index=grid)
        prior = close.shift(1)
        prior.iloc[0] = q.open.iloc[0]
        ret = close / prior - 1
        for n in (5, 15, 60):
            width = n // bar_minutes
            anchor, ref_anchor = close.shift(width), ref.shift(width)
            anchor.iloc[width-1] = q.open.iloc[0]
            ref_anchor.iloc[width-1] = z.open.iloc[0]
            available = valid.rolling(width).sum().eq(width)
            ref_available = ref_valid.rolling(width).sum().eq(width)
            value = (close / anchor - 1).where(available)
            ref_value = (ref / ref_anchor - 1).where(ref_available)
            out[f'min_return_{n}'] = value
            out[f'min_relative_return_{n}'] = value - ref_value
        for n in (15, 60):
            width = n // bar_minutes
            out[f'min_volatility_{n}'] = ret.rolling(width).std(ddof=0)
            out[f'min_amplitude_{n}'] = q.high.rolling(width).max() / q.low.rolling(width).min() - 1
        out['min_volume_ratio_5_30'] = q.volume.rolling(5//bar_minutes).mean() / q.volume.rolling(30//bar_minutes).mean().replace(0, np.nan) - 1
        prefix_valid = valid.cummin()
        vwap = q.amount.cumsum() / q.volume.cumsum().replace(0, np.nan)
        out['min_vwap_deviation'] = (close / vwap - 1).where(prefix_valid)
        out['min_return_since_open'] = close / q.open.iloc[0] - 1
        out['min_session_progress'] = np.arange(1, len(grid)+1)*bar_minutes/240
        chunks.append(out.reindex(original.index))
    if not chunks:
        return pd.DataFrame(columns=MINUTE_FEATURE_NAMES, index=s.index)
    return pd.concat(chunks)[MINUTE_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)


def attach_previous_daily(minute, daily, calendar):
    """One stock's 15 daily features, keyed by source day, never today's values.

    Caller must first verify actual financial publication dates and restrict
    universe membership. Missing previous-day features stay missing.
    """
    minute = minute.copy()
    minute.index = _local_index(minute.index)
    daily = daily[ENHANCED_FEATURE_NAMES].copy()
    daily.index = _local_index(daily.index).normalize()
    cal = _local_index(calendar).normalize()
    if daily.index.has_duplicates or cal.has_duplicates or not cal.is_monotonic_increasing:
        raise ValueError('Daily/calendar keys must be unique; calendar sorted')
    days = minute.index.normalize()
    positions=cal.get_indexer(days)
    if (positions<=0).any():
        raise ValueError('Missing previous trading day in calendar')
    lagged = daily.reindex(cal.to_numpy()[positions-1])
    lagged.index = minute.index
    return pd.concat([lagged, minute[MINUTE_FEATURE_NAMES]], axis=1)[FEATURE_NAMES]


def minute_features_batch(stock, benchmark, *, bar_minutes=5):
    """Vectorized equivalent of minute_features, with the same causal windows.

    Reindex every observed stock day to the explicit session grid before rolling;
    an omitted source bar must not shorten a rolling window silently.
    """
    if bar_minutes not in (1,5): raise ValueError('Only 1/5 minute source bars supported')
    columns=['open','high','low','close','volume','amount']
    def check(f,cols):
        f=f[cols].copy();f.index=_local_index(f.index)
        if f.index.has_duplicates or f.index.hasnans: raise ValueError('Duplicate or invalid bar timestamps')
        f=f.sort_index().astype(float)
        clock=f.index.hour*60+f.index.minute
        legal=(((clock>570)&(clock<=690))|((clock>780)&(clock<=900)))&(clock%bar_minutes==0)&(f.index.second==0)
        if not legal.all(): raise ValueError('Expected continuous-session bar END timestamps')
        if np.isinf(f.to_numpy()).any(): raise ValueError('Infinite bar values')
        return f
    s=check(stock,columns);b=check(benchmark,['open','close'])
    if s.empty: return pd.DataFrame(columns=MINUTE_FEATURE_NAMES,index=s.index)
    prices=s[columns[:4]]
    if (prices<=0).any().any() or (s[columns[4:]]<0).any().any() or (b<=0).any().any():
        raise ValueError('Invalid prices, shares or turnover')
    if ((s.high<prices.max(axis=1))|(s.low>prices.min(axis=1))).any(): raise ValueError('Inconsistent OHLC')
    days=s.index.normalize().unique().to_numpy()
    offsets=np.r_[np.arange(570+bar_minutes,691,bar_minutes),np.arange(780+bar_minutes,901,bar_minutes)]
    idx=pd.DatetimeIndex((days[:,None]+offsets.astype('timedelta64[m]')[None,:]).ravel(),name=s.index.name)
    q,z=s.reindex(idx),b.reindex(idx);groups=idx.normalize();nbar=len(offsets)
    valid=q.notna().all(axis=1);ref_valid=z.notna().all(axis=1)
    def rolling(series,width,method):
        # Each row is exactly one trading session. A strided window cannot
        # cross the day boundary; NaNs propagate like min_periods=width.
        values=series.to_numpy(dtype=float).reshape(len(days),nbar)
        windows=np.lib.stride_tricks.sliding_window_view(values,width,axis=1)
        reduced=getattr(np,method)(windows,axis=-1)
        padded=np.full(values.shape,np.nan)
        padded[:,width-1:]=reduced
        return pd.Series(padded.ravel(),index=idx)
    def anchor(series,first,width):
        a=series.groupby(groups,sort=False).shift(width)
        a.iloc[width-1::nbar]=first.iloc[::nbar].to_numpy()
        return a
    prior=anchor(q.close,q.open,1);ret=q.close/prior-1
    out=pd.DataFrame(index=idx)
    for n in (5,15,60):
        width=n//bar_minutes
        r=(q.close/anchor(q.close,q.open,width)-1).where(rolling(valid,width,'sum').eq(width))
        br=(z.close/anchor(z.close,z.open,width)-1).where(rolling(ref_valid,width,'sum').eq(width))
        out[f'min_return_{n}']=r;out[f'min_relative_return_{n}']=r-br
    for n in (15,60):
        width=n//bar_minutes
        out[f'min_volatility_{n}']=rolling(ret,width,'std')
        out[f'min_amplitude_{n}']=rolling(q.high,width,'max')/rolling(q.low,width,'min')-1
    out['min_volume_ratio_5_30']=rolling(q.volume,5//bar_minutes,'mean')/rolling(q.volume,30//bar_minutes,'mean').replace(0,np.nan)-1
    vwap=q.amount.groupby(groups).cumsum()/q.volume.groupby(groups).cumsum().replace(0,np.nan)
    out['min_vwap_deviation']=(q.close/vwap-1).where(valid.groupby(groups).cummin())
    first_open=pd.Series(np.repeat(q.open.iloc[::nbar].to_numpy(),nbar),index=idx)
    out['min_return_since_open']=q.close/first_open-1
    out['min_session_progress']=np.tile(np.arange(1,nbar+1)*bar_minutes/240,len(days))
    return out.reindex(s.index)[MINUTE_FEATURE_NAMES].replace([np.inf,-np.inf],np.nan)
