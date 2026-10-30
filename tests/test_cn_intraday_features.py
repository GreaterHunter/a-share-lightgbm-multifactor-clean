import numpy as np
import pandas as pd
import pytest

from src.cn_intraday_features import (
    FEATURE_NAMES, ENHANCED_FEATURE_NAMES, _grid,
    minute_features, attach_previous_daily,
)


def bars(day='2025-12-30', interval=1):
    index = _grid(pd.Timestamp(day), interval)
    price = 10 + np.arange(len(index))*.01
    return pd.DataFrame(dict(open=price, close=price, high=price+.01,
                             low=price-.01, volume=100., amount=100*price), index=index)


@pytest.mark.parametrize('interval', [1, 5])
def test_future_perturbation_and_prefix_invariance(interval):
    s = bars(interval=interval)
    b = s[['open', 'close']].copy()
    before = minute_features(s, b, bar_minutes=interval)
    cutoff = len(s)//2
    changed = s.copy()
    changed.iloc[cutoff:, :4] *= 2
    changed.iloc[cutoff:, 5] *= 2
    b.iloc[cutoff:] *= 3
    after = minute_features(changed, b, bar_minutes=interval)
    pd.testing.assert_frame_equal(before.iloc[:cutoff], after.iloc[:cutoff])
    prefix = minute_features(s.iloc[:cutoff], s[['open', 'close']].iloc[:cutoff], bar_minutes=interval)
    pd.testing.assert_frame_equal(before.iloc[:cutoff], prefix)


def test_29_features_previous_day_only():
    s = bars()
    minute = minute_features(s, s[['open', 'close']])
    cal = pd.to_datetime(['2025-12-29', '2025-12-30', '2025-12-31'])
    daily = pd.DataFrame([np.ones(15), np.ones(15)*999], index=cal[:2], columns=ENHANCED_FEATURE_NAMES)
    result = attach_previous_daily(minute, daily, cal)
    assert len(FEATURE_NAMES) == 29
    assert result[ENHANCED_FEATURE_NAMES].eq(1).all().all()


def test_vwap_relative_returns_and_lunch():
    s = bars()
    f = minute_features(s, s[['open', 'close']])
    assert f['min_relative_return_60'].dropna().eq(0).all()
    assert f.iloc[59].min_vwap_deviation == pytest.approx(s.close.iloc[59]/s.close.iloc[:60].mean()-1)
    assert f.iloc[119].min_session_progress == .5
    assert f.iloc[120].min_session_progress == pytest.approx(121/240)
    assert f.iloc[120].min_return_5 == pytest.approx(s.close.iloc[120]/s.close.iloc[115]-1)


def test_gaps_are_not_filled():
    s = bars()
    broken = s.drop(s.index[70])
    f = minute_features(broken, s[['open', 'close']])
    assert np.isnan(f.loc[s.index[72], 'min_return_5'])
    assert np.isnan(f.loc[s.index[75], 'min_return_5'])  # missing anchor
    assert np.isfinite(f.loc[s.index[76], 'min_return_5'])
    assert np.isnan(f.iloc[-1].min_vwap_deviation)
    b = s[['open', 'close']].drop(s.index[70])
    f = minute_features(s, b)
    assert np.isnan(f.loc[s.index[72], 'min_relative_return_5'])


def test_no_overnight_window():
    s = pd.concat([bars(), bars('2025-12-31')])
    f = minute_features(s, s[['open', 'close']])
    assert np.isnan(f.iloc[240].min_return_60)
    assert f.iloc[240].min_return_since_open == 0


def test_duplicates_bad_clock_and_ohlc_rejected():
    s = bars()
    with pytest.raises(ValueError, match='Duplicate'):
        minute_features(pd.concat([s, s.iloc[:1]]), s[['open', 'close']])
    bad = s.iloc[:1].copy()
    bad.index = [pd.Timestamp('2025-12-30 12:00')]
    with pytest.raises(ValueError, match='timestamps'):
        minute_features(bad, s[['open', 'close']])


@pytest.mark.parametrize('interval',[1,5])
def test_vectorized_matches_reference_with_missing_bars(interval):
    from src.cn_intraday_features import minute_features_batch
    s=pd.concat([bars(interval=interval),bars('2025-12-31',interval=interval)])
    benchmark=s[['open','close']].drop(s.index[20])
    s=s.drop(s.index[[3,17,40]])
    expected=minute_features(s,benchmark,bar_minutes=interval)
    got=minute_features_batch(s,benchmark,bar_minutes=interval)
    pd.testing.assert_frame_equal(got,expected,rtol=1e-10,atol=1e-12)
    bad = s.copy()
    bad.loc[bad.index[0], 'high'] = 1
    with pytest.raises(ValueError, match='OHLC'):
        minute_features(bad, s[['open', 'close']])
