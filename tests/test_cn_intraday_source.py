import pandas as pd
import pytest

from src.cn_intraday_source import normalize_public_bars


def source():
    index=pd.date_range('2025-12-30 09:30',periods=16,freq='min')
    return pd.DataFrame(dict(timestamp=index,symbol='000001',exchange='SZ',
        open=10.,high=10.,low=10.,close=10.,volume=100,turnover=1000.))


def test_auction_is_not_discarded_and_prefix_causal():
    f=source(); bars,daily,audit=normalize_public_bars(f,'SZ000001','2025-01-01','2025-12-31')
    assert bars.iloc[0].volume==600
    assert bars.iloc[0].amount==6000
    assert bars.volume.sum()==daily.volume.sum()==1600
    assert len(bars)==3
    small,_,_=normalize_public_bars(f.iloc[:11],'SZ000001','2025-01-01','2025-12-31')
    pd.testing.assert_frame_equal(small,bars.iloc[:2])


def test_missing_bar_does_not_invent_liquidity():
    f=source().drop(7)
    b,_,a=normalize_public_bars(f,'SZ000001','2025-01-01','2025-12-31')
    assert pd.Timestamp('2025-12-30 09:40') not in b.index
    assert a['incomplete_5m_buckets']==1
    b,_,_=normalize_public_bars(source().iloc[1:],'SZ000001','2025-01-01','2025-12-31')
    assert pd.Timestamp('2025-12-30 09:35') not in b.index


def test_source_identity_and_duplicate_checks():
    with pytest.raises(ValueError,match='identity'):
        normalize_public_bars(source(),'SH600519','2025-01-01','2025-12-31')
    with pytest.raises(ValueError,match='Duplicate'):
        normalize_public_bars(pd.concat([source(),source().iloc[:1]]),'SZ000001','2025-01-01','2025-12-31')


def test_cutoff_drops_future_before_normalization():
    f=source(); future=f.copy();future['timestamp']+=pd.DateOffset(years=1);future['close']=999
    b,_,_=normalize_public_bars(pd.concat([f,future]),'SZ000001','2025-01-01','2025-12-31')
    expected,_,_=normalize_public_bars(f,'SZ000001','2025-01-01','2025-12-31')
    pd.testing.assert_frame_equal(b,expected)


def test_carried_zero_volume_quote_is_not_an_executable_open():
    f=source();f.loc[:7,['volume','turnover']]=0
    f.loc[:7,['open','high','low','close']]=8
    b,d,a=normalize_public_bars(f,'SZ000001','2025-01-01','2025-12-31')
    assert b.iloc[0][['open','high','low','close']].isna().all()
    assert b.iloc[0].volume==0
    assert b.iloc[1].open==10
    assert d.iloc[0].open==10
    assert a['zero_trade_source_rows']==8


def test_zero_volume_cannot_hide_positive_turnover():
    f=source();f.loc[0,'volume']=0
    with pytest.raises(ValueError,match='Volume and turnover'):
        normalize_public_bars(f,'SZ000001','2025-01-01','2025-12-31')
