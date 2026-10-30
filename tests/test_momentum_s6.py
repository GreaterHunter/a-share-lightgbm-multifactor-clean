import numpy as np
import pandas as pd

from paper_futu.research_momentum_s6 import momentum_score,schedule,CASES,bounded_step_time


def test_momentum_skip_window_and_future_invariance():
    days=pd.bdate_range('2020-01-01',periods=600)
    close=pd.DataFrame({'a':np.arange(1.,601.),'b':np.arange(10.,610.)},index=days)
    score=momentum_score(close)
    assert score.iloc[:252].isna().all().all()
    assert score.iloc[300,0]==close.iloc[279,0]/close.iloc[48,0]-1
    changed=close.copy();changed.iloc[280:]*=50
    pd.testing.assert_frame_equal(momentum_score(changed).iloc[:301],score.iloc[:301])


def test_missing_prices_do_not_compress_calendar():
    close=pd.DataFrame({'a':np.arange(1.,401.)})
    close.iloc[48]=np.nan
    assert pd.isna(momentum_score(close).iloc[300,0])
    assert np.isfinite(momentum_score(close).iloc[299,0])


def test_monthly_schedule_no_same_close_execution_or_2026():
    calendar=pd.bdate_range('2021-12-01','2025-12-31')
    days,trades,signals=schedule(calendar,'monthly')
    assert len(trades)==48
    assert len(set(trades.to_period('M')))==48
    assert trades[0]==days[0]
    assert (signals<trades).all()
    assert (calendar.get_indexer(trades)-calendar.get_indexer(signals)==1).all()
    assert trades.max()<=pd.Timestamp('2025-12-31')


def test_s6_every_five_does_not_reset_at_year_boundary():
    calendar=pd.bdate_range('2021-12-01','2025-12-31')
    days,trades,_=schedule(calendar,'five_days')
    assert trades.equals(days[::5])


def test_paired_portfolio_and_main_strategy_settings():
    a,b=CASES['momentum_monthly'],CASES['s6_monthly_equal']
    assert {k:v for k,v in a.items() if k!='signal'}=={k:v for k,v in b.items() if k!='signal'}
    assert a['n_drop']==20 and a['top_tier_size']==20
    s=CASES['s6_main']
    assert (s['n_drop'],s['top_tier_size'],s['top_tier_weight'])==(6,6,.4)


def test_terminal_daily_boundary_requires_no_2026_session():
    from types import SimpleNamespace
    m=SimpleNamespace(_calendar=pd.to_datetime(['2025-12-30','2025-12-31']),start_index=0,get_trade_step=lambda:1)
    assert bounded_step_time(m)==(pd.Timestamp('2025-12-31'),pd.Timestamp('2025-12-31 23:59:59'))
    assert bounded_step_time(m,shift=1)==(pd.Timestamp('2025-12-30'),pd.Timestamp('2025-12-30 23:59:59'))
