import numpy as np
import pandas as pd
from paper_futu.cn_intraday_stop_account import StopPolicyAccount
from paper_futu.cn_intraday_ablation_account import AblationAccount


def step(a,t,scores,price=10.):
    codes=set(scores)|set(a.positions);p={c:price for c in codes}
    q={c:dict(open=price,factor=1.,volume=1e7) for c in codes}
    a.step(t,t+pd.Timedelta(minutes=5),scores,p,q,p)


def test_daily_rank_clock_and_price_exits_remain_intraday():
    a=AblationAccount(rank_cadence='daily')
    scores={f'SZ{i:06d}':float(120-i) for i in range(1,121)}
    step(a,pd.Timestamp('2025-01-02 09:45'),scores)
    flipped={c:-v for c,v in scores.items()}
    step(a,pd.Timestamp('2025-01-03 09:40'),flipped)
    assert not a.closed
    step(a,pd.Timestamp('2025-01-03 09:45'),flipped)
    assert any(x['reason']=='rank' for x in a.closed)
    b=AblationAccount(rank_cadence='daily')
    step(b,pd.Timestamp('2025-01-02 09:45'),scores)
    step(b,pd.Timestamp('2025-01-03 10:15'),scores,9.)
    assert b.closed and all(x['reason']=='stop' for x in b.closed)


def test_30m_does_not_update_strong_reference_between_checks():
    a=AblationAccount(rank_cadence='30m')
    first={f'SZ{i:06d}':float(12-i) for i in range(1,12)}
    step(a,pd.Timestamp('2025-01-02 10:00'),first)
    changed={**first,'SZ000011':20.}
    step(a,pd.Timestamp('2025-01-03 09:55'),changed)
    assert not a.closed and a.previous_ranks['SZ000011']==11
    step(a,pd.Timestamp('2025-01-03 10:00'),changed)
    assert a.closed[0]['reason']=='strong'


def test_5m_matches_original_stop4_engine():
    rng=np.random.default_rng(42);a=AblationAccount();b=StopPolicyAccount(stop_loss=.04)
    codes=[f'SZ{i:06d}' for i in range(1,121)]
    for day in pd.bdate_range('2025-01-02',periods=20):
        for minute in (585,600,660,840):
            scores=dict(zip(codes,rng.normal(size=len(codes))))
            price=float(rng.uniform(8,12));t=day+pd.Timedelta(minutes=minute)
            for acc in (a,b):step(acc,t,scores,price)
            assert a.orders==b.orders and a.closed==b.closed and a.positions==b.positions
