import numpy as np
import pandas as pd
import pytest

from paper_futu.cn_intraday_account import Account
from paper_futu.cn_intraday_rank_account import RankPolicyAccount


def run(a,day,scores,price=10.):
    codes=set(scores)|set(a.positions)
    prices={c:price for c in codes}
    quotes={c:dict(open=price,factor=1.,volume=1e7) for c in codes}
    ts=pd.Timestamp(day+' 10:00')
    a.step(ts,ts+pd.Timedelta(minutes=5),scores,prices,quotes,prices)


@pytest.mark.parametrize('rank,exits',[(80,False),(100,False),(101,True)])
def test_rank100_exact_boundary(rank,exits):
    a=RankPolicyAccount(rank_exit=100,strong_entry=False)
    run(a,'2025-01-02',{'SZ000001':2.})
    scores={f'SZ{i:06d}':float(rank-i+2) for i in range(2,rank+1)}
    scores['SZ000001']=0.
    run(a,'2025-01-03',scores)
    assert any(x['code']=='SZ000001' and x['reason']=='rank' for x in a.closed)==exits


def test_disabled_rank_keeps_removed_member_until_price_exit():
    a=RankPolicyAccount(rank_exit=None,strong_entry=False)
    run(a,'2025-01-02',{'SZ000001':2.})
    run(a,'2025-01-03',{'SZ000002':2.})
    assert 'SZ000001' in a.positions and not a.closed
    run(a,'2025-01-06',{'SZ000002':2.},price=11.5)
    assert any(x['code']=='SZ000001' and x['reason']=='profit' for x in a.closed)


def test_strong_replacement_switch():
    enabled=RankPolicyAccount(rank_exit=100,strong_entry=True)
    disabled=RankPolicyAccount(rank_exit=None,strong_entry=False)
    before={f'SZ{i:06d}':float(12-i) for i in range(1,12)}
    after={**before,'SZ000011':20.}
    for a in (enabled,disabled):
        run(a,'2025-01-02',before)
        run(a,'2025-01-03',after)
    assert enabled.closed[0]['reason']=='strong'
    assert not disabled.closed and 'SZ000011' not in disabled.positions


def test_no_rank_preserves_t1_and_stop_cooldown():
    a=RankPolicyAccount(rank_exit=None,strong_entry=False)
    scores={'SZ000001':1.}
    run(a,'2025-01-02',scores)
    run(a,'2025-01-02',scores,9.)
    assert a.positions['SZ000001'].pending=='stop' and not a.closed
    run(a,'2025-01-03',scores,9.)
    assert a.closed[0]['reason']=='stop'
    run(a,'2025-01-03',scores,9.)
    assert not a.positions
    run(a,'2025-01-06',scores,9.)
    assert 'SZ000001' in a.positions


@pytest.mark.parametrize('friction',[0.,.001])
def test_rank80_matches_sealed_engine_across_price_rank_and_partial_fills(friction):
    a=Account(friction);b=RankPolicyAccount(friction)
    rng=np.random.default_rng(42)
    codes=[f'SZ{i:06d}' for i in range(1,121)]
    for day in pd.bdate_range('2025-01-02',periods=30):
        for minute in (600,660,840):
            prices=dict(zip(codes,rng.uniform(8,12,len(codes))))
            scores=dict(zip(codes,rng.normal(size=len(codes))))
            quotes={c:dict(open=p,factor=1.,volume=float(rng.choice([0,4000,1e7]))) for c,p in prices.items()}
            ts=day+pd.Timedelta(minutes=minute)
            for acc in (a,b):acc.step(ts,ts+pd.Timedelta(minutes=5),scores,prices,quotes,prices)
            assert a.orders==b.orders and a.closed==b.closed
            assert a.positions==b.positions and a.cash==b.cash
        for acc in (a,b):acc.close_day(day,prices)
        assert a.daily==b.daily and a.holdings==b.holdings
    assert a.reconcile()==b.reconcile()
