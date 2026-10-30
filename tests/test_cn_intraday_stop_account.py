import numpy as np
import pandas as pd
import pytest

from paper_futu.cn_intraday_account import Account
from paper_futu.cn_intraday_stop_account import StopPolicyAccount
from paper_futu.research_rank_buffer_futu_fees import liquidation_breakeven


def tick(a,day,price):
    t=pd.Timestamp(day+' 10:00');code='SZ000001'
    a.step(t,t+pd.Timedelta(minutes=5),{code:1.},{code:price},
           {code:dict(open=price,factor=1.,volume=1e7)},{code:price})


@pytest.mark.parametrize('delta,stops',[(-.0001,True),(0,True),(.0001,False)])
def test_exact_fee_inclusive_four_percent_boundary(delta,stops):
    a=StopPolicyAccount(stop_loss=.04);tick(a,'2025-01-02',10.)
    h=a.positions['SZ000001']
    price=liquidation_breakeven(h.quantity,h.cost*.96,0.)+delta
    tick(a,'2025-01-03',price)
    assert bool(a.closed)==stops
    if stops:assert a.closed[0]['reason']=='stop'


def test_four_percent_changes_stop_but_preserves_t1_and_cooldown():
    a=StopPolicyAccount(stop_loss=.04);b=StopPolicyAccount(stop_loss=.05)
    for acc in (a,b):tick(acc,'2025-01-02',10.)
    h=a.positions['SZ000001'];price=liquidation_breakeven(h.quantity,h.cost*.955,0.)
    for acc in (a,b):tick(acc,'2025-01-02',price)
    assert a.positions['SZ000001'].pending=='stop' and not a.closed
    assert not b.positions['SZ000001'].pending
    tick(a,'2025-01-03',price);tick(a,'2025-01-03',price)
    assert len(a.closed)==1 and not a.positions
    tick(a,'2025-01-06',price)
    assert 'SZ000001' in a.positions


@pytest.mark.parametrize('friction',[0.,.001])
def test_five_percent_reproduces_original_engine(friction):
    a=Account(friction);b=StopPolicyAccount(friction,stop_loss=.05)
    rng=np.random.default_rng(42);codes=[f'SZ{i:06d}' for i in range(1,121)]
    for day in pd.bdate_range('2025-01-02',periods=25):
        for minute in (600,660,840):
            p=dict(zip(codes,rng.uniform(8,12,len(codes))))
            scores=dict(zip(codes,rng.normal(size=len(codes))))
            quotes={c:dict(open=v,factor=1.,volume=float(rng.choice([0,4000,1e7]))) for c,v in p.items()}
            t=day+pd.Timedelta(minutes=minute)
            for acc in (a,b):acc.step(t,t+pd.Timedelta(minutes=5),scores,p,quotes,p)
            assert a.orders==b.orders and a.closed==b.closed and a.positions==b.positions
        for acc in (a,b):acc.close_day(day,p)
        assert a.daily==b.daily
    assert a.reconcile()==b.reconcile()
