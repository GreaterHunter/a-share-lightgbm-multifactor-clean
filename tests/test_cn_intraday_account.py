import pandas as pd
from paper_futu.cn_intraday_account import Account


def q(p=10,v=1e7):return dict(open=p,factor=1.,volume=v)


def step(a,day,t,price,score=1):
    signal=pd.Timestamp(day+' '+t)
    a.step(signal,signal+pd.Timedelta(minutes=5),{'SZ000001':score},{'SZ000001':price},
           {'SZ000001':q(price)},{'SZ000001':price})


def test_t1_and_stop_cooldown_and_cash():
    a=Account();step(a,'2025-01-02','10:00',10)
    step(a,'2025-01-02','11:00',9)
    assert len(a.positions)==1 and a.blocked_t1==1
    step(a,'2025-01-03','10:00',9)
    assert not a.positions and len(a.closed)==1
    step(a,'2025-01-03','11:00',9)
    assert not a.positions
    step(a,'2025-01-06','10:00',9)
    assert len(a.positions)==1
    a.close_day(pd.Timestamp('2025-01-06'),{'SZ000001':9})
    assert a.reconcile()['status']=='passed'


def test_profit_can_reenter_later_same_day_but_not_same_fill():
    a=Account();step(a,'2025-01-02','10:00',10)
    step(a,'2025-01-03','10:00',12)
    assert not a.positions
    step(a,'2025-01-03','11:00',12)
    assert len(a.positions)==1
    a.reconcile()


def test_zero_volume_cannot_fill_and_partial_sale_does_not_free_slot():
    a=Account();s=pd.Timestamp('2025-01-02 10:00')
    a.step(s,s+pd.Timedelta(minutes=5),{'SZ000001':1},{'SZ000001':10},{'SZ000001':q(10,0)},{'SZ000001':10})
    assert not a.orders
    step(a,'2025-01-02','10:00',10)
    s=pd.Timestamp('2025-01-03 10:00')
    a.step(s,s+pd.Timedelta(minutes=5),{'SZ000001':1},{'SZ000001':9},{'SZ000001':q(9,2000)},{'SZ000001':9})
    assert a.positions['SZ000001'].pending=='stop'
    assert len(a.closed)==0 and a.orders[-1]['raw_quantity']==100
    a.reconcile()
