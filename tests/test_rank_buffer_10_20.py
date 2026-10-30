import copy
import pytest

from paper_futu.rank_buffer_10_20_engine import Account,Episode,Rules,price_action,fee,half_quantity


def data(n=80):
    codes=[f'SZ300{i:03d}' for i in range(1,n+1)]
    scores={c:float(n-i) for i,c in enumerate(codes)}
    quotes={c:dict(open=100.,close=100.,factor=1.) for c in codes}
    prev={c:100. for c in codes}
    return codes,scores,quotes,prev


def ep():
    return Episode(1,'SZ300001','2022-01-04',0,1e6,1000.,1000.,100000.,5000.,100.)


@pytest.mark.parametrize('k',[10,20])
def test_rules_at_half_rank_and_price_boundaries(k):
    r=Rules(k);p=ep()
    assert price_action(p,96.00001,k,r,0) is None
    assert price_action(p,96.,k,r,0)==('loss_half',False)
    assert price_action(p,96.,k+1,r,0) is None
    assert price_action(p,95.,k+1,r,0)==('loss_budget',True)
    assert price_action(p,95.,k,r,0)==('loss_budget',True)
    assert price_action(p,110.,k,r,0)==('profit_half',False)
    assert price_action(p,110.,k+1,r,0)==('profit_full',True)
    assert price_action(p,109.9999,k,r,0) is None


@pytest.mark.parametrize('k',[10,20])
def test_new_entry_weights_and_budget(k):
    codes,scores,quotes,prev=data();a=Account(k)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    assert len(a.positions)==k and a.cash>=50000
    for p in a.positions.values():
        assert p.initial_cost<=1e6*.95/k
        assert p.budget==min(p.entry_nav*.02,p.initial_cost*.05)
    before={c:p.quantity for c,p in a.positions.items()}
    quotes[codes[0]]['close']=103.
    a.step('2022-01-05',1,scores,quotes,prev)
    assert {c:p.quantity for c,p in a.positions.items()}==before


@pytest.mark.parametrize('k',[10,20])
def test_rank50_stays_rank51_exits(k):
    codes,scores,quotes,prev=data();a=Account(k)
    a.step('2022-01-04',0,scores,quotes,prev)
    c=codes[k-1];scores[c]=30.5
    a.step('2022-01-05',1,scores,quotes,prev)
    assert c in a.positions
    scores[c]=29.5
    a.step('2022-01-06',2,scores,quotes,prev)
    assert c not in a.positions and a.episodes[-1]['exit_reason']=='rank_exit'


@pytest.mark.parametrize('k',[10,20])
def test_two_strong_replacements_inside_rank50(k):
    codes,scores,quotes,prev=data();a=Account(k)
    a.step('2022-01-04',0,scores,quotes,prev)
    scores[codes[k]]=100.;scores[codes[k+1]]=99.
    a.step('2022-01-05',1,scores,quotes,prev)
    sold=[o for o in a.orders if o['date']=='2022-01-05' and o['direction']=='SELL']
    assert len(sold)==2 and all(o['reason']=='strong_replacement' for o in sold)
    assert codes[k] in a.positions and codes[k+1] in a.positions and len(a.positions)==k


def test_loss_half_remaining_budget_and_original_tp():
    p=ep();p.quantity=500;p.sold_net=500*96-fee(500*96,False);p.half_mode='loss'
    stop=p.remaining_budget_price(0)
    assert 94<stop<95
    assert price_action(p,stop,1,Rules(20),0)==('loss_budget',True)
    assert price_action(p,110,1,Rules(20),0)==('profit_after_loss_half',True)
    assert p.budget==5000


def test_profit_half_second_and_trailing():
    p=ep();p.quantity=500;p.sold_net=55000-fee(55000,False);p.half_mode='profit';p.half_price=110
    assert price_action(p,111,1,Rules(10),0) is None
    previous=p.trailing
    assert price_action(p,110,1,Rules(10),0) is None and p.trailing>=previous
    assert price_action(p,121,1,Rules(10),0)==('second_profit',True)
    assert price_action(p,114,1,Rules(10),0)==('profit_trailing',True)


@pytest.mark.parametrize('k',[10,20])
def test_half_once_no_slot_or_topup(k):
    codes,scores,quotes,prev=data();a=Account(k)
    a.step('2022-01-04',0,scores,quotes,prev)
    c=codes[0];initial=a.positions[c].quantity;budget=a.positions[c].budget
    prev[c]=a.positions[c].unit_cost*.96;quotes[c].update(open=prev[c],close=prev[c])
    a.step('2022-01-05',1,scores,quotes,prev)
    remaining=a.positions[c].quantity
    assert remaining<initial and len(a.positions)==k and a.positions[c].budget==budget
    a.step('2022-01-06',2,scores,quotes,prev)
    assert a.positions[c].quantity==remaining
    assert len([o for o in a.orders if o['code']==c and o['direction']=='BUY'])==1
    a.reconcile()


def test_partial_exit_survives_restart_and_rebound():
    codes,scores,quotes,prev=data();a=Account(10)
    a.step('2022-01-04',0,scores,quotes,prev)
    c=codes[0];prev[c]=94.;quotes[c].update(open=94.,close=94.,sell_capacity_raw=100)
    a.step('2022-01-05',1,scores,quotes,prev)
    assert a.positions[c].pending_full
    saved=copy.deepcopy(a);prev[c]=100.;quotes[c].update(open=100.,close=100.,sell_capacity_raw=100000)
    for obj in (a,saved):obj.step('2022-01-06',2,scores,quotes,prev)
    assert a.orders==saved.orders and c not in a.positions and a.cooldown[c]==5


@pytest.mark.parametrize('k',[10,20])
def test_cooldown_Dplus3(k):
    codes,scores,quotes,prev=data();a=Account(k)
    a.step('2022-01-04',0,scores,quotes,prev)
    c=codes[0];prev[c]=94.;quotes[c].update(open=94.,close=94.)
    a.step('2022-01-05',1,scores,quotes,prev)
    assert a.cooldown[c]==4
    for step in (2,3,4):
        scores[next(iter(a.positions))]=-100-step
        a.step(f'2022-01-{step+4:02d}',step,scores,quotes,prev)
        assert (c in a.positions)==(step==4)


def test_limit_suspension_T1_and_gap_not_clipped():
    p=ep();a=Account(10);a.positions[p.code]=p
    a.intent(p,'loss_budget',True,'2022-01-03',1.)
    quote=dict(open=80.,close=80.,factor=1.)
    assert not a.sell(p,'2022-01-04',0,quote,94.)
    assert not a.sell(p,'2022-01-05',1,quote,94.)
    quote['suspended']=True;assert not a.sell(p,'2022-01-06',2,quote,80.)
    quote['suspended']=False;a.sell(p,'2022-01-07',3,quote,80.)
    assert a.orders[-1]['price']==80 and a.episodes[-1]['budget_overshoot']>10000


def test_legal_half_and_invalid_K():
    assert half_quantity('SZ300001',100,1)==100
    assert half_quantity('SZ300001',300,1)==200
    assert half_quantity('SH688001',200,1)==200
    assert half_quantity('SH688001',500,1)==250
    with pytest.raises(ValueError):Rules(5)


def test_future_close_does_not_affect_orders_and_terminal_remains_open():
    codes,scores,quotes,prev=data();a,b=Account(20),Account(20)
    future=copy.deepcopy(quotes)
    for q in future.values():q.update(close=10000.,volume=0.)
    a.step('2025-12-31',0,scores,quotes,prev);b.step('2025-12-31',0,scores,future,prev)
    assert a.orders==b.orders and len(a.open_episodes('2025-12-31',0))==20 and not a.episodes


def test_ties_do_not_force_replacement():
    codes,scores,quotes,prev=data();a=Account(10)
    scores={c:100-i if 10<=i<20 else -i for i,c in enumerate(codes)}
    a.step('2022-01-04',0,scores,quotes,prev)
    a.previous_ranks[codes[0]]=3;a.previous_ranks[codes[1]]=4
    a.step('2022-01-05',1,{c:1. for c in codes},quotes,prev)
    assert not any(o['date']=='2022-01-05' and o['reason']=='strong_replacement' for o in a.orders)


def test_independent_reconciliation_detects_fee_corruption():
    import pandas as pd
    from paper_futu.rank_buffer_10_20_report import independent_audit
    codes,scores,quotes,prev=data();a=Account(20,.001)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    scores[codes[0]]=-100
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04')
    daily,orders,holdings=[pd.DataFrame(v) for v in (a.daily,a.orders,a.holdings)]
    for f in (daily,orders,holdings):f['date']=pd.to_datetime(f.date)
    closed=pd.DataFrame(a.episodes);opened=pd.DataFrame(a.open_episodes('2022-01-05',1))
    assert independent_audit(daily,orders,holdings,closed,opened,20,.001)['status']=='passed'
    orders.loc[0,'fee']+=1
    with pytest.raises(AssertionError):independent_audit(daily,orders,holdings,closed,opened,20,.001)
