import pytest
from paper_futu.research_top5_head2_full_exit import engine,entry_weight

def sample():
    codes=[f'SZ300{i:03d}' for i in range(1,61)]
    scores={c:60.-i for i,c in enumerate(codes)}
    quotes={c:dict(open=100.,close=100.,factor=1.) for c in codes}
    return codes,scores,quotes,{c:100. for c in codes}

def test_head_weights_and_no_rebalance():
    codes,scores,quotes,prev=sample();a=engine().Account(5)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    buys=a.orders.copy();assert len(buys)==5
    for i,o in enumerate(buys):
        target=1e6*.95*(.25 if i<2 else 1/6)
        assert o['notional']+o['fee']<=target
        assert target-o['notional']-o['fee']<10001
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04')
    assert a.orders==buys
    assert a.cash>=50000;a.reconcile()

@pytest.mark.parametrize('tp,sl',[(.10,.05),(.08,.04)])
@pytest.mark.parametrize('profit',[True,False])
def test_top5_full_exit(tp,sl,profit):
    codes,scores,quotes,prev=sample();a=engine().Account(5,profit=tp,stop=sl)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[0];ep=a.positions[c];q=ep.quantity
    price=ep.unit_cost*(1+tp if profit else 1-sl)
    quotes[c].update(open=price,close=price);prev[c]=price
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04')
    sells=[o for o in a.orders if o['code']==c and o['direction']=='SELL']
    assert len(sells)==1 and sells[0]['quantity']==q
    assert c not in a.positions and not a.episodes[-1]['half_mode']
    assert a.cooldown[c]==4;a.reconcile()

def test_fee_engine_source_isolation():
    from paper_futu.rank_buffer_full_exit_engine import Account,fee
    with pytest.raises(ValueError):Account(5)
    assert fee(100000,True)==50
    assert engine().fee(100000,True)==pytest.approx(55.87)

def test_entry_tiers_use_portfolio_rank():
    a=engine().Account(5);a.positions={'a':None,'b':None,'c':None}
    assert entry_weight(a,'d',{'a':10,'b':20,'c':30,'d':5})==pytest.approx(.2375)
    assert entry_weight(a,'d',{'a':10,'b':20,'c':30,'d':40})==pytest.approx(.95/6)
