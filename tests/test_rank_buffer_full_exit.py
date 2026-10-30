import copy
import pytest
from paper_futu.rank_buffer_full_exit_engine import Account,Episode,price_action
from paper_futu.research_rank_buffer_full_exit import verify_engine_scope


def sample():
    codes=[f'SZ300{i:03d}' for i in range(1,81)]
    scores={c:float(80-i) for i,c in enumerate(codes)}
    quotes={c:dict(open=100.,close=100.,factor=1.) for c in codes}
    return codes,scores,quotes,{c:100. for c in codes}


def test_only_permitted_engine_sections_changed():
    verify_engine_scope()


@pytest.mark.parametrize('k',[10,20])
@pytest.mark.parametrize('tp,sl',[(.10,.05),(.08,.04)])
def test_full_exit_thresholds_irrespective_of_rank(k,tp,sl):
    a=Account(k,profit=tp,stop=sl)
    p=Episode(1,'SZ300001','2022-01-04',0,1e6,1000.,1000.,100000.,100000.*sl,100.)
    for rank in (1,k,k+1,50):
        assert price_action(p,100*(1+tp),rank,a.rules,0)==('profit_full',True)
        assert price_action(p,100*(1+tp)-.0001,rank,a.rules,0) is None
        assert price_action(p,100*(1-sl),rank,a.rules,0)==('loss_budget',True)
        # Fee-inclusive budget trips before the unadjusted price threshold.
        assert price_action(p,100*(1-sl)+.1,rank,a.rules,0)==('loss_budget',True)
        assert price_action(p,100*(1-sl)+.3,rank,a.rules,0) is None


@pytest.mark.parametrize('k',[10,20])
def test_old_four_percent_half_removed(k):
    a=Account(k)
    p=Episode(1,'SZ300001','2022-01-04',0,1e6,1000.,1000.,100000.,5000.,100.)
    assert price_action(p,96.,1,a.rules,0) is None


@pytest.mark.parametrize('k',[10,20])
@pytest.mark.parametrize('tp,sl',[(.10,.05),(.08,.04)])
@pytest.mark.parametrize('side',['profit','loss'])
def test_trigger_exits_entire_position_not_half(k,tp,sl,side):
    codes,scores,quotes,prev=sample();a=Account(k,profit=tp,stop=sl)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[0];ep=a.positions[c];qty=ep.quantity
    price=ep.unit_cost*(1+tp if side=='profit' else 1-sl)
    prev[c]=price;quotes[c].update(open=price,close=price)
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04')
    sells=[o for o in a.orders if o['code']==c and o['direction']=='SELL']
    assert len(sells)==1 and sells[0]['quantity']==qty and c not in a.positions
    assert a.cooldown[c]==4 and a.episodes[-1]['half_mode']==''
    assert len(a.positions)<=k
    a.reconcile()


def test_partial_fill_keeps_full_exit_after_rebound_and_restart():
    codes,scores,quotes,prev=sample();a=Account(10)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[0];prev[c]=94.;quotes[c].update(open=94.,close=94.,sell_capacity_raw=100)
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04')
    assert a.positions[c].pending_full and not a.positions[c].half_mode
    resumed=copy.deepcopy(a);prev[c]=100.;quotes[c].update(open=100.,close=100.,sell_capacity_raw=100000)
    for obj in (a,resumed):obj.step('2022-01-06',2,scores,quotes,prev,signal_day='2022-01-05')
    assert a.orders==resumed.orders and c not in a.positions
    assert not a.episodes[-1]['half_mode']


def test_no_unrequested_threshold_search():
    with pytest.raises(ValueError):Account(10,profit=.12,stop=.06)
