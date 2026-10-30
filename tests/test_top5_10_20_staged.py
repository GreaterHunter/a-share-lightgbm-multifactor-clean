import pytest
from paper_futu.research_top5_10_20_staged import engine,make_account

@pytest.mark.parametrize('k',[5,10,20])
@pytest.mark.parametrize('tp,sl',[(.10,.05),(.08,.04)])
def test_scaled_rule_boundaries(k,tp,sl):
    a=make_account(k,tp,sl);m=engine()
    ep=m.Episode(1,'SZ300001','2022-01-04',0,1e6,1000,1000,100000,100000*sl,100)
    assert a.rules.loss_half==pytest.approx(sl*.8)
    assert a.rules.trailing_drawdown==sl
    assert m.price_action(ep,100*(1+tp),k,a.rules,0)==('profit_half',False)
    assert m.price_action(ep,100*(1+tp),k+1,a.rules,0)==('profit_full',True)
    assert m.price_action(ep,100*(1-sl*.8),k,a.rules,0)==('loss_half',False)
    assert m.price_action(ep,100*(1-sl*.8),k+1,a.rules,0) is None
    assert m.price_action(ep,100*(1-sl),1,a.rules,0)==('loss_budget',True)

@pytest.mark.parametrize('tp,sl',[(.10,.05),(.08,.04)])
@pytest.mark.parametrize('reason',['profit','loss'])
def test_half_once_budget_stays_and_next_exit_full(tp,sl,reason):
    codes=[f'SZ300{i:03d}' for i in range(1,61)]
    scores={c:60.-i for i,c in enumerate(codes)}
    quotes={c:dict(open=100.,close=100.,factor=1.) for c in codes};prev={c:100. for c in codes}
    a=make_account(5,tp,sl);a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[0];ep=a.positions[c];original=ep.quantity;budget=ep.budget
    price=ep.unit_cost*(1+tp if reason=='profit' else 1-sl*.8)
    prev[c]=price;quotes[c].update(open=price,close=price)
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04')
    assert 0<ep.quantity<=original/2 and ep.budget==budget and ep.half_mode==reason
    remaining=ep.quantity
    # Unchanged price does not refill or halve a second time.
    a.step('2022-01-06',2,scores,quotes,prev,signal_day='2022-01-05')
    assert ep.quantity==remaining and ep.budget==budget
    target=ep.half_price*(1+tp) if reason=='profit' else ep.unit_cost*(1+tp)
    prev[c]=target;quotes[c].update(open=target,close=target)
    a.step('2022-01-07',3,scores,quotes,prev,signal_day='2022-01-06')
    assert c not in a.positions and ep.budget==budget
    assert len([e for e in a.events if e.get('event')=='half_complete' and e.get('episode_id')==ep.episode_id])==1
    a.reconcile()

def test_source_isolation():
    from paper_futu.rank_buffer_10_20_engine import Account,fee
    assert Account(10).rules.loss_half==.04
    with pytest.raises(ValueError):Account(5)
    assert fee(100000,True)==50
    assert make_account(5,.08,.04).rules.loss_half==pytest.approx(.032)
