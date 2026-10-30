from copy import deepcopy
import pytest
from paper_futu.cn_rank80_strategy import bootstrap,decide,apply_fills,price_reason,estimated_fee

CODES=[f'SZ.{i:06d}' for i in range(1,101)]


def make_state(n=10):
    scores={c:100-i for i,c in enumerate(CODES)}
    signal=dict(predictions=scores,closes={c:100. for c in CODES},factors={c:1. for c in CODES})
    positions={c:dict(qty=100,can_sell_qty=100,average_cost=100.,cost_price_valid=True) for c in CODES[:n]}
    state=bootstrap(positions,1e6,signal);state['initialized']=True
    return state,signal,positions


def select(state,signal,positions,previous=None):
    return decide(state,signal,previous or signal['predictions'],positions,set(CODES),set(positions),'2026-10-29')


def test_retained_no_daily_reweight():
    st,sig,pos=make_state()
    d=select(st,sig,pos)
    assert not d['new_codes'] and not d['reasons'] and len(d['selected'])==10


def test_rank80_boundary_and_missing():
    st,sig,pos=make_state(1);c=CODES[0]
    sig['predictions'][c]=20.5
    assert c not in select(st,sig,pos)['reasons']
    sig['predictions'][c]=19.5
    assert select(st,sig,pos)['reasons'][c]=='rank_exit'


def test_only_new_first_strictly_higher():
    st,sig,pos=make_state();prev=dict(sig['predictions']);sig['predictions'][CODES[20]]=101
    d=select(st,sig,pos,prev)
    assert d['reasons']=={CODES[9]:'strong_entry'} and CODES[20] in d['new_codes']
    assert select(st,sig,pos)['reasons']=={}


def test_new_second_does_not_force_replace():
    st,sig,pos=make_state();prev=dict(sig['predictions']);sig['predictions'][CODES[20]]=99.5
    assert select(st,sig,pos,prev)['reasons']=={}


@pytest.mark.parametrize('close,reason',[(110.,'profit_full'),(95.,'loss_budget'),(100.,'')])
def test_price_full_exits(close,reason):
    st,sig,pos=make_state(1)
    assert price_reason(st['positions'][CODES[0]],close,1.)==reason


def test_pending_exit_not_cancelled_by_rebound():
    st,sig,pos=make_state(1);st['positions'][CODES[0]]['pending_reason']='loss_budget'
    sig['closes'][CODES[0]]=105
    assert select(st,sig,pos)['reasons'][CODES[0]]=='loss_budget'


def test_no_same_day_rebuy():
    st,sig,pos=make_state();sig['closes'][CODES[0]]=111
    d=select(st,sig,pos)
    assert CODES[0] not in d['new_codes'] and len(d['selected'])==10


def test_tplus_one_occupies_slot():
    st,sig,pos=make_state();sig['closes'][CODES[0]]=111
    d=decide(st,sig,sig['predictions'],pos,set(CODES),set(pos)-{CODES[0]},'2026-10-29')
    assert not d['new_codes'] and CODES[0] in d['selected']


def test_transition_top20_keep_best_ten_no_topup():
    st,sig,pos=make_state(20);st['initialized']=False
    d=select(st,sig,pos)
    assert len(d['reasons'])==10 and len(d['selected'])==10 and not d['new_codes']


def test_unknown_inventory_and_cost_block():
    st,sig,pos=make_state(1);pos[CODES[0]]['qty']=200
    with pytest.raises(ValueError,match='Inventory'):select(st,sig,pos)
    pos[CODES[0]]['cost_price_valid']=False
    with pytest.raises(ValueError,match='cost'):bootstrap(pos,1e6,sig)


def test_corporate_action_blocks_false_stop():
    st,sig,pos=make_state(1)
    with pytest.raises(ValueError,match='Corporate'):price_reason(st['positions'][CODES[0]],50,2.)


@pytest.mark.parametrize('reason,delay',[('profit_full',2),('loss_budget',3),('rank_exit',0)])
def test_cooldown_from_actual_full_fill_and_idempotence(reason,delay):
    st,sig,pos=make_state(1);c=CODES[0];st['positions'][c]['pending_reason']=reason
    d=select(st,sig,pos)
    order=dict(order_id='1',code=c,trd_side='SELL',dealt_qty=100,dealt_avg_price=100.)
    calendar=['2026-10-29','2026-10-30','2026-11-02','2026-11-03']
    out=apply_fills(d,[order],{},calendar[0],calendar,sig['factors'],1e6)
    assert out==apply_fills(d,[order],{},calendar[0],calendar,sig['factors'],1e6)
    if delay:assert out['cooldown'][c]==calendar[delay]
    else:assert c not in out['cooldown']


def test_partial_exit_keeps_budget_and_no_cooldown():
    st,sig,pos=make_state(1);c=CODES[0];st['positions'][c]['pending_reason']='loss_budget'
    d=select(st,sig,pos)
    order=dict(order_id='1',code=c,trd_side='SELL',dealt_qty=50,dealt_avg_price=95.)
    out=apply_fills(d,[order],{c:50},'2026-10-29',['2026-10-29'],sig['factors'],1e6)
    assert out['positions'][c]['budget']==st['positions'][c]['budget']
    assert out['positions'][c]['pending_reason']=='loss_budget' and not out['cooldown']


def test_new_entry_aggregate_fills_fees_once():
    st,sig,pos=make_state(0);d=select(st,sig,pos);c=CODES[0]
    orders=[dict(order_id=str(i),code=c,trd_side='BUY',dealt_qty=100,dealt_avg_price=100.) for i in (1,2)]
    out=apply_fills(d,orders,{c:200},'2026-10-29',['2026-10-29'],sig['factors'],1e6)
    assert out['positions'][c]['initial_cost']==20000+2*estimated_fee(10000,True)
    with pytest.raises(ValueError,match='Duplicate'):apply_fills(d,orders+orders,{c:200},'2026-10-29',['2026-10-29'],sig['factors'],1e6)


def test_cooldown_eligibility():
    st,sig,pos=make_state(0);c=CODES[0];st['cooldown'][c]='2026-10-30'
    assert c not in select(st,sig,pos)['new_codes']
    st['cooldown'][c]='2026-10-29'
    assert c in select(st,sig,pos)['new_codes']


@pytest.mark.parametrize('pending',['loss_budget','profit_full','rank_exit'])
@pytest.mark.parametrize('has_close',[True,False])
def test_pending_exit_factor_change_blocks_without_mutating_ledger(pending,has_close):
    st,sig,pos=make_state(1);c=CODES[0]
    st['positions'][c]['pending_reason']=pending
    sig['factors'][c]=1.02  # Verified cash dividend; broker share count is unchanged.
    if not has_close:sig['closes'].pop(c)
    original=deepcopy(st);original_pos=deepcopy(pos)
    with pytest.raises(ValueError,match='Corporate action.*reconciliation'):
        select(st,sig,pos)
    assert st==original and pos==original_pos


@pytest.mark.parametrize('pending',['','loss_budget'])
@pytest.mark.parametrize('has_close',[True,False])
def test_changed_factor_blocks_before_exit_reason_or_close(pending,has_close):
    st,sig,pos=make_state(1);c=CODES[0]
    st['positions'][c]['pending_reason']=pending
    sig['factors'][c]=1.02
    if not has_close:sig['closes'].pop(c)
    with pytest.raises(ValueError,match='Corporate action.*reconciliation'):
        select(st,sig,pos)


@pytest.mark.parametrize('source',['current','ledger'])
@pytest.mark.parametrize('invalid',[None,float('nan'),float('inf'),float('-inf'),0.,-1.])
def test_pending_exit_requires_valid_current_and_ledger_factors(source,invalid):
    st,sig,pos=make_state(1);c=CODES[0]
    st['positions'][c]['pending_reason']='loss_budget'
    target,key=(sig['factors'],c) if source=='current' else (st['positions'][c],'factor')
    if invalid is None:target.pop(key)
    else:target[key]=invalid
    import json
    original=json.dumps(st,sort_keys=True)
    with pytest.raises(ValueError,match='Corporate action.*reconciliation'):
        select(st,sig,pos)
    assert json.dumps(st,sort_keys=True)==original


@pytest.mark.parametrize('has_close',[True,False])
def test_unchanged_factor_preserves_pending_exit_and_migrated_cost(has_close):
    st,sig,pos=make_state(1);c=CODES[0]
    st['positions'][c]['pending_reason']='loss_budget'
    if has_close:sig['closes'][c]=110.  # A rebound must not replace the original intent.
    else:sig['closes'].pop(c)
    original=deepcopy(st)
    d=select(st,sig,pos)
    assert d['reasons'][c]=='loss_budget'
    assert d['state_before']['positions'][c]==original['positions'][c]
    assert st==original and st['positions'][c]['migrated']


def test_partial_exit_factor_check_preserves_original_cost_and_budget():
    st,sig,pos=make_state(1);c=CODES[0]
    st['positions'][c]['pending_reason']='loss_budget'
    decision=select(st,sig,pos)
    order=dict(order_id='partial',code=c,trd_side='SELL',dealt_qty=50,dealt_avg_price=95.)
    after=apply_fills(decision,[order],{c:50},'2026-10-29',['2026-10-29'],sig['factors'],1e6)
    pos[c].update(qty=50,can_sell_qty=50)
    original=deepcopy(after)
    continued=select(after,sig,pos)
    assert continued['reasons'][c]=='loss_budget'
    assert continued['state_before']['positions'][c]==original['positions'][c]
    assert original['positions'][c]['initial_cost']==st['positions'][c]['initial_cost']
    assert original['positions'][c]['budget']==st['positions'][c]['budget']
    assert original['positions'][c]['sold_net']>0
    sig['factors'][c]=1.02
    with pytest.raises(ValueError,match='Corporate action.*reconciliation'):
        select(after,sig,pos)
    assert after==original


@pytest.mark.parametrize('factor,blocked',[(1.+5e-7,False),(1.+2e-6,True)])
def test_pending_exit_keeps_original_factor_tolerance(factor,blocked):
    st,sig,pos=make_state(1);c=CODES[0]
    st['positions'][c]['pending_reason']='rank_exit';sig['factors'][c]=factor
    if blocked:
        with pytest.raises(ValueError,match='Corporate action.*reconciliation'):
            select(st,sig,pos)
    else:
        assert select(st,sig,pos)['reasons'][c]=='rank_exit'


@pytest.mark.parametrize('close',[0.,-1.,float('nan'),float('inf')])
def test_price_reason_checks_factor_before_invalid_close_return(close):
    st,sig,pos=make_state(1)
    with pytest.raises(ValueError,match='Corporate action.*reconciliation'):
        price_reason(st['positions'][CODES[0]],close,1.02)


def test_first_migration_keeps_verified_average_cost_with_unchanged_factor():
    st,sig,pos=make_state(20);st['initialized']=False
    original=deepcopy(st)
    decision=select(st,sig,pos)
    assert decision['transition'] and len(decision['selected'])==10
    for c in decision['selected']:
        assert decision['state_before']['positions'][c]==original['positions'][c]
        assert decision['state_before']['positions'][c]['cost_source'].startswith('broker_average_cost')
    assert st==original
