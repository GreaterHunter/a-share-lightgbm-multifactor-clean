import pytest
from paper_futu.research_top10_rank80_strong1 import make_account,engine

def sample():
    codes=[f'SZ300{i:03d}' for i in range(1,101)]
    scores={c:100.-i for i,c in enumerate(codes)}
    quotes={c:dict(open=100.,close=100.,factor=1.) for c in codes}
    return codes,scores,quotes,{c:100. for c in codes}

@pytest.mark.parametrize('mode',['full','staged'])
def test_rank80_keep_81_exit(mode):
    codes,scores,quotes,prev=sample();a=make_account(mode,'6_2')
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[9];scores[c]=20.5
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04');assert c in a.positions
    scores[c]=19.5
    a.step('2022-01-06',2,scores,quotes,prev,signal_day='2022-01-05');assert c not in a.positions
    assert a.episodes[-1]['exit_reason']=='rank_exit'

@pytest.mark.parametrize('mode',['full','staged'])
def test_only_new_rank1_replaces(mode):
    codes,scores,quotes,prev=sample();a=make_account(mode,'10_5')
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31');n=len(a.orders)
    incoming=codes[10];scores[incoming]=99.5
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04')
    assert incoming not in a.positions and len(a.orders)==n
    scores[incoming]=101
    a.step('2022-01-06',2,scores,quotes,prev,signal_day='2022-01-05')
    assert incoming in a.positions and len([o for o in a.orders if o['reason']=='strong_replacement'])==1
    n=len(a.orders)
    a.step('2022-01-07',3,scores,quotes,prev,signal_day='2022-01-06');assert len(a.orders)==n

def test_6_2_boundaries():
    for mode in ('full','staged'):
        a=make_account(mode,'6_2');m=engine(mode)
        ep=m.Episode(1,'SZ300001','2022-01-04',0,1e6,1000,1000,100000,2000,100)
        assert m.price_action(ep,106,1,a.rules,0)==(('profit_full',True) if mode=='full' else ('profit_half',False))
        assert m.price_action(ep,98,1,a.rules,0)==('loss_budget',True)
        assert m.price_action(ep,98.4,1,a.rules,0)==(None if mode=='full' else ('loss_half',False))
        if mode=='staged':assert a.rules.trailing_drawdown==.02 and a.rules.loss_half==.016

@pytest.mark.parametrize('mode',['full','staged'])
@pytest.mark.parametrize('reason,delay',[('profit_full',2),('loss_budget',3)])
def test_cooldown_unchanged(mode,reason,delay):
    codes,scores,quotes,prev=sample();a=make_account(mode,'6_2')
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[0];ep=a.positions[c];a.intent(ep,reason,True,'2022-01-04',1);a.sell(ep,'2022-01-05',1,quotes[c],100)
    assert a.cooldown[c]==1+delay
