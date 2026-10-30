import pytest
from paper_futu.research_top10_rank65_cooldown import make_account,cooldown_delay

def sample():
    codes=[f'SZ300{i:03d}' for i in range(1,81)]
    scores={c:80.-i for i,c in enumerate(codes)}
    quotes={c:dict(open=100.,close=100.,factor=1.) for c in codes}
    return codes,scores,quotes,{c:100. for c in codes}

@pytest.mark.parametrize('mode',['full','staged'])
def test_rank65_kept_66_exits(mode):
    codes,scores,quotes,prev=sample();a=make_account(mode,.10,.05)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[9];scores[c]=15.5
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04')
    assert c in a.positions
    scores[c]=14.5
    a.step('2022-01-06',2,scores,quotes,prev,signal_day='2022-01-05')
    assert c not in a.positions and a.episodes[-1]['exit_reason']=='rank_exit'
    assert c not in a.cooldown

@pytest.mark.parametrize('mode',['full','staged'])
@pytest.mark.parametrize('reason,delay',[('profit_full',2),('profit_half',2),('second_profit',2),('profit_after_loss_half',2),('loss_budget',3),('hard_stop',3),('loss_half',3),('profit_trailing',3)])
def test_exit_reason_applies_exact_cooldown(mode,reason,delay):
    codes,scores,quotes,prev=sample();a=make_account(mode,.10,.05)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[0];ep=a.positions[c]
    a.intent(ep,reason,True,'2022-01-04',1)
    assert a.sell(ep,'2022-01-05',1,quotes[c],prev[c])
    assert a.cooldown[c]==1+delay

@pytest.mark.parametrize('mode',['full','staged'])
@pytest.mark.parametrize('reason,delay',[('profit_full',2),('loss_budget',3)])
def test_reentry_blocked_until_exact_day(mode,reason,delay):
    codes,scores,quotes,prev=sample();a=make_account(mode,.10,.05)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[0];a.intent(a.positions[c],reason,True,'2022-01-04',1)
    only={c:100.}
    a.step('2022-01-05',1,only,quotes,prev,signal_day='2022-01-04')
    assert c not in a.positions
    for step in range(2,1+delay):
        a.step(f'2022-01-0{step+4}',step,only,quotes,prev,signal_day=f'2022-01-0{step+3}')
        assert c not in a.positions
    a.step('2022-01-10',1+delay,only,quotes,prev,signal_day='2022-01-07')
    assert c in a.positions

def test_half_has_no_cooldown():
    codes,scores,quotes,prev=sample();a=make_account('staged',.08,.04)
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[0];a.intent(a.positions[c],'profit_half',False,'2022-01-04',1)
    a.sell(a.positions[c],'2022-01-05',1,quotes[c],prev[c])
    assert c in a.positions and c not in a.cooldown
