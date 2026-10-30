import pytest
from paper_futu.research_rank80_two_thresholds import make_account,engine,CONFIGS,existing
from paper_futu import research_top10_rank65_cooldown as old
from paper_futu import research_top10_rank80_strong1 as prior
from tests.test_top10_rank80_strong1 import sample

def test_bounded_scope():
    assert len(CONFIGS)==28
    assert sum(existing('2026',c) is None for c in CONFIGS)==18

@pytest.mark.parametrize('mode',['full','staged'])
def test_rank80_strong2_boundary(mode):
    codes,scores,quotes,prev=sample();a=make_account(80,2,mode,'10_4')
    a.step('2022-01-04',0,scores,quotes,prev,signal_day='2021-12-31')
    c=codes[9];scores[c]=20.5
    a.step('2022-01-05',1,scores,quotes,prev,signal_day='2022-01-04');assert c in a.positions
    scores[c]=19.5
    a.step('2022-01-06',2,scores,quotes,prev,signal_day='2022-01-05');assert c not in a.positions
    incoming=codes[11];scores[incoming]=99.5
    a.step('2022-01-07',3,scores,quotes,prev,signal_day='2022-01-06')
    assert incoming in a.positions and any(o['reason']=='strong_replacement' for o in a.orders)

@pytest.mark.parametrize('mode',['full','staged'])
@pytest.mark.parametrize('t,tp,sl',[('10_4',.10,.04),('8_3',.08,.03)])
def test_threshold_and_scaling(mode,t,tp,sl):
    a=make_account(80,2,mode,t);m=engine(mode,2)
    ep=m.Episode(1,'SZ300001','2022-01-04',0,1e6,1000,1000,100000,100000*sl,100)
    assert m.price_action(ep,100*(1+tp),1,a.rules,0)==(('profit_full',True) if mode=='full' else ('profit_half',False))
    assert m.price_action(ep,100*(1-sl)-1e-8,1,a.rules,0)==('loss_budget',True)
    if mode=='staged':
        assert a.rules.loss_half==sl*.8 and a.rules.trailing_drawdown==sl
        assert m.price_action(ep,100*(1-sl*.8)-1e-8,1,a.rules,0)==('loss_half',False)

@pytest.mark.parametrize('cfg',[(65,2,'full','10_5'),(65,2,'staged','8_4'),(80,1,'full','6_2'),(80,1,'staged','10_5')])
def test_prior_rules_unchanged(cfg):
    rank,strong,mode,t=cfg;a=make_account(*cfg)
    b=old.make_account(mode,.10,.05) if rank==65 and t=='10_5' else old.make_account(mode,.08,.04) if rank==65 else prior.make_account(mode,t)
    codes,scores,quotes,prev=sample()
    for i in range(6):
        scores[codes[11]]=101 if i>1 else 88
        for x in (a,b):x.step(f'2022-01-{4+i:02d}',i,scores,quotes,prev,signal_day=f'2022-01-{3+i:02d}')
    assert a.orders==b.orders and a.daily==b.daily
