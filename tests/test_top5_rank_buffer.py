import math
import struct

import pytest

from paper_futu.top5_buffer_engine import (Episode,Account,fee,price_action,half_quantity,
                                         buy_quantity,ranking,liquidation_breakeven)
from paper_futu.top5_buffer_data import truncate_binary


def episode():
    return Episode(1,'SZ300001','2022-01-04',0,1e6,1900,1900,190000,19000,100)


@pytest.mark.parametrize('case',list('BCDE'))
def test_budget_priority_over_rank_and_half(case):
    ep=episode()
    assert price_action(ep,90.1,1,case,0)==('loss_budget',True)
    assert ep.budget==19000


@pytest.mark.parametrize('rank',[5,6])
def test_stop_variants_at_eight_percent(rank):
    ep=episode()
    assert price_action(ep,92,rank,'B',0) is None
    assert price_action(ep,92,rank,'C',0)==(('loss_half',False) if rank==5 else ('early_stop',True))
    assert price_action(ep,92,rank,'D',0)==(('loss_half',False) if rank==5 else None)
    assert price_action(ep,92,rank,'A',0) is None


@pytest.mark.parametrize('rank',[5,6])
def test_profit_boundary(rank):
    ep=episode()
    assert price_action(ep,124-1e-6,rank,'D',0) is None
    assert price_action(ep,124,rank,'D',0)==(('profit_half',False) if rank==5 else ('profit_full',True))


def test_profit_half_trailing_only_up_and_no_second_for_E():
    ep=episode();ep.quantity=900;ep.half_mode='profit';ep.half_price=124
    ep.sold_net=124000-fee(124000,False)
    assert price_action(ep,125,1,'D',0) is None
    first=ep.trailing
    assert price_action(ep,124,1,'D',0) is None
    assert ep.trailing>=first
    assert price_action(ep,124*1.24,1,'D',0)==('second_profit',True)
    assert price_action(ep,124*1.24,1,'E',0) is None
    assert price_action(ep,130,1,'E',0)==('profit_trailing',True)


def test_loss_half_preserves_original_profit_target_and_budget():
    ep=episode();ep.quantity=900;ep.half_mode='loss';ep.sold_net=92000-fee(92000,False)
    stop=ep.remaining_budget_price(0)
    assert stop < 90
    assert price_action(ep,stop,2,'D',0)==('loss_budget',True)
    assert price_action(ep,124,2,'D',0)==('profit_after_loss_half',True)
    assert ep.budget==19000


@pytest.mark.parametrize('code,q,expected', [('SZ300001',100,100),('SZ300001',300,200),
                                            ('SH688001',200,200),('SH688001',500,250),
                                            ('SZ300001',150,100)])
def test_legal_half(code,q,expected):
    assert half_quantity(code,q,1)==expected


def test_factor_adjusted_half_and_affordability():
    assert half_quantity('SZ300001',300/2,2)==100
    q=buy_quantity('SZ300001',10005,200,2)
    assert q==50 and q*200+fee(q*200,True)<=10005
    assert buy_quantity('SH688001',10000,100,1)==0
    assert buy_quantity('SZ300001',4,1,1)==0


def test_tie_stable_order():
    codes,ranks=ranking({'SZ300003':1,'SZ300001':1,'SZ300002':1})
    assert codes==['SZ300001','SZ300002','SZ300003'] and ranks['SZ300003']==3


def inputs(n=20,price=100):
    codes=[f'SZ300{i:03d}' for i in range(1,n+1)]
    scores={c:float(n-i) for i,c in enumerate(codes)}
    quotes={c:dict(open=price,close=price,factor=1.) for c in codes}
    prev={c:float(price) for c in codes}
    return codes,scores,quotes,prev


def test_no_daily_equalization_or_topup_after_half():
    codes,scores,quotes,prev=inputs()
    a=Account('D');a.step('2022-01-04',0,scores,quotes,prev)
    original=a.positions[codes[0]].quantity
    for c in codes:prev[c]=100
    prev[codes[0]]=124.2;quotes[codes[0]]=dict(open=124.2,close=124.2,factor=1.)
    a.step('2022-01-05',1,scores,quotes,prev)
    reduced=a.positions[codes[0]].quantity
    assert reduced<original and a.positions[codes[0]].half_mode=='profit'
    a.step('2022-01-06',2,scores,quotes,prev)
    assert a.positions[codes[0]].quantity==reduced
    assert len([o for o in a.orders if o['direction']=='BUY' and o['code']==codes[0]])==1
    assert a.reconcile()['status']=='passed'


def test_rank_15_kept_16_exits():
    codes,scores,quotes,prev=inputs()
    a=Account('A');a.step('2022-01-04',0,scores,quotes,prev)
    code=codes[4];scores[code]=5.5 # ranks: original score6/5 => rank15
    a.step('2022-01-05',1,scores,quotes,prev)
    assert code in a.positions
    scores[code]=4.5
    a.step('2022-01-06',2,scores,quotes,prev)
    assert code not in a.positions and a.episodes[-1]['exit_reason']=='rank_exit'


def test_two_new_top2_replace_two_worst_and_no_same_day_rebuy():
    codes,scores,quotes,prev=inputs()
    a=Account('A');a.step('2022-01-04',0,scores,quotes,prev)
    scores[codes[5]]=100.;scores[codes[6]]=99.
    a.step('2022-01-05',1,scores,quotes,prev)
    reasons=[o for o in a.orders if o['date']=='2022-01-05' and o['direction']=='SELL']
    assert len(reasons)==2
    assert all(o['reason']=='strong_replacement' for o in reasons)
    assert codes[5] in a.positions and codes[6] in a.positions


def test_identical_score_does_not_force_replacement():
    codes,scores,quotes,prev=inputs()
    a=Account('A');a.step('2022-01-04',0,scores,quotes,prev)
    # A name enters first two through deterministic tie ordering, but worst held
    # has identical score. No strict score advantage => no forced turnover.
    a.previous_ranks[codes[5]]=3
    scores={c:1. for c in codes};scores[codes[0]]=0.;scores[codes[1]]=0.
    # Test a worst-held tie directly by owning codes 6..10 and ranking all equal.
    b=Account('A')
    b.step('2022-01-04',0,{c:100-i if i>=5 and i<10 else -i for i,c in enumerate(codes)},quotes,prev)
    b.previous_ranks[codes[0]]=3;b.previous_ranks[codes[1]]=4
    b.step('2022-01-05',1,{c:1. for c in codes},quotes,prev)
    assert not any(o['date']=='2022-01-05' and o['reason']=='strong_replacement' for o in b.orders)


def test_cooldown_D_plus_3_not_3_more_days():
    codes,scores,quotes,prev=inputs()
    a=Account('B');a.step('2022-01-04',0,scores,quotes,prev)
    c=codes[0];prev[c]=89;quotes[c]=dict(open=89,close=89,factor=1.)
    a.step('2022-01-05',1,scores,quotes,prev)
    assert c not in a.positions and a.cooldown[c]==4
    # Clear one slot using rank exit each day to actually challenge reentry.
    for step in (2,3,4):
        held=next(iter(a.positions));scores[held]=-100-step
        a.step(f'2022-01-{step+4:02d}',step,scores,quotes,prev)
        assert (c in a.positions)==(step==4)


def test_blocked_exit_persists_after_rebound_and_occupies_slot():
    codes,scores,quotes,prev=inputs()
    a=Account('B');a.step('2022-01-04',0,scores,quotes,prev)
    c=codes[0];prev[c]=89;quotes[c]=dict(open=80,close=80,factor=1.)
    a.step('2022-01-05',1,scores,quotes,prev)
    assert c in a.positions and a.positions[c].pending_full and len(a.positions)==5
    prev[c]=100;quotes[c]=dict(open=100,close=100,factor=1.)
    a.step('2022-01-06',2,scores,quotes,prev)
    assert c not in a.positions and a.episodes[-1]['exit_reason']=='loss_budget'


def test_partial_fill_restart_preserves_budget_and_single_half():
    import copy
    codes,scores,quotes,prev=inputs()
    a=Account('D');a.step('2022-01-04',0,scores,quotes,prev)
    c=codes[0];ep=a.positions[c];budget=ep.budget
    prev[c]=ep.unit_cost*.92;quotes[c]=dict(open=prev[c],close=prev[c],factor=1.,sell_capacity_raw=100)
    a.step('2022-01-05',1,scores,quotes,prev)
    assert ep.half_mode=='loss' and ep.pending_quantity>0 and ep.budget==budget
    b=copy.deepcopy(a)
    for obj in (a,b):obj.step('2022-01-06',2,scores,quotes,prev)
    assert a.orders==b.orders and a.positions[c].budget==budget
    assert len([e for e in a.events if e['event']=='intent' and e.get('code')==c])==1


def test_T1_suspension_and_gap_not_clipped():
    ep=episode();a=Account('D');a.positions[ep.code]=ep
    a.intent(ep,'loss_budget',True,'2022-01-04',1.)
    quote=dict(open=80.,close=80.,factor=1.)
    assert not a.sell(ep,'2022-01-04',0,quote,89.) # T+1
    assert ep.pending_full
    quote['suspended']=True
    assert not a.sell(ep,'2022-01-05',1,quote,80.)
    quote['suspended']=False
    a.sell(ep,'2022-01-06',2,quote,80.)
    assert a.orders[-1]['price']==80 and a.episodes[-1]['budget_overshoot']>0


def test_open_episode_end_mark_not_liquidation():
    codes,scores,quotes,prev=inputs()
    a=Account('D');a.step('2025-12-31',0,scores,quotes,prev)
    opened=a.open_episodes('2025-12-31',0)
    assert len(opened)==5 and not a.episodes
    assert all(o['direction']=='BUY' for o in a.orders)
    assert a.reconcile()['status']=='passed'


def test_future_binary_suffix_deleted_or_perturbed_is_identical(tmp_path):
    source=tmp_path/'source.bin';result=tmp_path/'out.bin'
    prefix=struct.pack('<5f',0,1,2,3,4)
    source.write_bytes(prefix+struct.pack('<2f',999,math.nan))
    truncate_binary(source,result,4);original=result.read_bytes()
    source.write_bytes(prefix+struct.pack('<100f',*[1e10]*100))
    truncate_binary(source,result,4);assert result.read_bytes()==original==prefix
    source.write_bytes(prefix)
    truncate_binary(source,result,4);assert result.read_bytes()==original


def test_provider_preserves_historical_member_without_inventing_missing_prices(tmp_path,monkeypatch):
    from paper_futu import top5_buffer_data as data
    source=tmp_path/'source';provider=source/'provider'
    (provider/'calendars').mkdir(parents=True)
    (provider/'instruments').mkdir()
    (provider/'calendars/day.txt').write_text('2025-12-30\n2025-12-31\n2026-01-05\n',encoding='utf8')
    (provider/'instruments/csi300.txt').write_text('SH600001\t2010-01-01\t2026-10-13\n',encoding='utf8')
    monkeypatch.setattr(data,'SOURCE',source)
    out=tmp_path/'out'
    view,dates,codes=data.provider_view(out)
    assert codes=={'SH600001'} and str(dates[-1].date())=='2025-12-31'
    assert not list((view/'features').glob('**/*.bin'))
    assert len(data.read(out/'provider_prefix_manifest.json')['missing_source_fields'])==6


def test_initial_budget_and_cash_floor():
    codes,scores,quotes,prev=inputs()
    a=Account('D',.002);a.step('2022-01-04',0,scores,quotes,prev)
    assert a.cash>=50000
    assert all(e.budget==min(20000,e.initial_cost*.1) for e in a.positions.values())
    assert all(e.initial_cost<=190000 for e in a.positions.values())


def test_minimum_fee_breakeven():
    p=liquidation_breakeven(100,100,0)
    assert 100*p-fee(100*p,False)>=100-1e-8


def test_user_authorized_memory_observation_does_not_block(monkeypatch):
    from src import runtime_budget
    from paper_futu.top5_buffer_data import observe_memory
    monkeypatch.setattr(runtime_budget,'available_memory_bytes',lambda:100*1024**2)
    assert observe_memory(1.5,'test') < 1.5


def test_current_close_or_volume_cannot_change_open_orders():
    import copy
    codes,scores,quotes,prev=inputs()
    a,b=Account('D'),Account('D')
    alternate=copy.deepcopy(quotes)
    for q in alternate.values():q.update(close=99999.,volume=0.)
    a.step('2022-01-04',0,scores,quotes,prev)
    b.step('2022-01-04',0,scores,alternate,prev)
    assert a.orders==b.orders


def test_explicit_missing_open_never_falls_back_to_same_day_close():
    codes,scores,quotes,prev=inputs()
    quotes[codes[0]]['open']=float('nan')
    a=Account('A');a.step('2022-01-04',0,scores,quotes,prev)
    assert codes[0] not in a.positions and len(a.positions)==5


def test_half_does_not_free_slot():
    codes,scores,quotes,prev=inputs()
    a=Account('D');a.step('2022-01-04',0,scores,quotes,prev)
    before=set(a.positions)
    prev[codes[0]]=124.2;quotes[codes[0]].update(open=124.2,close=124.2)
    a.step('2022-01-05',1,scores,quotes,prev)
    assert set(a.positions)==before


def test_price_risk_queued_exit_is_not_relabelled_rank_exit():
    codes,scores,quotes,prev=inputs()
    a=Account('B');a.step('2022-01-04',0,scores,quotes,prev)
    c=codes[0];prev[c]=89;quotes[c].update(open=89,close=89);scores[c]=-100
    a.step('2022-01-05',1,scores,quotes,prev)
    assert a.episodes[-1]['exit_reason']=='loss_budget' and a.cooldown[c]==4


def test_saved_cash_nav_and_episode_reconcile_independently():
    import pandas as pd
    from paper_futu.top5_buffer_report import independent_audit,metrics
    codes,scores,quotes,prev=inputs()
    a=Account('D',.001)
    a.step('2022-01-04',0,scores,quotes,prev)
    scores[codes[4]]=-100
    a.step('2022-01-05',1,scores,quotes,prev)
    daily,orders,holdings=[pd.DataFrame(x) for x in (a.daily,a.orders,a.holdings)]
    for f in (daily,orders,holdings):f['date']=pd.to_datetime(f.date)
    closed=pd.DataFrame(a.episodes);opened=pd.DataFrame(a.open_episodes('2022-01-05',1))
    assert independent_audit(daily,orders,holdings,closed,opened)['status']=='passed'
    r=metrics(daily,pd.Series([0.,0.],index=daily.date))
    assert r['total_return']<0 and r['fees']>0 and r['max_drawdown']>0
    orders.loc[0,'fee']+=10
    with pytest.raises(AssertionError):independent_audit(daily,orders,holdings,closed,opened)


def test_filtered_parquet_ignores_future_rows(tmp_path):
    import pandas as pd
    from paper_futu.top5_buffer_data import bounded_parquet,fingerprint
    path=tmp_path/'features.parquet'
    f=pd.DataFrame({'datetime':pd.to_datetime(['2025-12-31','2026-01-05']), 'x':[1.,2.]})
    f.to_parquet(path,index=False)
    before=fingerprint(bounded_parquet(path))
    f.loc[1,'x']=1e20;f.to_parquet(path,index=False)
    assert fingerprint(bounded_parquet(path))==before
    f.iloc[:1].to_parquet(path,index=False)
    assert fingerprint(bounded_parquet(path))==before


def test_fold_selection_rejects_future_year_and_maturity():
    from paper_futu.top5_buffer_data import choice_from_folds
    rows=[dict(validation_year=y,train_label_end_max=f'{y-1}-12-31',valid_label_end_max=f'{y}-12-31',
               parameter_id='p00',trees=t,num_leaves=31,mean_rank_ic=.1 if t==20 else .05)
          for y in (2019,2020,2021) for t in (10,20)]
    assert choice_from_folds(rows,2022)==20
    rows[0]['train_label_end_max']='2019-01-01'
    with pytest.raises(ValueError):choice_from_folds(rows,2022)
    rows[0]['train_label_end_max']='2018-12-31';rows[0]['validation_year']=2026
    with pytest.raises(ValueError):choice_from_folds(rows,2022)


def test_full_report_on_synthetic_15_accounts(tmp_path,monkeypatch):
    """Exercise artifact/report wiring, never substitute these for real results."""
    import pandas as pd
    from paper_futu import research_top5_rank_buffer as runner
    from paper_futu.top5_buffer_data import save,digest
    from paper_futu.top5_buffer_report import report
    monkeypatch.setattr(runner,'OUT',tmp_path)
    save(tmp_path/'protocol.json',dict(implementation=runner.scientific_hashes(),protected={},limitations=['synthetic test only']))
    save(tmp_path/'audit_complete.json',dict(artifacts={}))
    (tmp_path/'data').mkdir()
    dates=['2022-01-04','2023-01-03','2024-01-02','2025-12-30','2025-12-31']
    pd.DataFrame({'benchmark_return':[0.,.01,-.01,.02,0.]},index=pd.to_datetime(dates)).to_parquet(tmp_path/'data/benchmark.parquet')
    for case in 'ABCDE':
        for bp in (0,10,20):
            codes,scores,quotes,prev=inputs()
            a=Account(case,bp/10000)
            for i,day in enumerate(dates):
                if i:
                    for c in list(a.positions):scores[c]=-100-i
                    scores[codes[10+i]]=100+i
                a.step(day,i,scores,quotes,prev)
            directory=tmp_path/'accounts'/f'{case}_{bp:02d}bp';directory.mkdir(parents=True)
            records=dict(daily=a.daily,orders=a.orders,events=a.events,holdings=a.holdings,
                         episodes=a.episodes,open_episodes=a.open_episodes(dates[-1],4))
            for name,values in records.items():pd.DataFrame(values).to_parquet(directory/(name+'.parquet'),index=False)
            save(directory/'complete.json',dict(protocol_sha256=digest(tmp_path/'protocol.json'),
                artifacts={str(p):digest(p) for p in directory.iterdir()}))
    report(tmp_path)
    assert (tmp_path/'report.html').exists()
    assert len(pd.read_csv(tmp_path/'comparison.csv'))==15
    assert len(pd.read_csv(tmp_path/'yearly.csv'))==60
