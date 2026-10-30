from datetime import datetime
from copy import deepcopy
import json
import pytest
from paper_futu import cn_rank80_runtime as runtime
from paper_futu.cn_common import CHINA_TZ
from paper_futu.cn_rank80_strategy import bootstrap,apply_fills,price_reason
from paper_futu.rank_buffer_full_exit_engine import Episode,Rules,price_action
from paper_futu.research_rank_buffer_futu_fees import fee


@pytest.mark.parametrize('close',[80.,94.,95.,96.,99.,100.,109.,110.,111.])
def test_price_trigger_matches_frozen_backtest(monkeypatch,close):
    from paper_futu import rank_buffer_full_exit_engine as research
    monkeypatch.setattr(research,'fee',fee)
    ep=Episode(1,'SZ000001','2026-10-20',0,1e6,100.,100.,10025.,501.25,100.25)
    live=dict(qty=100,initial_cost=10025.,budget=501.25,unit_cost=100.25,sold_net=0.,factor=1.)
    expected=price_action(ep,close,1,Rules(10),0.)
    assert price_reason(live,close,1.)==(expected[0] if expected else '')


@pytest.fixture
def worker(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'DIRECTORY',tmp_path/'rank')
    runtime.DIRECTORY.mkdir()
    monkeypatch.setattr(runtime,'STATE_DIR',tmp_path)
    monkeypatch.setattr(runtime,'enabled',lambda:True)
    monkeypatch.setattr(runtime,'dates',lambda:['2026-10-27','2026-10-28','2026-10-29','2026-10-30'])
    import paper_futu.update_cn_data as data
    monkeypatch.setattr(data,'completed_date',lambda *a:'2026-10-28')
    monkeypatch.setattr(data,'calendar',lambda:None)
    (tmp_path/'2026-10-28').mkdir()
    (tmp_path/'2026-10-28/independent_data_ready.json').write_text('{}')
    def forbidden(*a,**k):raise AssertionError('Unexpected submission or broker call')
    monkeypatch.setattr(runtime,'execute',forbidden)
    return tmp_path


@pytest.mark.parametrize('hour,minute,stage',[(8,50,'outside_session'),(9,20,'waiting_for_0931'),(9,51,'window_closed_next_session_only'),(13,0,'window_closed_next_session_only'),(17,0,'outside_session')])
def test_no_out_of_window_trading(worker,monkeypatch,hour,minute,stage):
    monkeypatch.setattr(runtime,'china_now',lambda:datetime(2026,10,29,hour,minute,tzinfo=CHINA_TZ))
    assert runtime.tick()['stage']==stage


def test_offline_retry_does_not_create_intent(worker,monkeypatch):
    monkeypatch.setattr(runtime,'china_now',lambda:datetime(2026,10,29,9,35,tzinfo=CHINA_TZ))
    def offline(*a,**k):raise OSError('offline')
    monkeypatch.setattr(runtime.socket,'create_connection',offline)
    assert runtime.tick()['stage']=='waiting_for_opend_login'
    assert not list(runtime.DIRECTORY.glob('*/manifest.json'))


def test_completed_and_blocked_batches_never_resubmit(worker,monkeypatch):
    monkeypatch.setattr(runtime,'china_now',lambda:datetime(2026,10,29,9,35,tzinfo=CHINA_TZ))
    folder=runtime.DIRECTORY/'2026-10-29';folder.mkdir()
    (folder/'blocked.json').write_text('{"error":"ambiguous"}')
    assert runtime.tick()['stage']=='requires_review'
    (folder/'committed.json').write_text('{}')
    assert runtime.tick()['stage']=='already_completed'


def test_initial_resize_preserves_cost_and_only_once():
    c='SZ.000001';sig=dict(factors={c:1.})
    pos={c:dict(qty=100,can_sell_qty=100,average_cost=100.,cost_price_valid=True)}
    st=bootstrap(pos,1e6,sig)
    d=dict(state_before=st,initial_allocation=True,new_codes=[],selected=[c],reasons={})
    orders=[dict(order_id='1',code=c,trd_side='BUY',dealt_qty=100,dealt_avg_price=110.)]
    out=apply_fills(d,orders,{c:200},'2026-10-29',['2026-10-29'],sig['factors'],1e6)
    assert 105<out['positions'][c]['unit_cost']<106
    d['state_before']=out;d['initial_allocation']=False
    with pytest.raises(ValueError,match='top-up'):
        apply_fills(d,orders,{c:300},'2026-10-29',['2026-10-29'],sig['factors'],1e6)
