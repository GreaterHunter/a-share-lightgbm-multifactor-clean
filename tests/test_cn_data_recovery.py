"""Recovery remains causal, restartable and independent of OpenD/order execution."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pandas as pd
import pytest

from paper_futu.cn_common import CHINA_TZ
from paper_futu import independent_cn_market as market,update_cn_data as updater
from paper_futu import cn_daily_history as history


pytestmark = pytest.mark.usefixtures('synthetic_issuer_evidence')

def test_diluted_dividend_registry_checks_evidence_and_arithmetic():
    assert market.verified_cash_deduction('SZ000425','2026-10-21',{'fh_sh':'.45'})==.0447445
    assert market.verified_cash_deduction('SZ300433','2026-10-20',{'fh_sh':'1'})==.0999645
    assert market.verified_cash_deduction('SZ000425','2026-10-20',{'fh_sh':'.45'}) is None
    with pytest.raises(ValueError,match='mismatch'):
        market.verified_cash_deduction('SZ000425','2026-10-21',{'fh_sh':'45'})


@pytest.mark.parametrize('instrument,day,prior,dividend,ratio',[
    ('SZ000425','2026-10-21',7.22,.45,1.0062358750684295),
    ('SZ300433','2026-10-20',25.76,1.,1.0038957077067299)])
def test_verified_special_cash_factors_without_relaxing_tolerance(instrument,day,prior,dividend,ratio):
    date=pd.Timestamp(day);prev=date-pd.Timedelta(days=1)
    raw=pd.DataFrame({c:[prior,prior-.1] for c in market.OHLC},index=[prev,date]);raw['volume']=[10000.,20000.]
    raw.attrs['actions']={day:dict(cqr=day,fh_sh=str(dividend),FHcontent='10派'+str(dividend)+'元')}
    old=pd.Series(dict(open=prior*.5,close=prior*.5,high=prior*.5,low=prior*.5,volume=200.,factor=.5),name=(instrument,prev))
    factors=pd.Series([1.,ratio],index=[prev,date])
    rows,_=market.normalize_stock(instrument,old,old,raw,raw,factors,[],[date])
    assert rows[0]['factor']==pytest.approx(.5*ratio)
    factors.iloc[-1]=1.01
    with pytest.raises(ValueError,match='cross-source mismatch'):
        market.normalize_stock(instrument,old,old,raw,raw,factors,[],[date])


def frame(day,instrument='SZ002294',price=10.):
    return pd.DataFrame(dict(open=[price],close=[price],high=[price],low=[price],volume=[100.],factor=[1.]),
        index=pd.MultiIndex.from_tuples([(instrument,pd.Timestamp(day))],names=['instrument','datetime']))


def test_merge_identical_evidence_once_but_never_replace_conflicting_history():
    a=frame('2026-10-19')
    assert len(history.merge_history([a,a]))==1
    with pytest.raises(ValueError,match='Conflicting'):
        history.merge_history([a,frame('2026-10-19',price=11.)])


def test_new_members_own_anchor_backfills_before_the_global_last_day(monkeypatch,tmp_path):
    dates=pd.to_datetime(['2026-10-16','2026-10-19','2026-10-20'])
    # Normalization itself accepts all per-stock missing dates, including the
    # date preceding the current 300-name slice. Saved historical membership stays untouched.
    old=frame('2026-10-16').iloc[0]
    source=pd.DataFrame({c:[10.,11.,12.] for c in market.OHLC},index=dates)
    source['volume']=[10000.,11000.,12000.]
    factors=pd.Series([1.],index=[dates[0]])
    rows,_=market.normalize_stock('SZ002294',old,old,source,source,factors,[],dates[1:])
    assert [r['datetime'] for r in rows]==list(dates[1:])


def test_supplements_are_price_only_and_cannot_change_an_old_bar(tmp_path,monkeypatch):
    monkeypatch.setattr(history,'ROOT',tmp_path);monkeypatch.setattr(history,'STATE_DIR',tmp_path/'state')
    cache=tmp_path/'cache';cache.mkdir()
    result=history.save_supplement('2026-10-20',frame('2026-10-19'),cache)
    p=tmp_path/result
    meta=json.loads(p.with_name('supplement.json').read_text())
    assert meta['historical_membership_asserted'] is False
    with pytest.raises(ValueError,match='Conflicting'):
        history.save_supplement('2026-10-20',frame('2026-10-19',price=11.),cache)
    assert pd.read_parquet(p).iloc[0].close==10.


@pytest.mark.parametrize('day,stamp,allowed,blocked',[
 ('2026-10-26','2026-10-27T08:50:00',True,None),
 ('2026-10-26','2026-10-27T12:00:00',False,'missed_execution_window_no_stale_activation'),
 ('2026-10-27','2026-10-28T08:50:00',False,'missed_rebalance_signal_manual_review'),
 ('2026-10-21','2026-10-22T08:50:00',False,None)])
def test_shutdown_recovery_can_infer_but_cannot_chase_stale_trade(day,stamp,allowed,blocked,monkeypatch):
    import paper_futu.execution_override as override
    monkeypatch.setattr(override,'late_activation_allowed',lambda *a:False)
    monkeypatch.setattr(updater,'calendar',lambda:pd.date_range('2026-10-01','2026-10-31',freq='B'))
    actual=updater.activation_allowed(day,dict(next_rebalance_signal_date='2026-10-26'),datetime.fromisoformat(stamp).replace(tzinfo=CHINA_TZ))
    assert actual==(allowed,blocked)


def test_catchup_uses_saved_universes_in_order_and_never_fakes_missing_members(tmp_path,monkeypatch):
    monkeypatch.setattr(updater,'STATE_DIR',tmp_path)
    monkeypatch.setattr(updater,'china_now',lambda:datetime(2026,10,22,8,tzinfo=CHINA_TZ))
    monkeypatch.setattr(updater,'calendar',lambda:pd.to_datetime(['2026-10-19','2026-10-20','2026-10-21','2026-10-22']))
    for day in ['2026-10-19','2026-10-20']:
        (tmp_path/day).mkdir();(tmp_path/day/'members.csv').write_text('saved')
    (tmp_path/'2026-10-19/independent_data_ready.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(updater,'update',lambda day:calls.append(day) or {'date':day})
    r=updater.catchup()
    assert calls==['2026-10-20','2026-10-21'] and r['orders_submitted']==0


def test_public_data_modules_import_without_futu():
    code="import sys; sys.modules['futu']=None; import paper_futu.update_cn_data; import paper_futu.cn_daily_history; assert 'futu' in sys.modules and sys.modules['futu'] is None"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


def test_public_failure_still_runs_readonly_account_observation(monkeypatch,tmp_path):
    from paper_futu import recover_cn_session as recovery
    monkeypatch.setattr(recovery,'STATE_DIR',tmp_path)
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=1 if 'paper_futu.update_cn_data' in command else 0)
    monkeypatch.setattr(recovery.subprocess,'run',run)
    monkeypatch.setattr(recovery.time,'sleep',lambda s:None)
    r=recovery.recover(2,1)
    assert len(calls)==3 and 'paper_futu.record_cn_account' in calls[-1]
    assert r['trade_executor_started'] is False
    assert all('--submit' not in cmd for cmd in calls)


def test_public_subprocess_timeout_does_not_prevent_account_check(monkeypatch,tmp_path):
    from paper_futu import recover_cn_session as recovery
    monkeypatch.setattr(recovery,'STATE_DIR',tmp_path)
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if 'paper_futu.update_cn_data' in command:raise subprocess.TimeoutExpired(command,1800)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(recovery.subprocess,'run',run)
    r=recovery.recover(1,1)
    assert r['public_attempt_exit_codes']==[124]
    assert 'paper_futu.record_cn_account' in calls[-1]


def test_lock_busy_exit_is_not_claimed_as_fresh_data(monkeypatch,tmp_path):
    from paper_futu import recover_cn_session as recovery
    monkeypatch.setattr(recovery,'STATE_DIR',tmp_path)
    monkeypatch.setattr(recovery.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(updater,'completed_date',lambda *a:'2026-10-21')
    monkeypatch.setattr(updater,'calendar',lambda:pd.to_datetime(['2026-10-21']))
    r=recovery.recover(1,1)
    assert r['public_attempt_exit_codes']==[75] and r['stage']!='recovery_complete'


def test_late_login_observes_now_without_forging_yesterdays_nav(tmp_path,monkeypatch):
    from paper_futu import record_cn_account as account,execute_cn_plan as executor
    now=datetime(2026,10,22,0,10,tzinfo=CHINA_TZ)
    monkeypatch.setattr(account,'STATE_DIR',tmp_path)
    from paper_futu import cn_common
    provider = tmp_path / 'provider'
    (provider / 'calendars').mkdir(parents=True, exist_ok=True)
    (provider / 'calendars/day_future.txt').write_text('2026-10-21\n2026-10-22\n')
    monkeypatch.setattr(cn_common, 'HISTORICAL_PROVIDER', provider)
    monkeypatch.setattr(account,'china_now',lambda:now)
    class Connected:
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Trade:
        def close(self):pass
    monkeypatch.setattr(account.socket,'create_connection',lambda *a,**k:Connected())
    monkeypatch.setattr(executor,'account_context',lambda:Trade())
    monkeypatch.setattr(executor,'paper_account_id',lambda t:123)
    assets=types.SimpleNamespace(cash=100.,total_assets=1000.)
    monkeypatch.setattr(executor,'save_account_snapshot',lambda *a:(assets,pd.DataFrame(dict(qty=[100,0]))))
    r=account.observe(now)
    assert r['stage']=='account_observed' and r['holdings']==1
    assert len(list((tmp_path/'account_observations').glob('*.json')))==1
    assert not list(tmp_path.glob('nav_saved*'))


def test_data_worker_source_keeps_account_branch_after_handled_data_error():
    text=Path('paper_futu/cn_worker.py').read_text(encoding='utf-8')
    assert "status.update(data_stage='error',data_error=str(error))" in text
    assert "from paper_futu.record_cn_account import observe" in text
    assert "total_seconds()>=600" in text


def test_offline_opend_is_a_normal_account_wait_not_a_data_dependency(monkeypatch,tmp_path):
    from paper_futu import record_cn_account as account
    monkeypatch.setattr(account,'STATE_DIR',tmp_path)
    from paper_futu import cn_common
    provider = tmp_path / 'provider'
    (provider / 'calendars').mkdir(parents=True, exist_ok=True)
    (provider / 'calendars/day_future.txt').write_text('2026-10-21\n2026-10-22\n')
    monkeypatch.setattr(cn_common, 'HISTORICAL_PROVIDER', provider)
    def offline(*a,**k):raise ConnectionRefusedError()
    monkeypatch.setattr(account.socket,'create_connection',offline)
    r=account.observe(datetime(2026,10,22,8,tzinfo=CHINA_TZ))
    assert r['stage']=='waiting_for_opend' and r['orders_submitted']==0
    assert not list(tmp_path.glob('nav_saved*'))


def test_scheduler_contains_logon_and_bounded_readonly_retry():
    text=Path('scripts/install_cn_data_schedule.ps1').read_text(encoding='utf-8')
    assert '-AtLogOn' in text and 'PT30S' in text
    assert 'paper_futu.recover_cn_session' in text
    assert 'paper_futu.record_cn_account' in text and '-Minutes 10' in text
    assert 'IgnoreNew' in text and 'StartWhenAvailable' in text
    assert '--submit' not in text
