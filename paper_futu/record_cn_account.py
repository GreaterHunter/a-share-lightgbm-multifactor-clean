"""Observe current CN simulated funds/positions; NEVER submits or cancels orders."""
from __future__ import annotations
import json
import socket
import sys
from paper_futu.cn_common import STATE_DIR,china_now
from paper_futu.independent_cn_market import write_json


def _observe(now=None):
    now=now or china_now()
    status_path=STATE_DIR/'account_recovery_status.json'
    try:
        with socket.create_connection(('127.0.0.1',11111),timeout=2):pass
    except OSError:
        value=dict(stage='waiting_for_opend',time=now.isoformat(),orders_submitted=0,
                   message='Public daily prices do not require OpenD. Current account observation awaits login.')
        write_json(status_path,value);return value
    # Import SDK only AFTER public data completion; data updater never imports it.
    from paper_futu.execute_cn_plan import account_context,paper_account_id,save_account_snapshot
    trade=None
    try:
        trade=account_context()
        aid=paper_account_id(trade)
        assets,positions=save_account_snapshot(trade,aid)
        held=positions.loc[positions.qty.gt(0)]
        observed=china_now()
        value=dict(stage='account_observed',time=observed.isoformat(),cash=float(assets.cash),
                   assets=float(assets.total_assets),holdings=len(held),orders_submitted=0,
                   historical_close_reconstructed=False)
        write_json(STATE_DIR/'account_observations'/f"{observed:%Y%m%d_%H%M%S}.json",value)
        marker=STATE_DIR/f'nav_saved_{observed.date().isoformat()}.json'
        # Overnight logon can observe the account NOW, but never relabel it as yesterday's close.
        from paper_futu.cn_common import HISTORICAL_PROVIDER
        trading_dates=set((HISTORICAL_PROVIDER/'calendars/day_future.txt').read_text().splitlines())
        if (observed.date().isoformat() in trading_dates and
            15*60+35 <= observed.hour*60+observed.minute < 23*60 and not marker.exists()):
            write_json(marker,value)
        write_json(status_path,value);return value
    except Exception as exc:
        value=dict(stage='waiting_for_opend_login_or_account',time=china_now().isoformat(),
                   reason=str(exc),orders_submitted=0)
        write_json(status_path,value);return value
    finally:
        if trade is not None:trade.close()


def observe(now=None):
    import msvcrt
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    with (STATE_DIR/'account_observation.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:return dict(stage='account_observation_already_running',orders_submitted=0)
        return _observe(now)


def main():
    if sys.stdout is None:sys.stdout=(STATE_DIR/'account_recovery.stdout.log').open('a',encoding='utf-8')
    if sys.stderr is None:sys.stderr=(STATE_DIR/'account_recovery.stderr.log').open('a',encoding='utf-8')
    print(json.dumps(observe(),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
