"""Bounded Windows-logon recovery: public prices first; read-only account second."""
from __future__ import annotations
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
from paper_futu.cn_common import ROOT,STATE_DIR,china_now
from paper_futu.independent_cn_market import write_json


def recover(attempts=3,wait_seconds=60):
    outputs=[]
    for attempt in range(1,attempts+1):
        status=dict(stage='updating_public_data',attempt=attempt,pid=os.getpid(),time=china_now().isoformat(),orders_submitted=0)
        write_json(STATE_DIR/'session_recovery_status.json',status)
        try:
            result=subprocess.run([sys.executable,'-X','utf8','-m','paper_futu.update_cn_data','--catch-up'],
                cwd=ROOT,stdout=sys.stdout,stderr=sys.stderr,timeout=1800,creationflags=subprocess.CREATE_NO_WINDOW)
            code=result.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            code=124
            print('Public-data process did not complete:',str(exc),flush=True)
        if code==0:
            # Updater's lock-busy exit is deliberately zero, but is NOT proof of freshness.
            from paper_futu.update_cn_data import completed_date,calendar
            latest=completed_date(china_now(),calendar())
            if not (STATE_DIR/latest/'independent_data_ready.json').exists():code=75
        outputs.append(code)
        if code==0:break
        if attempt<attempts:time.sleep(wait_seconds)
    # Deliberately separate: failure of the public feed cannot suppress current NAV.
    sdk=ROOT/'paper_futu/.venv/Scripts/python.exe'
    try:
        observed=subprocess.run([str(sdk),'-X','utf8','-m','paper_futu.record_cn_account'],cwd=ROOT,
            stdout=sys.stdout,stderr=sys.stderr,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        account_code=observed.returncode
    except (OSError,subprocess.TimeoutExpired):account_code=124
    account_status=STATE_DIR/'account_recovery_status.json'
    account_stage=(json.loads(account_status.read_text(encoding='utf-8')).get('stage')
                   if account_status.exists() else 'account_status_unavailable')
    value=dict(stage='recovery_complete' if outputs[-1]==0 else 'public_data_error_account_checked',
        time=china_now().isoformat(),public_attempt_exit_codes=outputs,account_process_exit_code=account_code,
        account_stage=account_stage,orders_submitted=0,trade_executor_started=False)
    write_json(STATE_DIR/'session_recovery_status.json',value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts',type=int,default=3)
    parser.add_argument('--wait-seconds',type=int,default=60)
    args=parser.parse_args()
    if not 1<=args.attempts<=3 or not 1<=args.wait_seconds<=300:raise ValueError('Unbounded recovery settings')
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    if sys.stdout is None:sys.stdout=(STATE_DIR/'session_recovery.stdout.log').open('a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=(STATE_DIR/'session_recovery.stderr.log').open('a',encoding='utf-8',buffering=1)
    with (STATE_DIR/'session_recovery.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            print('Recovery already running; no duplicate');return
        value=recover(args.attempts,args.wait_seconds)
        print(json.dumps(value,ensure_ascii=False),flush=True)
        if value['stage']!='recovery_complete':raise SystemExit(1)


if __name__=='__main__':main()
