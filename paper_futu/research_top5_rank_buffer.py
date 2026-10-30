"""Top5 ranking buffer / staged exits, strictly pre-2026 offline research.

python -m paper_futu.research_top5_rank_buffer --as-of 2025-12-31 --stage all
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import time
import traceback

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'

from paper_futu.top5_buffer_data import ROOT, START, END, read, save, digest, observe_memory

OUT = ROOT/'reports/experiments/top5_rank_buffer_asof_20251231'


def scientific_hashes():
    paths = [ROOT/'paper_futu'/n for n in ('research_top5_rank_buffer.py', 'top5_buffer_data.py',
                                          'top5_buffer_engine.py', 'top5_buffer_report.py')]
    paths += [ROOT/'src'/n for n in ('factor_library.py','build_enhanced_dataset.py')]
    paths += [ROOT/'src/runtime_budget.py', ROOT/'paper_futu/final_ablation_core.py',
              ROOT/'paper_futu/benchmark_attribution.py']
    return {str(p): digest(p) for p in paths}


def protected():
    paths = [ROOT/'configs/futu_cn_paper.json', ROOT/'configs/futu_cn_forward_protocol.json']
    paths += list((ROOT/'models/paper_futu/s6_qreg_20260930').glob('*'))
    return {str(p): digest(p) for p in paths if p.is_file()}


def verify_files(files):
    for p, h in files.items():
        if digest(p) != h: raise ValueError(f'Artifact changed: {p}')


@contextmanager
def lock():
    """OS-held byte lock; released even after interruption. Never duplicate runners."""
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
        try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e: raise RuntimeError('Another Top5 experiment is running') from e
        try: yield
        finally:
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def protocol():
    p = read(OUT/'protocol.json')
    if p['implementation'] != scientific_hashes():
        raise ValueError('Implementation changed; audit cache compatibility before resuming')
    verify_files(p['protected'])
    return p


def audit():
    from paper_futu.top5_buffer_data import prepare
    existing = OUT/'audit_complete.json'
    if existing.exists():
        protocol(); verify_files(read(existing)['artifacts']); return
    p = dict(start=START, end=END, initial_cash=1e6, cases=list('ABCDE'), friction=[0.,.001,.002],
             max_stocks=5, risk_degree=.95, new_entry_weight=.19, no_rebalance_retained=True,
             rank_exit_above=15, strong_entry_top=2, cooldown_trading_days=3,
             loss_budget='min(entry NAV * 2%, initial fee-inclusive cost * 10%)',
             trailing_breakeven='remaining fee-inclusive pro-rata entry cost plus expected exit fee',
             first_profit=.24, second_profit=.24, trailing_drawdown=.10,
             signal='completed close', execution='next ideal open', rf_annual=.01,
             max_workers=1, threads=4, startup_gib=None, task_gib=None,
             memory_gate_enabled=False, memory_authorization='2026-10-27 user: directly continue, no memory protection',
             implementation=scientific_hashes(), protected=protected(),
             limitations=['Daily ideal opening fill, not verified 09:31 fills; no market impact or queue model.',
                          'Uniform +/-9.5% directional filter; not complete historical ST/board limit reconstruction.',
                          'Adjusted-price inventory includes corporate actions synthetically; not a raw-share dividend cash ledger.',
                          'Historical finance revisions lack a complete vintage archive.',
                          '2022-2025 already studied; previous 2026 observations cannot be unobserved.'])
    path = OUT/'protocol.json'
    if path.exists() and read(path) != p:
        # An initial memory-gate refusal has no scientific outputs to reuse.
        # Keep the old binding visibly; never silently rebind any completed work.
        has_outputs = any(f.is_file() for folder in ('models','accounts') for f in (OUT/folder).rglob('*'))
        if has_outputs: raise ValueError('Existing protocol differs; outputs require cache compatibility review')
        save(OUT/'pre_execution_protocols'/(digest(path)+'.json'),
             dict(reason='Unaccepted audit data will be rebuilt; no fitted model or account caches exist or are reused',previous=read(path)))
    save(path, p)
    prepare(OUT)
    verify_files(p['protected'])
    files = list((OUT/'data').rglob('*'))+[OUT/'models_audit.json',OUT/'data_audit.json',OUT/'daily_ic.csv',
                                          OUT/'provider_prefix_manifest.json']
    save(existing, dict(status='passed', artifacts={str(f):digest(f) for f in files if f.is_file()}))


def replay(case, friction, market, predictions, dates, previous_dates):
    import pandas as pd
    from paper_futu.top5_buffer_engine import Account
    require_memory = observe_memory
    name = f'{case}_{round(friction*10000):02d}bp'
    directory = OUT/'accounts'/name; complete = directory/'complete.json'
    if complete.exists():
        c = read(complete)
        if c['protocol_sha256'] != digest(OUT/'protocol.json'): raise ValueError('Account protocol changed')
        verify_files(c['artifacts']); return
    require_memory(1.5, f'Top5 account {name} startup')
    account = Account(case, friction)
    for i, day in enumerate(dates):
        if i % 50 == 0:
            require_memory(1., f'Top5 account {name} safety floor')
            save(OUT/'progress.json', dict(account=name, day=str(day.date()), completed_days=i, total_days=len(dates)))
        prior = previous_dates[day]
        part = predictions.get(day)
        if part is None: raise ValueError(f'Missing prediction {day}')
        if not (part.datetime == prior).all(): raise ValueError('Prediction signal/execution off by one session')
        quotes = market.get(day, {})
        previous = {code: row['close'] for code, row in market.get(prior, {}).items()}
        account.step(str(day.date()), i, dict(zip(part.instrument, part.prediction)), quotes, previous)
    checks = account.reconcile()
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {'daily':account.daily,'orders':account.orders,'events':account.events,
               'episodes':account.episodes,'open_episodes':account.open_episodes(str(dates[-1].date()),len(dates)-1),
               'holdings':account.holdings}
    for n, records in outputs.items():
        pd.DataFrame(records).to_parquet(directory/(n+'.parquet'), index=False)
    checks.update(last_day=END, no_forced_terminal_sale=True,
                  episodes=len(account.episodes), open_episodes=len(account.positions), orders=len(account.orders))
    save(directory/'reconciliation.json',checks)
    artifacts = {str(f):digest(f) for f in directory.iterdir() if f.is_file() and f.name!='complete.json'}
    save(complete,dict(status='complete',case=case,friction=friction,
                       protocol_sha256=digest(OUT/'protocol.json'),artifacts=artifacts))
    print(f'COMPLETE {name} NAV={account.daily[-1]["nav"]:.2f} episodes={len(account.episodes)}',flush=True)


def backtest():
    require_memory = observe_memory
    require_memory(1.5,'Top5 sequential accounts startup')
    import pandas as pd
    protocol(); verify_files(read(OUT/'audit_complete.json')['artifacts'])
    panel = pd.read_parquet(OUT/'data/market.parquet').reset_index()
    # Today's total volume is deliberately NOT supplied to the opening executor.
    market = {date: g.set_index('instrument')[['open','close','factor']].to_dict('index')
              for date,g in panel.groupby('datetime',sort=True)}
    del panel
    pred = pd.read_parquet(OUT/'data/predictions.parquet').reset_index()
    predictions = {date:g for date,g in pred.groupby('execution_date',sort=True)}
    calendar = pd.DatetimeIndex(pd.to_datetime((OUT/'data/provider/calendars/day.txt').read_text().splitlines()))
    dates = calendar[(calendar >= START) & (calendar <= END)]
    prior = {d:calendar[calendar.get_loc(d)-1] for d in dates}
    for case in 'ABCDE':
        for friction in (0.,.001,.002):
            replay(case,friction,market,predictions,dates,prior)
    verify_files(protocol()['protected'])


def main():
    import psutil
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--as-of',default=END,choices=[END])
    parser.add_argument('--stage',default='all',choices=['audit','backtest','report','all'])
    args = parser.parse_args()
    with lock():
        status = dict(pid=os.getpid(),created=psutil.Process().create_time(),stage=args.stage,
                      started=time.time(),status='running',available_gib=psutil.virtual_memory().available/2**30)
        save(OUT/'status.json',status)
        try:
            if args.stage in ('audit','all'): audit()
            if args.stage in ('backtest','all'): backtest()
            if args.stage in ('report','all'):
                from paper_futu.top5_buffer_report import report
                report(OUT)
            status.update(status='complete',elapsed=time.time()-status['started'])
        except BaseException as error:
            status.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            raise
        finally: save(OUT/'status.json',status)


if __name__ == '__main__': main()
