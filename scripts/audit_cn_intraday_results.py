"""Independent saved-ledger and sampled prediction audit, after research completes."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from paper_futu.research_cn_intraday import RUN, V2, SOURCE, CHECKPOINTS, calendars, verify, read, save, digest


def check_account(directory,bp):
    orders=pd.read_parquet(directory/'orders.parquet')
    daily=pd.read_parquet(directory/'daily.parquet')
    held=pd.read_parquet(directory/'holdings.parquet')
    orders['time']=pd.to_datetime(orders.time);orders['signal_time']=pd.to_datetime(orders.signal_time)
    daily['date']=pd.to_datetime(daily.date);held['date']=pd.to_datetime(held.date)
    if not orders.time.is_monotonic_increasing:raise ValueError('Orders out of sequence')
    cash=1e6;quantities={};entries={};exits={};fees={};notional={}
    for o in orders.itertuples():
        if o.signal_time>=o.time:raise ValueError('Noncausal fill')
        day=o.time.normalize();code=o.code
        expected=max(3.,o.value*.0003)+15.+max(.01,o.value*.0000487)+3*max(.01,o.value*.00002)+(0. if o.side=='BUY' else o.value*.001)+o.value*bp/10000
        np.testing.assert_allclose(o.fee,expected,atol=1e-8,rtol=0)
        np.testing.assert_allclose(o.quantity*o.price,o.value,rtol=1e-10,atol=1e-6)
        np.testing.assert_allclose(cash,o.cash_before,atol=1e-5,rtol=0)
        if o.side=='BUY':
            if quantities.get(code,0)>1e-7:raise ValueError('Retained stock topped up')
            if code in exits:
                last,reason=exits[code]
                if last==o.time or (reason=='stop' and last.normalize()>=day):raise ValueError('Cooldown violation')
            minimum,unit=(200,1) if code.startswith('SH688') else (100,100)
            if o.raw_quantity<minimum-1e-6 or abs(o.raw_quantity/unit-round(o.raw_quantity/unit))>1e-5:raise ValueError('Invalid buy lot')
            entries[code]=day;quantities[code]=o.quantity;cash-=o.value+o.fee
        else:
            if entries[code]>=day:raise ValueError('T+1 violated')
            quantities[code]-=o.quantity;cash+=o.value-o.fee
            if quantities[code]<-1e-6:raise ValueError('Oversell')
            if quantities[code]<=1e-7:exits[code]=(o.time,o.reason)
        if sum(q>1e-7 for q in quantities.values())>10:raise ValueError('Position cap exceeded')
        if cash<-1e-6:raise ValueError('Negative cash')
        np.testing.assert_allclose(cash,o.cash_after,atol=1e-5,rtol=0)
        fees[day]=fees.get(day,0)+o.fee;notional[day]=notional.get(day,0)+o.value
    # Actual source bars must permit every fill. No carried zero-trade quote.
    for code,g in orders.groupby('code'):
        source=pd.read_parquet(V2/'bars_5m'/f'{code}.parquet')
        source=source.reindex(pd.DatetimeIndex(g.time)+pd.Timedelta(minutes=5))
        if source.open.isna().any() or (source.volume<=0).any():raise ValueError('Fill without trades')
        np.testing.assert_allclose(g.raw_price,source.open.to_numpy(),rtol=2e-7,atol=1e-5)
        if (g.raw_quantity.to_numpy()>source.volume.to_numpy()*.05+1e-5).any():raise ValueError('Participation cap exceeded')
    missing_trade_days=[]
    for code,g in held.groupby('code'):
        source=pd.read_parquet(V2/'daily'/f'{code}.parquet',columns=['volume'])
        observed=source.volume.reindex(pd.DatetimeIndex(g.date))
        bad=observed.isna()|observed.le(0)
        if bad.any():missing_trade_days.extend(dict(code=code,date=str(d.date())) for d in observed.index[bad])
    groups=dict(tuple(held.groupby('date')));order_groups=dict(tuple(orders.groupby(orders.time.dt.normalize())))
    prior_nav=1e6;cash=1e6;positions={}
    for d in daily.itertuples():
        for o in order_groups.get(d.date,pd.DataFrame()).itertuples():
            sign=1 if o.side=='BUY' else -1
            cash-=sign*o.value+o.fee;positions[o.code]=positions.get(o.code,0)+sign*o.quantity
        h=groups.get(d.date,pd.DataFrame(columns=['value','code','quantity']))
        actual={c for c,q in positions.items() if q>1e-7}
        if actual!=set(h.code):raise ValueError('Daily holdings mismatch')
        for row in h.itertuples():np.testing.assert_allclose(row.quantity,positions[row.code],atol=1e-6,rtol=0)
        nav=cash+h.value.sum()
        np.testing.assert_allclose([d.nav,d.cash,d.net_return,d.fees,d.turnover],
            [nav,cash,nav/prior_nav-1,fees.get(d.date,0),notional.get(d.date,0)/prior_nav],atol=1e-5,rtol=0)
        prior_nav=nav
    return dict(status='passed',fills=len(orders),days=len(daily),final_nav=prior_nav,
        checks=['fee components','cash','holdings cap','no top-ups','legal buy lots','T+1',
                'stop cooldown','no same-time reentry','signal before fill','raw source price',
                'nonzero execution volume','5% participation','daily NAV/returns/turnover'],
        held_days_without_recorded_trades=missing_trade_days,
        valuation_limit='No-trade holdings retain the last usable mark when official close missing; historical delisting settlements not reconstructed')


def prediction_diagnostics():
    labels=pd.concat([pd.read_parquet(p,columns=['instrument','label','label_end']) for p in (RUN/'train_parts').glob('*.parquet')])
    labels=labels.loc[labels.index>='2022-01-04'].dropna(subset=['label'])
    labels=labels.reset_index().set_index(['datetime','instrument'])
    records=[]
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        p=pd.read_parquet(path)
        p=p[np.isin(p.index.hour*60+p.index.minute,CHECKPOINTS)].reset_index().set_index(['datetime','instrument'])
        x=p.join(labels,how='inner').dropna(subset=['label'])
        if x.empty:continue
        if (x.label_end>pd.Timestamp('2025-12-31 23:59:59')).any():raise ValueError('Unmature label in evaluation')
        for model in ('daily15','minute29'):
            r=x[[model,'label']].groupby(level='datetime').rank(method='average')
            z=r-r.groupby(level='datetime').transform('mean')
            cross=(z[model]*z.label).groupby(level='datetime').sum()
            denominator=np.sqrt((z[model]**2).groupby(level='datetime').sum()*(z.label**2).groupby(level='datetime').sum())
            ic=(cross/denominator.replace(0,np.nan)).rename('rank_ic').reset_index();ic['variant']=model
            records.append(ic)
    results=pd.concat(records,ignore_index=True);results.to_csv(RUN/'sampled_rank_ic.csv',index=False)
    results['year']=results.datetime.dt.year
    summary=results.groupby(['variant','year']).rank_ic.agg(['mean','std','count']).reset_index()
    summary.to_csv(RUN/'sampled_rank_ic_yearly.csv',index=False)
    return summary.to_dict('records')


def main():
    p=verify();completion=read(RUN/'completion.json')
    for path,h in completion['artifacts'].items():
        if digest(path)!=h:raise ValueError('Report artifact changed')
    model_records=read(RUN/'train_complete.json')['models']
    if len(model_records)!=32:raise ValueError('Incomplete quarterly model set')
    expected={(m,y,q) for m in ('daily15','minute29') for y in range(2022,2026) for q in range(1,5)}
    if {(r['variant'],r['year'],r['quarter']) for r in model_records}!=expected:
        raise ValueError('Quarterly model identity mismatch')
    for r in model_records:
        cutoff=pd.Timestamp(r['cutoff'])
        if pd.Timestamp(r['label_end_max'])>cutoff or pd.Timestamp(r['signal_max'])>cutoff:
            raise ValueError('Training boundary crossed')
        path=RUN/'models'/r['variant']/f'{r["year"]}Q{r["quarter"]}'/'model.txt'
        if digest(path)!=r['model_sha256'] or r['reload_error']>1e-10:raise ValueError('Model audit failed')
    feature_checks=list((RUN/'feature_checks').glob('*.json'))
    if len(feature_checks)!=657:raise ValueError('Incomplete stock feature set')
    for f in feature_checks:
        for path,h in read(f)['artifacts'].items():
            if digest(path)!=h:raise ValueError('Input feature/market cache changed')
    sealed=read(SOURCE.parent/'audit_complete.json')['artifacts']
    for f in [SOURCE/'features.parquet',SOURCE/'benchmark.parquet']:
        if str(f) not in sealed or digest(f)!=sealed[str(f)]:raise ValueError('Sealed daily source changed')
    provider_files=0
    for path,h in sealed.items():
        if str(SOURCE/'provider') in path:
            if digest(path)!=h:raise ValueError('Sealed provider input changed')
            provider_files+=1
    checks={}
    for model in ('daily15','minute29'):
        for bp in (0,10):
            name=f'{model}_{bp}bp';checks[name]=check_account(RUN/'accounts'/name,bp)
            print('AUDITED '+name,flush=True)
    ic=prediction_diagnostics()
    verify()
    save(RUN/'independent_audit.json',dict(status='passed',accounts=checks,sampled_rank_ic=ic,
        implementation_sha256=digest(__file__),report_completion_sha256=digest(RUN/'completion.json'),
        quarterly_models_audited=32,feature_and_market_caches_verified=True,sealed_daily_sources_verified=True,
        sealed_provider_files_verified=provider_files,
        production_protected_unchanged=True,scope='Research ledgers and matured sampled labels; not proof of executable live alpha'))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
