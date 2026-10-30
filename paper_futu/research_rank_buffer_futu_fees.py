"""Independent replay with provisionally published Futu CN paper transaction fees.

Private module instances inject fees without mutating either sealed engine module.
Custody fees are excluded pending clarification of the published share-count basis.
"""
from __future__ import annotations
import importlib.util
import math
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
from paper_futu.top5_buffer_data import ROOT, START, END, read, save, digest

SOURCE=ROOT/'reports/experiments/top5_rank_buffer_asof_20251231'
BASE=ROOT/'reports/experiments/rank_buffer_full_exit_asof_20251231'
OUT=ROOT/'reports/experiments/rank_buffer_futu_fees_asof_20251231'
VARIANTS=('staged_10_5','full_10_5','full_8_4')

def fee_components(value, buying):
    if value<=0:return dict(commission=0.,platform=0.,handling=0.,regulatory=0.,transfer_cn=0.,transfer_hk=0.,stamp=0.)
    return dict(commission=max(3.,value*.0003),platform=15.,handling=max(.01,value*.0000487),
                regulatory=max(.01,value*.00002),transfer_cn=max(.01,value*.00002),
                transfer_hk=max(.01,value*.00002),stamp=0. if buying else value*.001)

def fee(value,buying,friction=0.):
    return sum(fee_components(value,buying).values())+max(value,0)*friction

def buy_quantity(code,money,adjusted_price,factor,friction=0.):
    if not all(math.isfinite(x) and x>0 for x in (money,adjusted_price,factor)):return 0.
    minimum,step=(200,1) if code.startswith('SH688') else (100,100)
    raw_price=adjusted_price/factor
    lo,hi=0,int(money/raw_price)//step
    while lo<hi:
        mid=(lo+hi+1)//2;value=mid*step*raw_price
        if value+fee(value,True,friction)<=money:lo=mid
        else:hi=mid-1
    return lo*step/factor if lo*step>=minimum else 0.

def liquidation_breakeven(quantity,basis,friction):
    if quantity<=0 or not 0<=friction<.9:raise ValueError('Invalid liquidation inputs')
    lo=0.;hi=max(100.,basis*2+100.)
    while hi-fee(hi,False,friction)<basis:hi*=2
    for _ in range(70):
        mid=(lo+hi)/2
        if mid-fee(mid,False,friction)>=basis:hi=mid
        else:lo=mid
    return hi/quantity

def engine(staged):
    filename='rank_buffer_10_20_engine' if staged else 'rank_buffer_full_exit_engine'
    name=f'_futu_fee_private_{filename}'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'paper_futu'/f'{filename}.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        module.fee=fee;module.buy_quantity=buy_quantity;module.liquidation_breakeven=liquidation_breakeven
    return sys.modules[name]

def verify(files):
    for path,h in files.items():
        if digest(path)!=h:raise ValueError(f'Changed source: {path}')

def ledger_audit(daily,orders,holdings,closed,opened,k):
    import numpy as np
    cash=1e6;qty={};previous=1e6
    og=dict(tuple(orders.groupby('date')));hg=dict(tuple(holdings.groupby('date')))
    for row in daily.itertuples():
        fees=0.;g=og.get(row.date)
        if g is not None:
            assert not set(g.loc[g.direction.eq('BUY'),'code']) & set(g.loc[g.direction.eq('SELL'),'code'])
            for o in g.itertuples():
                # Independent explicit arithmetic, not the engine fee function.
                expected=max(3,o.notional*.0003)+15+max(.01,o.notional*.0000487)+3*max(.01,o.notional*.00002)+(0 if o.direction=='BUY' else o.notional*.001)
                np.testing.assert_allclose(o.fee,expected,atol=1e-8)
                np.testing.assert_allclose(cash,o.cash_before,atol=1e-6,rtol=0)
                assert str(o.trigger_day)<str(row.date.date())
                sign=1 if o.direction=='BUY' else -1
                cash-=sign*o.notional+o.fee;qty[o.code]=qty.get(o.code,0)+sign*o.quantity
                assert qty[o.code]>=-1e-6
                np.testing.assert_allclose(cash,o.cash_after,atol=1e-6,rtol=0);fees+=o.fee
        h=hg.get(row.date);value=0 if h is None else h.value.sum()
        assert {c for c,q in qty.items() if q>1e-6}==(set() if h is None else set(h.code))
        if h is not None:
            assert len(h)<=k
            for p in h.itertuples():np.testing.assert_allclose(qty[p.code],p.quantity,atol=1e-6,rtol=0)
        np.testing.assert_allclose([cash,cash+value,fees,(cash+value)/previous-1],
                                   [row.cash,row.nav,row.fees,row.net_return],atol=1e-6,rtol=0)
        previous=row.nav
    np.testing.assert_allclose(previous,1e6+closed.pnl.sum()+opened.pnl.sum(),atol=1e-5,rtol=0)
    return dict(status='passed',days=len(daily),orders=len(orders))

def run():
    import pandas as pd
    from paper_futu.top5_buffer_report import metrics,trade_metrics
    completion=read(BASE/'completion.json');assert completion['status']=='complete'
    assert completion['manifest_sha256']==digest(BASE/'report_manifest.json')
    p=read(BASE/'protocol.json')
    bindings={**p['sources'],**p['implementation'],**read(BASE/'report_manifest.json')['artifacts'],
              str(BASE/'report_manifest.json'):digest(BASE/'report_manifest.json'),__file__:digest(__file__)}
    verify(bindings);verify(p['protected'])
    protocol=dict(start=START,end=END,fee_source='https://support.futunn.com/topic692',
        fee_checked_date='2026-10-28',rounding='Full precision per component; broker rounding unverified',
        scope='Six base-cost accounts only; no extra slippage; constant current published fee assumptions throughout history',
        custody_fee='Excluded: published share-count basis ambiguous and account charging unverified',
        actual_fee_evidence='2026-10-20: 63832 notional, 39.812 cash fee; published formula 41.0881384; not exact replication',
        no_training=True,source=bindings,protected=p['protected'])
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    panel=pd.read_parquet(SOURCE/'data/market.parquet').reset_index();assert panel.datetime.max()<=pd.Timestamp(END)
    market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')};del panel
    pred=pd.read_parquet(SOURCE/'data/predictions.parquet').reset_index();assert pred.execution_date.max()<=pd.Timestamp(END)
    predictions=dict(tuple(pred.groupby('execution_date')))
    calendar=pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    assert calendar.max()==pd.Timestamp(END)
    days=calendar[(calendar>=START)&(calendar<=END)]
    benchmark=pd.read_parquet(SOURCE/'data/benchmark.parquet').benchmark_return
    rows=[];years=[];trades=[];checks={};curves={}
    for k in (10,20):
        for v in VARIANTS:
            name=f'top{k}_{v}';d=OUT/'accounts'/name;d.mkdir(parents=True,exist_ok=True)
            if not (d/'complete.json').exists():
                mod=engine(v=='staged_10_5')
                kwargs={} if v=='staged_10_5' else dict(profit=.10 if v=='full_10_5' else .08,stop=.05 if v=='full_10_5' else .04)
                a=mod.Account(k,**kwargs)
                for step,day in enumerate(days):
                    prev=calendar[calendar.get_loc(day)-1];r=predictions[day];assert r.datetime.eq(prev).all()
                    a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],
                           {c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
                a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
                if v!='staged_10_5':assert all(not e['half_mode'] for e in a.episodes+opened)
                records=dict(daily=a.daily,orders=a.orders,holdings=a.holdings,episodes=a.episodes,open_episodes=opened,events=a.events)
                for n,r in records.items():pd.DataFrame(r).to_parquet(d/f'{n}.parquet',index=False)
                save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
            done=read(d/'complete.json');assert done['protocol_sha256']==digest(OUT/'protocol.json');verify(done['artifacts'])
            daily,orders,holdings,closed,opened=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes')]
            for f in (daily,orders,holdings):f['date']=pd.to_datetime(f.date)
            assert pd.DatetimeIndex(daily.date).equals(benchmark.index)
            checks[name]=ledger_audit(daily,orders,holdings,closed,opened,k)
            info=dict(top_k=k,variant=v);rows.append(dict(**info,**metrics(daily,benchmark)))
            for y,g in daily.groupby(daily.date.dt.year):years.append(dict(**info,year=y,**metrics(g,benchmark)))
            trades.append(dict(**info,**trade_metrics(closed,opened)))
            curves[name]=daily
            print(f'COMPLETE {name}: NAV={daily.nav.iloc[-1]:.2f}',flush=True)
    table=pd.DataFrame(rows);old=pd.read_csv(BASE/'comparison.csv').query('friction_bp==0')
    table=table.merge(old[['top_k','variant','cagr','fees','total_return','max_drawdown']],on=['top_k','variant'],suffixes=('','_old'))
    for n,t in dict(comparison=table,yearly=pd.DataFrame(years),trades=pd.DataFrame(trades)).items():t.to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for name,daily in curves.items():axs[0 if 'top10_' in name else 1].plot(daily.date,daily.nav/1e6,label=name)
    for ax in axs:ax.legend();ax.grid(alpha=.2);ax.set_ylabel('Net wealth')
    fig.tight_layout();fig.savefig(OUT/'nav.png',dpi=150);plt.close(fig)
    display=table[['top_k','variant','total_return','cagr','cagr_old','max_drawdown','fees']].copy()
    for c in ('total_return','cagr','cagr_old','max_drawdown'):display[c]=display[c].map(lambda x:f'{x:.2%}')
    display.fees=display.fees.map(lambda x:f'{x:,.2f}')
    page='<html lang="zh"><meta charset="utf-8"><title>富途模拟盘公开交易费用回放</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto}td,th{padding:9px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>Top10/20：富途模拟盘公开交易费用回放</h1>'
    page+='<p>2022-01-04至2025-12-31；初始100万元。沿用原预测、信号、持仓和执行规则，重新回放现金账户；不重训，不修改模拟盘。旧结果也已扣旧费用。</p>'
    page+='<p>公开交易费：佣金0.03%最低3元，平台15元，经手0.00487%最低0.01元，证管及两项过户费各0.002%最低0.01元，卖出印花税0.1%。每项保留计算精度；历史统一使用该费表，不是历史费率复原。未另加滑点。<a href="https://support.futunn.com/topic692">官方来源</a>。</p>'
    page+='<p><b>暂定口径：</b>一笔实际买单63832元扣费39.812元，而官网公式41.088元，两者存在差异；模拟账户不支持费用明细查询。证券组合费的计费基数未核实，本表未扣该项，不能称为完全复刻账单的全部费用。实际盘口成交亦可能不同于理想开盘成交。</p>'
    page+=display.to_html(index=False)+'<img src="nav.png"><h2>逐年结果（小数表示收益率）</h2>'+pd.DataFrame(years)[['top_k','variant','year','total_return','max_drawdown','fees']].round(5).to_html(index=False)
    page+='<p>止盈止损由收盘价触发、下一交易日开盘执行。2022—2025已经反复研究，不是独立测试；未使用2026交易数据。</p></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    verify(bindings);verify(p['protected'])
    save(OUT/'verification.json',dict(status='passed',checks=checks,sources_unchanged=True,live_unchanged=True))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('report_manifest.json','completion.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',accounts=6,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(display.to_string(index=False),flush=True)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
