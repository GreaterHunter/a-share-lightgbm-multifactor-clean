"""Bounded 2022-2025 classic 12-1 momentum versus frozen quarterly S6."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import html
import importlib
import os
from pathlib import Path
import time
from unittest.mock import patch

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'

import numpy as np
import pandas as pd
from paper_futu.top5_buffer_data import ROOT, START, END, read, save, digest, init_qlib

SOURCE = ROOT/'reports/experiments/top5_rank_buffer_asof_20251231'
OUT = ROOT/'reports/experiments/momentum_vs_s6_asof_20251231'
CASES = {
    'momentum_monthly': dict(signal='momentum', schedule='monthly', top_k=20,
                             n_drop=20, top_tier_size=20, top_tier_weight=.5),
    's6_main': dict(signal='s6', schedule='five_days', top_k=20,
                    n_drop=6, top_tier_size=6, top_tier_weight=.4),
    's6_monthly_equal': dict(signal='s6', schedule='monthly', top_k=20,
                             n_drop=20, top_tier_size=20, top_tier_weight=.5),
}
LABELS = {'momentum_monthly':'12—1月动量／Top20等权／月调仓',
          's6_main':'S6主策略／前6名40%／5日调仓',
          's6_monthly_equal':'S6信号／Top20等权／月调仓'}


def momentum_score(close):
    """Rows are the full trading calendar; no compression across missing prices."""
    return (close.shift(21)/close.shift(252)-1).replace([np.inf,-np.inf],np.nan)


def schedule(calendar, kind):
    dates=calendar[(calendar>=START)&(calendar<=END)]
    if kind=='monthly':
        trades=dates[~dates.to_period('M').duplicated()]
    elif kind=='five_days': trades=dates[::5]
    else: raise ValueError(kind)
    signals=calendar[calendar.get_indexer(trades)-1]
    assert (signals<trades).all()
    return dates,trades,signals


def verify(files):
    for p,h in files.items():
        if digest(p)!=h: raise ValueError(f'Artifact changed: {p}')


def code_hashes():
    return {str(ROOT/p):digest(ROOT/p) for p in (
        'paper_futu/research_momentum_s6.py','paper_futu/final_ablation_account.py',
        'paper_futu/final_ablation_core.py','src/portfolio_strategy.py',
        'paper_futu/top5_buffer_report.py','paper_futu/top5_buffer_data.py')}


def bounded_step_time(manager,trade_step=None,shift=0):
    """Qlib terminal daily interval, without reading/adding a future session."""
    if trade_step is None:trade_step=manager.get_trade_step()
    i=manager.start_index+trade_step-shift
    if i<0 or i>=len(manager._calendar):raise IndexError('Outside bounded execution calendar')
    start=manager._calendar[i]
    if i==len(manager._calendar)-1:
        assert start==pd.Timestamp(END)
        return start,start+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
    return start,manager._calendar[i+1]-pd.Timedelta(seconds=1)


@contextmanager
def lock():
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try: yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def audit():
    c=read(SOURCE/'completion.json')
    assert c['status']=='complete' and c['report_manifest_sha256']==digest(SOURCE/'report_manifest.json')
    verify(read(SOURCE/'audit_complete.json')['artifacts'])
    source_protocol=read(SOURCE/'protocol.json')
    verify(source_protocol['implementation']);verify(source_protocol['protected'])
    models=read(SOURCE/'models_audit.json')['periods']
    assert len(models)==16
    for m in models:
        assert m['label_end_max']<=m['cutoff']<m['start'] and m['end']<=END and m['reload_equal']
        assert digest(m['model_path'])==m['model_sha256']
    paths=['completion.json','audit_complete.json','models_audit.json','protocol.json',
           'data/predictions.parquet','data/benchmark.parquet']
    bindings={str(SOURCE/p):digest(SOURCE/p) for p in paths}
    bindings.update(read(SOURCE/'audit_complete.json')['artifacts'])
    bindings.update({m['model_path']:m['model_sha256'] for m in models})
    protocol=dict(start=START,end=END,initial_cash=1e6,cases=CASES,
        momentum='adjusted_close[t-21] / adjusted_close[t-252] - 1; full trading calendar',
        universe='historical CSI300 membership at signal close',risk_degree=.95,
        buy_cost=.0005,sell_cost=.0015,min_fee=5,execution='previous close sizing; next ideal open fills',
        monthly='first trading day of each month using preceding month final close',
        s6='quarterly QREG raw15 H10; 16 audited pre-2026 models reused; no refit',
        risk_free_annual=.01,annualization=252,threads=4,workers=1,
        protected=source_protocol['protected'],sources=bindings,implementation=code_hashes(),
        limitations=source_protocol['limitations']+[
            'Classic momentum excludes missing or nonpositive lookback endpoint prices; no imputation.',
            'Same historical membership, but missing momentum history can reduce eligible names versus S6.',
            'Main-strategy comparison changes both signals and portfolio rules; monthly equal S6 is the paired control.',
            'Reuses legacy audited S6 sizing and rounding, not newer live integer tolerance logic.',
            '95% is the prior-close target; gaps and legacy affordability can change realized cash fraction.',
            'One fixed calendar schedule, no start-date or parameter search.'])
    if (OUT/'protocol.json').exists():
        old=read(OUT/'protocol.json')
        if old!=protocol:
            assert not list((OUT/'accounts').glob('*/complete.json')),'Cannot amend completed accounts'
            assert {k:v for k,v in old.items() if k!='implementation'}=={k:v for k,v in protocol.items() if k!='implementation'}
            save(OUT/'initial_failed_protocol.json',old)
            save(OUT/'implementation_amendment.json',dict(reason='Qlib last-day interval lacked next calendar date; bound at final day 23:59:59 without future data',
                 completed_accounts=0,old=old['implementation'],new=protocol['implementation']))
            save(OUT/'protocol.json',protocol)
    else: save(OUT/'protocol.json',protocol)
    calendar=pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    assert calendar.max()==pd.Timestamp(END)
    pred=pd.read_parquet(SOURCE/'data/predictions.parquet')
    assert not pred.index.has_duplicates
    d=pred.index.get_level_values('datetime');pos=calendar.get_indexer(d)
    assert (pos>=0).all() and (pos+1<len(calendar)).all()
    assert np.array_equal(calendar.to_numpy()[pos+1],pred.execution_date.to_numpy())
    assert (pred.model_cutoff<=d).all() and np.isfinite(pred.prediction).all()
    assert pred.execution_date.max()<=pd.Timestamp(END)
    save(OUT/'audit.json',dict(status='passed',models=16,prediction_rows=len(pred),no_training=True))
    return calendar,pred[['prediction']]


def build_momentum(calendar):
    """Read only the previously physically truncated provider, including warmup."""
    provider=SOURCE/'data/provider'
    members=pd.read_csv(provider/'instruments/csi300.txt',sep='\t',header=None,
                        names=['instrument','start','end'],parse_dates=['start','end'])
    codes=sorted(members.instrument.unique())
    close=pd.DataFrame(np.nan,index=calendar,columns=codes)
    eligible=pd.DataFrame(False,index=calendar,columns=codes)
    for code in codes:
        p=provider/'features'/code.lower()/'close.day.bin'
        if p.exists():
            values=np.fromfile(p,dtype='<f4');start=int(values[0]);values=values[1:]
            assert start+len(values)<=len(calendar)
            close.loc[calendar[start:start+len(values)],code]=values
    for m in members.itertuples():
        eligible.loc[(calendar>=m.start)&(calendar<=m.end),m.instrument]=True
    close=close.where(close>0)
    score=momentum_score(close).where(eligible)
    # Bounded-future perturbation regression: later prices cannot alter earlier signals.
    cut=pd.Timestamp('2024-12-31');perturbed=close.copy()
    perturbed.loc[perturbed.index>cut]*=13
    pd.testing.assert_frame_equal(momentum_score(perturbed).loc[:cut],momentum_score(close).loc[:cut])
    earliest=calendar[calendar.get_loc(pd.Timestamp(START))-1]
    score=score.loc[earliest:END]
    coverage=pd.DataFrame(dict(members=eligible.loc[score.index].sum(axis=1),
                              momentum_available=score.notna().sum(axis=1)))
    coverage.to_csv(OUT/'momentum_coverage.csv',encoding='utf-8-sig')
    frame=score.rename_axis(index='datetime',columns='instrument').stack(future_stack=True).dropna().rename('prediction').to_frame()
    frame=frame.reorder_levels(['instrument','datetime']).sort_index()
    frame.to_parquet(OUT/'momentum_signals.parquet')
    save(OUT/'signal_audit.json',dict(lookback=252,skip=21,source_end=END,
        future_perturbation_invariance=True,rows=len(frame),
        minimum_coverage=int(coverage.momentum_available.min()),
        maximum_coverage=int(coverage.momentum_available.max())))
    return frame,codes


def replay(name,config,prediction,calendar,codes,benchmark):
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.evaluate import backtest_daily
    from qlib.backtest.utils import TradeCalendarManager
    from paper_futu.final_ablation_account import AuditedExchange,AuditedStrategy
    directory=OUT/'accounts'/name;directory.mkdir(parents=True,exist_ok=True)
    done=directory/'complete.json'
    if done.exists():
        c=read(done);assert c['protocol_sha256']==digest(OUT/'protocol.json')
        verify(c['artifacts']);return
    days,trades,signals=schedule(calendar,config['schedule']);prior=calendar[calendar.get_loc(days[0])-1]
    signal=prediction.loc[prediction.index.get_level_values('datetime').isin(signals)]
    assert signal.index.get_level_values('datetime').nunique()==len(signals)
    signal.to_parquet(directory/'signals.parquet')
    pd.DataFrame(dict(signal_date=signals,execution_date=trades)).to_csv(directory/'schedule.csv',index=False)
    market=AuditedExchange(freq='day',start_time=prior,end_time=END,codes=codes,
        deal_price='open',open_cost=.0005,close_cost=.0015,min_cost=5,trade_unit=100,
        limit_threshold=('($open / Ref($close, 1) - 1) >= 0.095','($open / Ref($close, 1) - 1) <= -0.095'))
    market.reset_audit(0.,20)
    strategy=AuditedStrategy(signal=signal,risk_degree=.95,only_tradable=True,
                            **{k:v for k,v in config.items() if k not in ('signal','schedule')})
    module=importlib.import_module('qlib.backtest.backtest')
    with patch.object(module,'tqdm',partial(module.tqdm,disable=True)), patch.object(TradeCalendarManager,'get_step_time',bounded_step_time):
        raw,positions=backtest_daily(start_time=prior,end_time=END,strategy=strategy,
            executor=SimulatorExecutor(time_per_step='day',generate_portfolio_metrics=True,
                verbose=False,trade_type=SimulatorExecutor.TT_SERIAL),account=1e6,
            benchmark=benchmark.reindex(pd.DatetimeIndex([prior]).append(days),fill_value=0.),
            exchange_kwargs={'exchange':market})
    r=raw.reindex(days);assert r[['account','cash','return','cost']].notna().all().all()
    orders=pd.DataFrame(market.journal);filled=orders.loc[orders.notional>0]
    assert filled.datetime.isin(trades).all()
    expected_fees=np.where(filled.direction.eq(1),np.maximum(filled.notional*.0005,5.),np.maximum(filled.notional*.0015,5.))
    np.testing.assert_allclose(filled.fee,expected_fees,atol=1e-7)
    totals=filled.groupby('datetime')[['notional','fee']].sum().reindex(days,fill_value=0.)
    previous=r.account.shift(1,fill_value=1e6)
    np.testing.assert_allclose(r.cost*previous,totals.fee,atol=1e-5)
    np.testing.assert_allclose((1+r['return']-r.cost).cumprod()*1e6,r.account,atol=1e-4)
    holdings=[];contrib=[];old_values={};cash=1e6;quantities={}
    for day in days:
        position=positions[day];market.check_holding_limit(position,day,'close')
        current={}
        today=filled.loc[filled.datetime.eq(day)]
        for o in today.itertuples():
            assert np.isclose(o.cash_before,cash,atol=1e-5)
            signed=o.adjusted_quantity*(1 if o.direction==1 else -1)
            quantities[o.instrument]=quantities.get(o.instrument,0.)+signed
            assert quantities[o.instrument]>=-1e-7
            cash+=(-o.notional if o.direction==1 else o.notional)-o.fee
            assert cash>=-1e-5
        for stock in position.get_stock_list():
            q=position.get_stock_amount(stock);p=position.get_stock_price(stock);current[stock]=q*p
            np.testing.assert_allclose(q,quantities[stock],atol=1e-5)
            holdings.append(dict(date=day,instrument=stock,quantity=q,price=p,value=q*p,weight=q*p/r.loc[day,'account']))
        np.testing.assert_allclose([cash,cash+sum(current.values())],[r.loc[day,'cash'],r.loc[day,'account']],atol=1e-4)
        flow=today.assign(flow=np.where(today.direction.eq(1),today.notional,-today.notional)).groupby('instrument').flow.sum()
        for stock in set(old_values)|set(current)|set(flow.index):
            contrib.append(dict(date=day,instrument=stock,pnl_before_fees=current.get(stock,0)-old_values.get(stock,0)-flow.get(stock,0)))
        old_values=current
    h=pd.DataFrame(holdings);pnl=pd.DataFrame(contrib)
    np.testing.assert_allclose(previous+pnl.groupby('date').pnl_before_fees.sum().reindex(days,fill_value=0)-totals.fee,r.account,atol=1e-4)
    daily=pd.DataFrame(dict(date=days,nav=r.account.values,cash=r.cash.values,
        net_return=(r['return']-r.cost).values,fees=totals.fee.values,
        turnover=(totals.notional/previous).values,
        holdings=h.groupby('date').size().reindex(days,fill_value=0).values,
        cash_ratio=(r.cash/r.account).values,max_weight=h.groupby('date').weight.max().reindex(days,fill_value=0).values))
    for label,frame in dict(daily=daily,orders=orders,holdings=h,contributions=pnl).items():
        frame.to_parquet(directory/(label+'.parquet'),index=False)
    pd.DataFrame(market.generation_journal).to_parquet(directory/'generation_diagnostics.parquet',index=False)
    save(directory/'reconciliation.json',dict(status='passed',cash_nav_fees_quantities=True,
         schedule_count=len(trades),filled_orders=len(filled),final_nav=float(r.account.iloc[-1]),
         first_signal=str(signals[0].date()),last_execution=str(trades[-1].date()),no_2026=True))
    save(done,dict(status='complete',protocol_sha256=digest(OUT/'protocol.json'),
        artifacts={str(f):digest(f) for f in directory.iterdir() if f.is_file() and f.name!='complete.json'}))
    print(f'COMPLETE {name} NAV={r.account.iloc[-1]:.2f} fees={totals.fee.sum():.2f}',flush=True)


def report(benchmark):
    from paper_futu.top5_buffer_report import metrics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];annual=[];fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for name in CASES:
        directory=OUT/'accounts'/name;c=read(directory/'complete.json');verify(c['artifacts'])
        daily=pd.read_parquet(directory/'daily.parquet');orders=pd.read_parquet(directory/'orders.parquet')
        m=metrics(daily,benchmark);m.update(case=name,filled_orders=int(orders.notional.gt(0).sum()))
        rows.append(m)
        for year,g in daily.groupby(daily.date.dt.year):annual.append(dict(case=name,year=int(year),**metrics(g,benchmark)))
        nav=daily.nav/1e6;peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
        axes[0].plot(daily.date,nav,label=name)
        axes[1].plot(daily.date,(nav/peak-1)*100,label=name)
        contributions=pd.read_parquet(directory/'contributions.parquet').groupby('instrument').pnl_before_fees.sum().sort_values(ascending=False)
        contributions.to_csv(directory/'stock_contributions.csv',encoding='utf-8-sig')
    axes[0].plot(benchmark.index,(1+benchmark).cumprod(),label='H00300 total return',color='black',linestyle='--')
    axes[0].set_ylabel('NAV (initial = 1)');axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'nav_drawdown.png',dpi=160);plt.close(fig)
    df=pd.DataFrame(rows);yearly=pd.DataFrame(annual)
    df.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    yearly.to_csv(OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    show=df[['case','total_return','cagr','max_drawdown','sharpe','information_ratio','geometric_excess','fees','filled_orders','average_holdings','cash_ratio']].copy()
    show['case']=show.case.map(LABELS)
    for col in ('total_return','cagr','max_drawdown','geometric_excess','cash_ratio'):show[col]=show[col].map(lambda x:f'{x:.2%}')
    for col in ('sharpe','information_ratio','average_holdings'):show[col]=show[col].map(lambda x:f'{x:.3f}')
    show.fees=show.fees.map(lambda x:f'{x:,.2f}')
    show.columns=['策略','累计净收益','年化','最大回撤','夏普(无风险1%)','信息比率','累计几何超额','费用(元)','成交笔数','平均持仓','平均现金占比']
    y=yearly.pivot(index='year',columns='case',values='total_return').rename(columns=LABELS)
    y['H00300全收益']=yearly.groupby('year').benchmark_return.first()
    y=y.map(lambda x:f'{x:.2%}')
    p=read(OUT/'protocol.json');verify(p['protected']);verify(p['sources']);assert p['implementation']==code_hashes()
    limitations=''.join('<li>'+html.escape(x)+'</li>' for x in p['limitations'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>横截面动量与S6</title>
<style>body{font:16px/1.65 Arial,"Microsoft Yahei",sans-serif;margin:35px;max-width:1450px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #ddd;text-align:right}th{background:#eef3f8}img{max-width:100%}code{background:#eee}</style>
<h1>经典横截面动量与S6：2022—2025</h1>
<p>区间2022-01-04至2025-12-31，初始100万元。动态沪深300股票池；买入0.05%、卖出0.15%、每笔最低5元；同一已审计开盘成交引擎。年化按252个交易日计算，夏普无风险年利率1%，信息比率对H00300全收益指数。年末估值，不虚构强制卖出。</p>
<p>动量分数为复权收盘价[t−21]/复权收盘价[t−252]−1。月末收盘形成信号，次月首个交易日开盘交易；首次用2021-12-31信号于2022-01-04建仓。2022年前行情只用于形成历史信号或训练；不使用2026数据。不增加止盈止损，月度组合无额外换股数量限制。</p>
<p>S6主策略复用16个季度模型，前6名40%、其余14名60%，每5日调仓、最多替换6只；S6月度等权对照与动量保持相同持仓规则，用于区分信号与组合差异。交易仅执行仓位差，调仓间权重漂移。</p>'''+show.to_html(index=False,escape=False)+'<h2>逐年净收益</h2>'+y.to_html()+'<h2>净值与回撤</h2><img src="nav_drawdown.png"><h2>边界与限制</h2><ul>'+limitations+'</ul><p>本轮不训练、不搜索参数、不修改S6模拟盘。此前历史区间已反复研究，结果不是独立样本外证明。红利低波尚未纳入本轮。</p></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('run.lock','status.json','completion.json','report_manifest.json')}
    save(OUT/'report_manifest.json',dict(artifacts=artifacts,source_verified=True,protected_verified=True))
    save(OUT/'completion.json',dict(status='complete',accounts=3,source_unchanged=True,
         report_manifest_sha256=digest(OUT/'report_manifest.json')))
    print(df[['case','total_return','cagr','max_drawdown','fees']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage',choices=['audit','backtest','report','all'],default='all')
    args=parser.parse_args()
    with lock():
        started=time.time();save(OUT/'status.json',dict(status='running',pid=os.getpid(),stage=args.stage))
        try:
            calendar,pred=audit()
            if args.stage=='audit':return
            benchmark=pd.read_parquet(SOURCE/'data/benchmark.parquet').benchmark_return
            expected=calendar[(calendar>=START)&(calendar<=END)]
            assert benchmark.index.equals(expected) and np.isfinite(benchmark).all()
            if args.stage in ('backtest','all'):
                mom,codes=build_momentum(calendar);init_qlib(SOURCE/'data/provider')
                for name,config in CASES.items():
                    save(OUT/'status.json',dict(status='running',pid=os.getpid(),case=name,stage='backtest'))
                    replay(name,config,mom if config['signal']=='momentum' else pred,calendar,codes,benchmark)
            if args.stage in ('report','all'):report(benchmark)
            save(OUT/'status.json',dict(status='complete',elapsed=time.time()-started))
        except BaseException as e:
            save(OUT/'status.json',dict(status='failed',error=repr(e)));raise


if __name__=='__main__':main()
