"""4% versus 5% stop-loss ablation using sealed minute29 predictions; no model retraining.

python -m paper_futu.research_cn_intraday_stop4 --stage all
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paper_futu.research_cn_intraday import (
    ROOT, RUN as BASE, SOURCE, END, TEST_START, read, save, digest, calendars,
    verify as verify_original, binary_series,
)
from paper_futu.cn_intraday_stop_account import StopPolicyAccount

OUT=BASE.parent.parent/'cn_intraday_stop4_asof_20251231'
CASES={'stop5':(80,True,.05),'stop4':(80,True,.04)}
LABELS={'stop5':'前80/1，止盈10%／止损5%','stop4':'前80/1，止盈10%／止损4%'}


def implementation():
    return {str(ROOT/p):digest(ROOT/p) for p in (
        'paper_futu/research_cn_intraday_stop4.py','paper_futu/cn_intraday_stop_account.py',
        'scripts/audit_cn_intraday_results.py','paper_futu/top5_buffer_report.py')}


def verify():
    verify_original()
    p=read(OUT/'protocol.json')
    if p['implementation']!=implementation():raise RuntimeError('Implementation changed')
    for path,h in p['inputs'].items():
        if digest(path)!=h:raise RuntimeError(f'Changed input {path}')
    return p


def audit():
    verify_original()
    if read(BASE/'independent_audit.json')['status']!='passed':raise RuntimeError('Original audit missing')
    inputs={}
    for f in (BASE/'protocol.json',BASE/'completion.json',BASE/'independent_audit.json',BASE/'train_complete.json'):
        inputs[str(f)]=digest(f)
    for path,h in read(BASE/'completion.json')['artifacts'].items():
        if digest(path)!=h:raise RuntimeError('Original report changed')
        inputs[path]=h
    for f in (BASE/'predictions').glob('*.parquet'):inputs[str(f)]=digest(f)
    if len(list((BASE/'predictions').glob('*.parquet')))!=48:raise RuntimeError('Missing predictions')
    for f in (BASE/'feature_checks').glob('*.json'):
        for path,h in read(f)['artifacts'].items():
            if digest(path)!=h:raise RuntimeError('Feature/market source changed')
            if Path(path).parent.name=='market_parts':inputs[path]=h
    for r in read(BASE/'train_complete.json')['models']:
        path=BASE/'models'/r['variant']/f'{r["year"]}Q{r["quarter"]}'/'model.txt'
        if digest(path)!=r['model_sha256']:raise RuntimeError('Original model changed')
        inputs[str(path)]=r['model_sha256']
    for bp in (0,10):
        for f in (BASE/'accounts'/f'minute29_{bp}bp').glob('*'):
            if f.is_file():inputs[str(f)]=digest(f)
    sealed=read(SOURCE.parent/'audit_complete.json')['artifacts']
    for path,h in sealed.items():
        if str(SOURCE/'provider') in path or path==str(SOURCE/'benchmark.parquet'):
            if digest(path)!=h:raise RuntimeError('Sealed daily data changed')
            inputs[path]=h
    p=dict(start=TEST_START,end=END,model='sealed minute29',retrain=False,
        cases={c:dict(rank_exit=r,strong_entry=s,stop_loss=sl) for c,(r,s,sl) in CASES.items()},
        stop_semantics='Fee-inclusive cumulative liquidation loss relative to initial fee-inclusive cost; only stop budget changes. T+1 and execution gaps may exceed the threshold.',
        base_rules_overridden_by_cases=read(BASE/'protocol.json')['rules'],extra_per_side_bp=[0,10],
        inputs=inputs,implementation=implementation(),
        limitations=read(BASE/'protocol.json')['limitations'],
        fees='Same fixed Futu-paper fee convention; custody excluded; rounding unverified',
        comparison='2022-2025 already explored; not an independent test; no production changes')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if read(OUT/'protocol.json')!=p:raise RuntimeError('Existing protocol differs')
    else:save(OUT/'protocol.json',p)
    save(OUT/'audit_complete.json',dict(status='passed',input_files=len(inputs),prediction_months=48))


def backtest():
    verify()
    if (OUT/'backtest_complete.json').exists():return
    cal,days,grid=calendars();test_days=days[days>=TEST_START];testgrid=grid[grid>=TEST_START]
    accounts={(case,bp):StopPolicyAccount(bp/10000,rank_exit=r,strong_entry=s,stop_loss=sl)
              for case,(r,s,sl) in CASES.items() for bp in (0,10)}
    codes=[p.stem for p in sorted((BASE/'market_parts').glob('*.parquet'))]
    official={};prior={}
    for c in codes:
        s=binary_series(SOURCE/'provider',c,'close',cal)
        official[c]=s.reindex(test_days).to_numpy();prior[c]=s.shift().reindex(test_days).to_numpy()
    carry=[];index_of={d:i for i,d in enumerate(test_days)}
    for month in pd.period_range(TEST_START,END,freq='M'):
        lo,hi=month.start_time,month.end_time
        predictions=pd.read_parquet(BASE/'predictions'/f'{month}.parquet',columns=['instrument','minute29'])
        if not np.isfinite(predictions.minute29).all():raise RuntimeError('Nonfinite score')
        if predictions.reset_index().duplicated(['datetime','instrument']).any():raise RuntimeError('Duplicate prediction')
        parts=[]
        for c in codes:
            m=pd.read_parquet(BASE/'market_parts'/f'{c}.parquet',filters=[('datetime','>=',lo),('datetime','<=',hi)])
            m['instrument']=c;parts.append(m)
        market=pd.concat(parts);del parts
        pgroup=dict(tuple(predictions.groupby(level=0)));mgroup=dict(tuple(market.groupby(level=0)))
        del predictions,market;gc.collect()
        for ts in testgrid[(testgrid>=lo)&(testgrid<=hi)]:
            m=mgroup[ts].set_index('instrument');quotes=m.to_dict('index')
            if len(carry)>=2:
                signal,sc,sp=carry.pop(0);execution=ts-pd.Timedelta(minutes=5)
                di=index_of[ts.normalize()];previous={c:prior[c][di] for c in codes}
                for a in accounts.values():a.step(signal,execution,sc,sp,quotes,previous)
            p=pgroup[ts];scores=dict(zip(p.instrument,p.minute29))
            carry.append((ts,scores,dict(zip(m.index,m.close))))
            if ts.hour==15:
                di=index_of[ts.normalize()];closes={c:official[c][di] for c in codes}
                for a in accounts.values():a.close_day(ts.normalize(),closes)
        state=dict(stage='backtest',month=str(month),months_complete=(month.year-2022)*12+month.month,total_months=48,
            time=time.time(),nav={f'{c}_{bp}bp':a.nav() for (c,bp),a in accounts.items()})
        save(OUT/'progress.json',state)
        print('BACKTEST '+str(month)+' '+str({k:round(v,2) for k,v in state['nav'].items()}),flush=True)
        del pgroup,mgroup;gc.collect()
    records=[]
    for (case,bp),a in accounts.items():
        folder=OUT/'accounts'/f'{case}_{bp}bp';folder.mkdir(parents=True,exist_ok=True)
        check=a.reconcile();artifacts={}
        for name,rows in [('daily',a.daily),('orders',a.orders),('holdings',a.holdings),('closed',a.closed)]:
            frame=pd.DataFrame(rows)
            if case=='stop5':
                old=pd.read_parquet(BASE/'accounts'/f'minute29_{bp}bp'/f'{name}.parquet')
                pd.testing.assert_frame_equal(frame,old,check_exact=True)
            frame.to_parquet(folder/f'{name}.parquet',index=False)
            artifacts[str(folder/f'{name}.parquet')]=digest(folder/f'{name}.parquet')
        save(folder/'reconciliation.json',check)
        records.append(dict(case=case,extra_bp=bp,**check,artifacts=artifacts,
                            sealed_baseline_exact_match=case=='stop5'))
    save(OUT/'backtest_complete.json',dict(status='complete',accounts=records))


def report():
    from scripts.audit_cn_intraday_results import check_account
    from paper_futu.top5_buffer_report import metrics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify();complete=read(OUT/'backtest_complete.json');checks={};rows=[];annual=[];exits=[]
    benchmark=pd.read_parquet(SOURCE/'benchmark.parquet').benchmark_return
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for r in complete['accounts']:
        for path,h in r['artifacts'].items():
            if digest(path)!=h:raise RuntimeError('Account artifact changed')
        case,bp=r['case'],r['extra_bp'];key=f'{case}_{bp}bp';folder=OUT/'accounts'/key
        if case=='stop5':
            checks[key]=dict(status='passed',method='Exact equality of every daily/holdings/orders/closed row to independently audited original')
        else:checks[key]=check_account(folder,bp)
        print('AUDIT '+key+' passed',flush=True)
        d=pd.read_parquet(folder/'daily.parquet');c=pd.read_parquet(folder/'closed.parquet');o=pd.read_parquet(folder/'orders.parquet')
        d['date']=pd.to_datetime(d.date)
        if not pd.DatetimeIndex(d.date).equals(benchmark.index):raise RuntimeError('Unequal comparison interval')
        cal=pd.DatetimeIndex(d.date)
        c['holding_sessions']=cal.get_indexer(pd.to_datetime(c.exit_day))-cal.get_indexer(pd.to_datetime(c.entry_day))
        n=len(c);rank=int(c.reason.isin(['rank','strong']).sum())
        rows.append(dict(case=case,extra_bp=bp,**metrics(d,benchmark),fills=len(o),closed=n,rank_exits=rank,
            rank_exit_share=rank/n,win_rate=float(c.pnl.gt(0).mean()),mean_holding_sessions=float(c.holding_sessions.mean()),
            mean_win=float(c.loc[c.pnl.gt(0),'return_on_cost'].mean()),mean_loss=float(c.loc[c.pnl.lt(0),'return_on_cost'].mean())))
        for year,g in d.groupby(d.date.dt.year):annual.append(dict(case=case,extra_bp=bp,year=int(year),**metrics(g,benchmark)))
        for reason,g in c.groupby('reason'):exits.append(dict(case=case,extra_bp=bp,reason=reason,count=len(g),share=len(g)/n,pnl=float(g.pnl.sum())))
        if bp==0:
            nav=d.nav.to_numpy()/1e6;axes[0].plot(d.date,nav,label=case)
            axes[1].plot(d.date,nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1,label=case)
    axes[0].plot(benchmark.index,(1+benchmark).cumprod(),color='black',linestyle='--',label='H00300')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Net asset value');axes[1].set_ylabel('Drawdown');fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=150);plt.close(fig)
    summary=pd.DataFrame(rows);years=pd.DataFrame(annual)
    for name,frame in [('comparison',summary),('yearly',years),('exit_reasons',pd.DataFrame(exits))]:frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    save(OUT/'independent_audit.json',dict(status='passed',accounts=checks,baseline_exactly_reproduced=True,source_and_production_unchanged=True))
    def table(frame,cols,percent=()):
        f=frame[list(cols)].copy();f['case']=f.case.map(LABELS)
        for col in percent:f[col]=f[col].map(lambda v:f'{v:.2%}')
        for col in ('fees','mean_holding_sessions'):
            if col in f:f[col]=f[col].map(lambda v:f'{v:,.2f}')
        return f.rename(columns=cols).to_html(index=False)
    top=table(summary,dict(case='规则',extra_bp='每边额外摩擦(bp)',total_return='累计净收益',cagr='净年化',max_drawdown='最大回撤',
        fees='全期费用(元)',closed='完整交易次数',rank_exits='排名退出次数',win_rate='胜率',mean_holding_sessions='平均持有交易日'),
        ('total_return','cagr','max_drawdown','win_rate'))
    year=table(years,dict(case='规则',extra_bp='额外摩擦(bp)',year='年份',total_return='年度净收益',max_drawdown='年度最大回撤',benchmark_return='H00300收益',geometric_excess='几何超额'),
        ('total_return','max_drawdown','benchmark_return','geometric_excess'))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分钟策略止损4%消融</title><style>body{font:16px/1.7 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #ddd;text-align:right}th{background:#eef3f8}.table{overflow:auto}img{max-width:100%}</style><h1>分钟策略：止损5%与4%对照</h1><p>2022-01-04至2025-12-31，初始100万元。复用已审计的29因子季度模型与5分钟预测，不重训、不调参。只把含费累计止损预算由初始含费成本的5%改为4%。</p><p>其他规则不变：Top10、前80退出、新进第1名替换、止盈10%全部退出；新仓目标9.5%，保留仓位漂移；止盈无额外冷却，止损最早次交易日重入，保留A股T+1。信号隔一根完整5分钟线后按后续开盘成交，单根参与量不超过5%。止损阈值不保证实际成交损失上限。期末只估值、不虚构平仓。5%原版从头回放并逐笔严格复现此前账户。</p><h2>全期（已经扣费）</h2><div class="table">'+top+'</div><img src="curves.png"><h2>逐年</h2><div class="table">'+year+'</div><h2>口径及限制</h2><p>基础费用沿用原项目富途模拟假设，未含托管费；额外10bp为每边0.1%，重新回放账户并影响含费止损及资金路径，不是从年化中简单扣除。无风险年利率1%，年化按252交易日。沪深300全收益H00300为绩效基准；分钟相对收益使用510300 ETF代理。</p><p>公开分钟数据来源及历史财务修订仍有限制；复权库存简化公司行为，统一9.5%方向过滤不能完整还原板块涨跌停、排队和市场冲击。止损阈值不保证成交损失上限。历史区间已反复研究，不是新独立测试。本轮不切换V10、不连接券商、不修改持仓或调度。</p><p><a href="comparison.csv">完整指标</a> · <a href="yearly.csv">逐年数据</a> · <a href="exit_reasons.csv">退出原因</a> · <a href="independent_audit.json">独立核验</a> · <a href="protocol.json">固定配置与哈希</a></p></html>'
    (OUT/'report.html').write_text(page,encoding='utf8')
    verify()
    artifacts={str(p):digest(p) for p in OUT.glob('*') if p.suffix in ('.csv','.html','.png')}
    artifacts[str(OUT/'independent_audit.json')]=digest(OUT/'independent_audit.json')
    save(OUT/'completion.json',dict(status='complete_and_audited',accounts=4,new_model_fits=0,protocol_sha256=digest(OUT/'protocol.json'),artifacts=artifacts))
    print(summary.to_string(index=False),flush=True)


def main():
    import msvcrt
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--stage',choices=['audit','backtest','report','all'],default='all');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        stage='init'
        try:
            for stage in ('audit','backtest','report'):
                if args.stage in (stage,'all'):
                    save(OUT/'status.json',dict(status='running',stage=stage,pid=os.getpid(),time=time.time()))
                    globals()[stage]()
            save(OUT/'status.json',dict(status='complete' if (OUT/'completion.json').exists() else 'stage_complete',pid=os.getpid(),time=time.time()))
        except Exception as exc:
            save(OUT/'status.json',dict(status='failed',stage=stage,error=repr(exc),pid=os.getpid(),time=time.time()))
            raise


if __name__=='__main__':main()
