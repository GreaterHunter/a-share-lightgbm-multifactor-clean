"""Three frozen S6 construction policies, five phases, three cost levels.

Offline only. Reuses audited Qlib execution and immutable predictions/risk data.
No broker import, training, deployment mutation, or historical parameter search.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from functools import partial
import gc
import html
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='4'
import numpy as np
import pandas as pd
from paper_futu import research_barra as base
from src.barra_overlay import POLICY, causal_risk, asset_covariance, adjusted_scores, optimize_weights

ROOT, DATA, SOURCE = base.ROOT, base.DATA, base.ACCOUNT
RISK=ROOT/'reports/experiments/barra_s6_industry_size_asof_20261013/main'
OUT=ROOT/'reports/experiments/s6_barra_overlay_asof_20261013'
START,END=base.START,base.END
read,save,digest=base.read,base.save,base.sha
CASES=('R0','R1','R2');COSTS=(0,10,20)
PORTFOLIO=dict(top_k=20,n_drop=6,top_tier_size=6,top_tier_weight=.4)


def verify():
    p=read(OUT/'protocol.json')
    for path,h in {**p['sources'],**p['implementation']}.items():
        if digest(path)!=h:raise ValueError('Bound input changed: '+path)
    return p


def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    base.verify_sources(RISK)
    for stage in ('data','factors','analysis'):assert base.stage_done(RISK,stage)
    assert read(RISK/'full_complete.json')['status']=='complete'
    source=read(SOURCE/'streams.json')['streams']['history_QREG']
    assert digest(source['path'])==source['sha256']
    sources={str(SOURCE/'streams.json'):digest(SOURCE/'streams.json'),source['path']:source['sha256']}
    sources.update(read(RISK/'protocol.json')['sources'])
    for name in ('exposures.parquet','factor_returns.parquet','specific_returns.parquet',
                 'data_profile.json','full_binding.json','protocol.json','full_complete.json'):
        path=RISK/name;sources[str(path)]=digest(path)
    official=ROOT/'data/external/csi300_total_return_asof_20261013/H00300.json'
    sources[str(official)]=digest(official)
    files=['src/barra_overlay.py','src/barra_risk.py','paper_futu/research_s6_barra_overlay.py',
           'paper_futu/final_ablation_account.py','paper_futu/final_ablation_core.py',
           'paper_futu/s6_explicit_account.py','src/portfolio_strategy.py','src/compare_models.py',
           'paper_futu/benchmark_attribution.py']
    p=dict(name='S6 Barra overlay v1',start=START,end=END,policy=POLICY,
        cases=dict(R0='Original S6 ranking/weights',R1='Same S6 selection rule; optimized weights',
                   R2='Risk-adjusted selection from top60 plus holdings, then optimized weights'),
        account_count=45,offsets=list(range(5)),extra_friction_bp=list(COSTS),portfolio=PORTFOLIO,
        prediction=source,initial_cash=1e6,risk_degree=.95,
        risk='Prior-close factor exposure, covariance and specific variance; 252-day annualization',
        selection='R2 subtracts 0.10 marginal-risk percentile and 0.10 excess-industry penalty from score percentiles; deterministic heuristic, not global joint optimum',
        objective='annual_variance/0.20^2 + 2*weight_anchor_squared/0.95^2 + sector_excess_squared/0.30^2 + 0.02*smooth_L1_turnover/0.95',
        constraints='NAV weights 1%-8%, total95%, weighted percentile score floor reference minus .02; tradability bounds; locked inherited excess explicit',
        risk_failure='Record failure; revert to original S6 selection and weights for that rebalance; never label fallback risk-compliant',
        industry_limit='30% is a soft penalty threshold, NOT a guaranteed cap',
        R1_comparison='Same selection algorithm, not forced identical executed lists: rounding, tradability and account path can change later names',
        execution='Existing audited daily ideal-open engine; preceding-close sizing, raw100share lots, +/-9.5% directional filter, T+1, min fee5; recent live integer allocator is NOT mixed into this comparison',
        cost='buy0.05%/sell0.15%, minimum5 plus extra friction on executed notional; no return subtraction shortcut',
        limitations=['Historical dates already used for research','Risk model is self-built, not licensed Barra',
                    'Industry first-publication archive incomplete; existing93day assumption retained',
                    'No historical index weights: absolute risk only, not benchmark-relative forecast TE',
                    'Close-close covariance proxy for open-executed strategy; annual risk assumes time scaling',
                    'No hard maximum drawdown guarantee; no forward independent alpha proof',
                    'Daily lot/cash drift can differ from continuous target; actual risk is separately recorded'],
        runtime=dict(workers=1,threads=4,task_free_gib=.5,no_model_training=True),
        sources=sources,implementation={str(ROOT/f):digest(ROOT/f) for f in files})
    path=OUT/'protocol.json'
    if path.exists() and read(path)!=p:raise ValueError('Protocol changed: preserve output or use a new experiment')
    save(path,p)
    streams={f'history_{case}_f{cost}':source for case in CASES for cost in COSTS}
    save(OUT/'streams.json',dict(streams=streams));verify()
    print('AUDIT complete; fixed 45 accounts, no fitting',flush=True)


def prepare():
    verify()
    marker=OUT/'risk_cache_complete.json'
    if marker.exists():
        for p,h in read(marker)['artifacts'].items():assert digest(p)==h
        return
    f=pd.read_parquet(RISK/'factor_returns.parquet');u=pd.read_parquet(RISK/'specific_returns.parquet')
    e=pd.read_parquet(RISK/'exposures.parquet');columns=list(f.columns)
    calendar=pd.DatetimeIndex(pd.read_csv(DATA/'provider/calendars/day.txt',header=None)[0])
    prior=calendar[calendar<pd.Timestamp(START)][-1]
    e=e.loc[(e.index.get_level_values('datetime')>=prior)&(e.index.get_level_values('datetime')<=pd.Timestamp(END))]
    paths={};industry=[i for i,c in enumerate(columns) if c.startswith('industry:')]
    for i,(day,g) in enumerate(e.groupby(level='datetime',sort=True)):
        g=g.droplevel('datetime');x,cov,d=causal_risk(day,f,u,g,columns)
        path=OUT/'risk_cache'/f'{day:%Y%m%d}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,codes=g.index.to_numpy(dtype=str),x=x,f=cov,d=d,
                            industry=x[:,industry],date=str(day.date()))
        paths[str(path)]=digest(path)
        if i%150==0:print('Risk snapshots',i,flush=True)
    save(marker,dict(artifacts=paths,columns=columns,days=len(paths),
        maximum_observation_date=END,causality='Histories explicitly sliced through each snapshot date'))
    del e,f,u;gc.collect()
    import pyarrow as pa
    pa.default_memory_pool().release_unused()
    print('Risk snapshots complete',len(paths),flush=True)


class RiskCache:
    def __init__(self):
        record=read(OUT/'risk_cache_complete.json');self.columns=record['columns'];self.hashes=record['artifacts']
        self.items=OrderedDict()
    def get(self,day):
        day=pd.Timestamp(day).normalize();key=f'{day:%Y%m%d}'
        if key not in self.items:
            path=OUT/'risk_cache'/f'{key}.npz'
            if str(path) not in self.hashes:return None
            if digest(path)!=self.hashes[str(path)]:raise ValueError('Risk cache changed')
            with np.load(path,allow_pickle=False) as z:
                r={k:z[k] for k in ('codes','x','f','d','industry')}
                assert str(z['date'])==str(day.date())
            r['lookup']={s:i for i,s in enumerate(r['codes'])};self.items[key]=r
            if len(self.items)>24:self.items.popitem(last=False)
        return self.items[key]


def make_strategy(case,cache,selection_log):
    from paper_futu.final_ablation_account import AuditedStrategy,select_holdings
    from qlib.backtest.decision import Order
    class Overlay(AuditedStrategy):
        def generate_target_weight_position(self,score,current,trade_start_time,trade_end_time):
            if isinstance(score,pd.DataFrame):score=score.iloc[:,0]
            score=score.dropna();day=pd.Timestamp(trade_start_time).normalize()
            step=self.trade_calendar.get_trade_step()
            signal_day=pd.Timestamp(self.trade_calendar.get_step_time(step,shift=1)[0]).normalize()
            assert signal_day<day
            original=super().generate_target_weight_position(score,current,trade_start_time,trade_end_time)
            rec=dict(datetime=str(day.date()),signal_date=str(signal_day.date()),case=case,
                     original_selected=list(original),status='baseline')
            if not original:return original
            if case=='R0':
                rec.update(selected=list(original),target_weights={s:.95*w for s,w in original.items()})
                selection_log.append(rec);return original
            r=cache.get(signal_day)
            if r is None or not set(original).issubset(r['lookup']):
                rec.update(status='fallback_missing_selected_risk',selected=list(original),
                           target_weights={s:.95*w for s,w in original.items()})
                selection_log.append(rec);return original
            buyable={s for s in score.index if self.trade_exchange.is_stock_tradable(s,trade_start_time,trade_end_time,direction=Order.BUY)}
            held=set(current.get_stock_list())
            sellable={s for s in held if self.trade_exchange.is_stock_tradable(s,trade_start_time,trade_end_time,direction=Order.SELL)}
            nav=current.calculate_value()
            current_w={s:current.get_stock_amount(s)*current.get_stock_price(s)/nav for s in held}
            pct=score.rank(method='average',pct=True)
            selected=list(original)
            if case=='R2':
                pool=set(score.sort_values(ascending=False,kind='stable').head(POLICY['candidate_count']).index)|held|set(original)
                codes=sorted(pool&set(r['lookup'])&set(pct.index))
                ix=[r['lookup'][s] for s in codes]
                cov=asset_covariance(r['x'][ix],r['f'],r['d'][ix])
                ref=np.array([.95*original.get(s,0) for s in codes])
                adjusted=pd.Series(adjusted_scores(pct.loc[codes].to_numpy(),cov,r['industry'][ix],ref),index=codes)
                # Non-risk-covered unsellable holdings remain explicit and force fallback.
                selected=select_holdings(adjusted,sorted(held),20,6,buyable&set(codes),sellable)
                selected=sorted(selected,key=lambda s:(-score.get(s,-np.inf),s))
            if not set(selected).issubset(r['lookup']) or not set(selected).issubset(pct.index):
                rec.update(status='fallback_missing_retained_risk',selected=list(original),
                           target_weights={s:.95*w for s,w in original.items()})
                selection_log.append(rec);return original
            if len(held-set(selected))>6:raise ValueError('Replacement limit exceeded')
            n=len(selected);head=min(6,n)
            ref=np.array([.95*(.4/head if i<head else .6/(n-head)) if head<n else .95/n for i in range(n)])
            ix=[r['lookup'][s] for s in selected]
            cov=asset_covariance(r['x'][ix],r['f'],r['d'][ix])
            present=np.array([current_w.get(s,0) for s in selected])
            lo=np.full(n,POLICY['minimum_weight']);hi=np.full(n,POLICY['maximum_weight'])
            for i,s in enumerate(selected):
                if s in held and s not in sellable:lo[i]=max(lo[i],present[i]);hi[i]=max(hi[i],lo[i])
                if s not in buyable:hi[i]=min(hi[i],present[i])
            w,info=optimize_weights(cov,r['industry'][ix],pct.loc[selected].to_numpy(),ref,present,lower=lo,upper=hi)
            if not info['success']:
                rec.update(status='fallback_solver',optimization=info,selected=list(original),
                           target_weights={s:.95*v for s,v in original.items()})
                selection_log.append(rec);return original
            rec.update(status='optimized',optimization=info,selected=selected,
                target_weights=dict(zip(selected,w.tolist())),removed=sorted(held-set(selected)),
                target_score_relative_original=float(pct.loc[selected].to_numpy()@w/.95-
                    sum(pct[s]*v for s,v in original.items())))
            selection_log.append(rec)
            return dict(zip(selected,w/.95))
    return Overlay


def metrics(daily):
    r=daily.net_return.to_numpy();nav=np.cumprod(1+r);peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
    b=(1+daily.bench).prod();rf=np.expm1(np.log1p(.01)/252)
    return dict(total_return=float(nav[-1]-1),cagr=float(nav[-1]**(252/len(r))-1),
        benchmark_return=float(b-1),annual_geometric_excess=float((nav[-1]/b)**(252/len(r))-1),
        max_drawdown=float((1-nav/peak).max()),fees=float(daily.fees_cash.sum()),
        mean_daily_turnover=float(daily.turnover.mean()),mean_holdings=float(daily.holdings_count.mean()),
        sharpe=float((r.mean()-rf)/r.std(ddof=1)*np.sqrt(252)),
        information_ratio=float((daily.net_return-daily.bench).mean()/(daily.net_return-daily.bench).std()*np.sqrt(252)))


def account(case,offset,cost):
    p=verify();name=f'{case}_f{cost}';directory=OUT/f'accounts/history_{name}_{offset}';marker=directory/'complete.json'
    if marker.exists():
        m=read(marker);assert m['protocol_sha256']==digest(OUT/'protocol.json')
        for path,h in m['artifacts'].items():assert digest(path)==h
        print('Reuse',name,offset,flush=True);return
    from paper_futu import s6_explicit_account as adapter
    from paper_futu import final_ablation_account as engine
    from src import runtime_budget
    from paper_futu.benchmark_attribution import official_frame
    from paper_futu.overfit_four_core import verify_account
    from src.compare_models import summarize_report
    cache=RiskCache();journal=[];strategy=make_strategy(case,cache,journal)
    original_exchange=engine.AuditedExchange
    class Exchange(original_exchange):
        def reset_audit(self,friction,top_k=None):super().reset_audit(cost/10000,top_k)
    context=SimpleNamespace(DATA=DATA,OUT=OUT,NAME='S6_Barra_overlay',START=pd.Timestamp(START),
        HISTORY_START=pd.Timestamp(START),END=pd.Timestamp(END),PORTFOLIO=PORTFOLIO,
        protocol=lambda:p,read=read,digest=digest)
    original_guard=runtime_budget.require_memory
    def guard(required,purpose):return original_guard(.5,'Single-process, no-training Barra account')
    with patch.object(engine,'AuditedStrategy',strategy),patch.object(engine,'AuditedExchange',Exchange),patch.object(runtime_budget,'require_memory',guard):
        adapter._EXCHANGE=None;adapter.init_qlib(context)
        result,files=adapter.replay(context,name,offset,mode='history')
    daily=pd.read_parquet(directory/'daily.parquet')
    if case=='R0' and cost==0:
        old=pd.read_parquet(SOURCE/f'accounts/history_QREG_{offset}/daily.parquet')
        pd.testing.assert_index_equal(old.index,daily.index)
        np.testing.assert_allclose(daily.account,old.account,atol=1e-4,rtol=1e-10)
    official=official_frame(read(ROOT/'data/external/csi300_total_return_asof_20261013/H00300.json'),'H00300')
    b=official.close.pct_change(fill_method=None).reindex(daily.index)
    b.iloc[0]=official.loc[START,'close']/official.loc[START,'open']-1
    assert b.notna().all();daily['price_benchmark']=daily.bench;daily['bench']=b
    daily['benchmark_nav']=(1+b).cumprod();daily['active_return']=daily.net_return-b
    daily['relative_excess_nav']=daily.portfolio_nav/daily.benchmark_nav
    daily.to_parquet(directory/'daily.parquet')
    row=summarize_report(daily,name).iloc[0].to_dict();row['period']='all'
    pd.DataFrame([row]).to_csv(directory/'metrics.csv',index=False)
    audit_result=verify_account(directory)
    orders=pd.read_parquet(directory/'orders.parquet');filled=orders.loc[orders.notional>0]
    expected=np.maximum(5,filled.notional*np.where(filled.direction.eq(1),.0005,.0015))+filled.notional*cost/10000
    np.testing.assert_allclose(filled.fee,expected,atol=1e-8)
    assert all(pd.Timestamp(j['signal_date'])<pd.Timestamp(j['datetime']) for j in journal)
    save(directory/'selection.json',journal)
    holdings=pd.read_parquet(directory/'holdings.parquet');risk_rows=[]
    factor_returns=pd.read_parquet(RISK/'factor_returns.parquet');previous=None
    for day,g in holdings.groupby('datetime'):
        r=cache.get(day);values=g.set_index('instrument').market_value;w=values/daily.loc[day,'account']
        known=w.index.intersection(r['lookup']) if r else []
        coverage=float(w.reindex(known).sum()/w.sum()) if len(w) else 1.
        rec=dict(datetime=day,coverage=coverage,max_stock_weight=float(w.max()),stock_weight=float(w.sum()))
        if r and coverage>=.98:
            ix=[r['lookup'][s] for s in known];ww=w.loc[known].to_numpy()
            exp=ww@r['x'][ix];common=float(exp@r['f']@exp);specific=float(np.sum(ww**2*r['d'][ix]))
            sector=ww@r['industry'][ix]
            rec.update(predicted_volatility=float(np.sqrt((common+specific)*252)),
                bank_weight=float(exp[cache.columns.index('industry:SW:银行')]),
                max_industry_weight=float(sector.max()),common_risk_fraction=float(common/(common+specific)))
        else:rec.update(predicted_volatility=np.nan,bank_weight=np.nan,max_industry_weight=np.nan,common_risk_fraction=np.nan)
        risk_rows.append(rec)
    risk=pd.DataFrame(risk_rows);risk.to_parquet(directory/'realized_holdings_risk.parquet',index=False)
    periods={'all':metrics(daily),**{str(y):metrics(g) for y,g in daily.groupby(daily.index.year)}}
    for period,m in periods.items():
        rr=risk if period=='all' else risk.loc[risk.datetime.dt.year.eq(int(period))]
        for col in ('bank_weight','max_industry_weight','max_stock_weight','predicted_volatility','stock_weight'):
            m['mean_'+col]=float(rr[col].mean())
    for f in ('selection.json','realized_holdings_risk.parquet'):
        files.append(directory/f)
    save(marker,dict(case=case,cost_bp=cost,offset=offset,periods=periods,audit=audit_result,
        fallback_count=sum(j['status'].startswith('fallback') for j in journal),rebalance_count=len(journal),
        missing_risk_days=int(risk.predicted_volatility.isna().sum()),
        baseline_reproduced=case=='R0' and cost==0,protocol_sha256=digest(OUT/'protocol.json'),
        artifacts={str(f):digest(f) for f in files}))
    print('ACCOUNT',case,cost,offset,periods['all'],flush=True)


def report():
    verify();entries=[]
    for cost in COSTS:
        for case in CASES:
            for offset in range(5):
                d=OUT/f'accounts/history_{case}_f{cost}_{offset}';m=read(d/'complete.json')
                assert m['protocol_sha256']==digest(OUT/'protocol.json')
                for path,h in m['artifacts'].items():assert digest(path)==h
                entries.append(m)
    rows=[]
    for cost in COSTS:
        for case in CASES:
            group=[e for e in entries if e['cost_bp']==cost and e['case']==case]
            for period in ('all',*[str(y) for y in range(2022,2027)]):
                m={k:float(np.mean([e['periods'][period][k] for e in group])) for k in group[0]['periods'][period]}
                m.update(case=case,cost_bp=cost,period=period,
                    worst_phase_drawdown=max(e['periods'][period]['max_drawdown'] for e in group),
                    cagr_std=float(np.std([e['periods'][period]['cagr'] for e in group])),
                    total_fallbacks=sum(e['fallback_count'] for e in group))
                rows.append(m)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    save(OUT/'comparison.json',dict(status='complete',accounts=45,summary=rows,
        baseline_reproduced=all(e['baseline_reproduced'] for e in entries if e['case']=='R0' and e['cost_bp']==0)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True)
    names={'R0':'原S6','R1':'Barra调整权重','R2':'Barra调整选股与权重'}
    for case in CASES:
        daily=[pd.read_parquet(OUT/f'accounts/history_{case}_f0_{i}/daily.parquet') for i in range(5)]
        nav=pd.concat([d.portfolio_nav for d in daily],axis=1)
        dd=pd.concat([1-d.portfolio_nav/np.maximum(1,d.portfolio_nav.cummax()) for d in daily],axis=1)
        risk=pd.concat([pd.read_parquet(OUT/f'accounts/history_{case}_f0_{i}/realized_holdings_risk.parquet').set_index('datetime').bank_weight for i in range(5)],axis=1)
        axes[0].plot(nav.index,nav.mean(axis=1),label=names[case]);axes[1].plot(dd.index,-100*dd.mean(axis=1),label=names[case])
        axes[2].plot(risk.index,100*risk.mean(axis=1),label=names[case])
    axes[0].plot(daily[0].index,daily[0].benchmark_nav,color='black',ls='--',label='沪深300全收益')
    for ax,title in zip(axes,['五起点平均净值（成本后）','五起点平均日终回撤（%）','五起点平均银行仓位（%账户净值）']):
        ax.set_title(title);ax.legend(ncol=4);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    def table(headers,values):
        return '<table><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in values)+'</table>'
    main=frame.loc[frame.period.eq('all')&frame.cost_bp.eq(0)]
    primary=[[names[r.case],f'{r.cagr:.2%}',f'{r.annual_geometric_excess:.2%}',f'{r.max_drawdown:.2%}',f'{r.worst_phase_drawdown:.2%}',f'{r.fees/10000:.2f}',f'{r.mean_bank_weight:.2%}',f'{r.mean_daily_turnover:.2%}'] for r in main.itertuples()]
    years=[]
    for year in range(2022,2027):
        g=frame.loc[frame.period.eq(str(year))&frame.cost_bp.eq(0)].set_index('case')
        years.append([str(year) if year<2026 else '2026截至10月13日',*[f"{g.loc[c,'total_return']:.2%}" for c in CASES],f"{g.iloc[0].benchmark_return:.2%}"])
    stress=[[names[r.case],r.cost_bp,f'{r.cagr:.2%}',f'{r.worst_phase_drawdown:.2%}',f'{r.fees/10000:.2f}'] for r in frame.loc[frame.period.eq('all')].itertuples()]
    total_fallbacks=sum(e['fallback_count'] for e in entries)
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>S6加入Barra后的组合回测</title><style>body{{max-width:1250px;margin:30px auto;padding:0 24px;font:16px/1.7 system-ui;color:#193448}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #dde6ec}}th{{background:#edf4f8}}th:first-child,td:first-child{{text-align:left}}img{{width:100%}}.note{{background:#fff1d6;padding:18px}}</style>
<h1>S6加入Barra后的组合回测</h1><p>2022-01-04至2026-10-13；同一份S6季度重训H10预测、每5日调仓、最多20只与替换6只、95%目标股票仓位。3组×5起点×3成本＝45账户，不重新训练，不切换模拟盘。</p>
<h2>基础成本结果</h2>{table(['版本','平均复合年化','年化几何超额H00300','平均最大回撤','最差起点回撤','全期平均费用/万元','平均银行仓位','日均双边换手'],primary)}
<p>每条账户单独计算复合年化后平均；回撤含初始净值。费用从100万元起始账户实际成交计算。所有起点比较区间一致，跨年不重置。</p>
<img src="comparison.png"><h2>逐年收益</h2>{table(['年份',*[names[c] for c in CASES],'沪深300全收益'],years)}
<h2>每边增加交易摩擦</h2>{table(['版本','额外摩擦/bp','平均年化','最差回撤','平均费用/万元'],stress)}
<h2>固定规则与失败记录</h2><p>R1沿用S6选股算法，仅优化权重；账户路径、整手和成交限制可能使后续入选名单不同。R2从原排名前60、现持仓及原S6必要保留名单构建候选，预测百分位减0.10×边际风险百分位及0.10×行业超额惩罚，再沿用最多换6只的规则。它是确定性启发式，不是全局联合选股最优解。</p>
<p>连续权重：单股1%—8%账户净值、总和95%；加权预测百分位最多比本次参照降低2个百分点。目标函数包括预测方差、偏离原分层权重、行业超过30%的惩罚及换手惩罚；系数在回测前固定。30%行业线是软惩罚，不保证仓位低于30%。不可卖旧仓位的边界显式保留。风险缺失或求解失败返回原S6规则并记账，45账户合计{total_fallbacks}次回退，详见每账户selection.json，不能把回退称作风险约束已满足。</p>
<p class="note">沿用已审计历史成交引擎：前收盘估算股数、次日理想开盘成交、旧100股整手规则、统一±9.5%方向过滤、T+1和最低5元费用。近期模拟盘整手修复未混入此研究，不能把结果直接宣称为09:31可实现收益。连续目标风险与实际成交后的风险分别保存；现金和仓位可能因整手、跳空及不可成交偏离目标。</p>
<h2>数据与结论边界</h2><p>风险数据按前一交易日暴露和截至该日的因子/特异收益估计；行业沿用93日可用时间假设，完整首次发布档案仍缺失。自建Barra风格模型，不是MSCI商业模型。没有历史指数权重，风险控制是绝对风险，不报告伪造的指数主动风险。2022—2026已反复参与研究，本轮不是独立测试；不根据结果反复调整参数。H00300为税前全收益基准，策略分红以复权近似。1%无风险利率用于夏普。所有原数据、模型及部署哈希保持不变。</p>
<p>完整数值见comparison.csv与comparison.json；各账户保留净值、订单、费用、持仓、风险和选股理由。协议SHA256：{digest(OUT/'protocol.json')}。</p></html>'''
    (OUT/'report.html').write_text(body,encoding='utf-8')
    verify()
    files=[OUT/n for n in ('comparison.csv','comparison.json','comparison.png','report.html')]
    save(OUT/'report_manifest.json',{str(f):digest(f) for f in files})
    save(OUT/'completion.json',dict(status='complete',accounts=45,original_sources_unchanged=True,
        protocol_sha256=digest(OUT/'protocol.json'),report_manifest_sha256=digest(OUT/'report_manifest.json')))
    print(main.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['audit','prepare','account','report','all'],default='all')
    parser.add_argument('--case',choices=CASES,default='R0');parser.add_argument('--offset',type=int,choices=range(5),default=0)
    parser.add_argument('--cost',type=int,choices=COSTS,default=0);args=parser.parse_args()
    if args.stage=='audit':audit()
    elif args.stage=='prepare':prepare()
    elif args.stage=='account':account(args.case,args.offset,args.cost)
    elif args.stage=='report':report()
    else:
        from filelock import FileLock
        OUT.mkdir(parents=True,exist_ok=True)
        with FileLock(str(OUT/'controller.lock'),timeout=0):
            audit();prepare();(OUT/'logs').mkdir(exist_ok=True)
            done=0
            for cost in COSTS:
                for offset in range(5):
                    for case in CASES:
                        save(OUT/'status.json',dict(state='running',case=case,cost=cost,offset=offset,completed=done,total=45,pid=os.getpid(),updated=time.time()))
                        with (OUT/f'logs/{case}_f{cost}_{offset}.log').open('a',encoding='utf-8') as log:
                            result=subprocess.run([sys.executable,'-X','utf8','-u','-m','paper_futu.research_s6_barra_overlay',
                                '--stage','account','--case',case,'--offset',str(offset),'--cost',str(cost)],cwd=ROOT,
                                env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'},stdout=log,stderr=subprocess.STDOUT)
                        if result.returncode:
                            save(OUT/'status.json',dict(state='failed',case=case,cost=cost,offset=offset,exit_code=result.returncode,completed=done))
                            raise SystemExit(result.returncode)
                        done+=1;print('Finished',done,'/45',case,cost,offset,flush=True)
            report();save(OUT/'status.json',dict(state='complete',completed=45,total=45,updated=time.time()))


if __name__=='__main__':main()
