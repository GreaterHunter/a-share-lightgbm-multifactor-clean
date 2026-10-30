"""Fixed-definition historical style-regime diagnostic and observed forward NAV.

No strategy changes, re-training, order submission or reconstructed missing NAV.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import html
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from paper_futu import research_barra as base
from paper_futu.research_barra_full import OUT as FULL,residual_parts,bind,metadata_verify
from paper_futu.benchmark_attribution import official_frame
from paper_futu.cn_common import STATE_DIR,CHINA_TZ,china_now,HISTORICAL_PROVIDER

OUT=base.ROOT/'reports/experiments/s6_style_weakness_asof_20261013_forward_20261026'
REGIMES={0:'双弱',1:'银行弱、价值强',2:'银行强、价值弱',3:'双强'}


def lagged_regimes(factors,window=20):
    if not factors.index.is_unique or not factors.index.is_monotonic_increasing:raise ValueError('Invalid factor calendar')
    bank=[c for c in factors if c.startswith('industry:') and c.endswith('\u94f6\u884c')]
    if len(bank)!=1:raise ValueError('Ambiguous bank factor')
    result=pd.DataFrame(index=factors.index)
    result['bank_history']=factors[bank[0]].rolling(window,min_periods=window).sum().shift(1)
    value=(factors.book_to_price+factors.earnings_yield)/2
    result['value_history']=value.rolling(window,min_periods=window).sum().shift(1)
    result['known']=result[['bank_history','value_history']].notna().all(axis=1)
    result['regime']=pd.Series(pd.NA,index=result.index,dtype='Int64')
    valid=result.known
    result.loc[valid,'regime']=(result.loc[valid,'bank_history'].gt(0).astype(int)*2+
                               result.loc[valid,'value_history'].gt(0).astype(int))
    return result


def conditional_mean_hac(returns,mask,lags=20):
    """Indicator-weighted mean with HAC on the ORIGINAL calendar, not compressed days."""
    x=np.asarray(returns,float);m=np.asarray(mask,bool);n=len(x);k=int(m.sum())
    if n<=lags+2 or len(m)!=n or not np.isfinite(x).all() or k<20:raise ValueError('Insufficient valid conditional observations')
    mean=float(x[m].mean());influence=m*(x-mean)/(k/n)
    omega=float(influence@influence/n)
    for lag in range(1,lags+1):omega+=2*(1-lag/(lags+1))*float(influence[lag:]@influence[:-lag]/n)
    se=float(np.sqrt(max(omega,0)/n))
    return dict(observations=k,annual_conditional_arithmetic_mean=mean*252,
        ci_low=(mean-1.96*se)*252,ci_high=(mean+1.96*se)*252)


def close_observation(path,day):
    x=base.read(path);stamp=datetime.fromisoformat(x['time'])
    if stamp.tzinfo is None:raise ValueError('Naive account observation time')
    local=stamp.astimezone(CHINA_TZ)
    if str(local.date())!=day or (local.hour,local.minute)<(15,35):raise ValueError('Not a same-day closing observation')
    if not np.isfinite(x['assets']) or x['assets']<=0:raise ValueError('Invalid observed NAV')
    return x


def audit():
    OUT.mkdir(parents=True,exist_ok=True);metadata_verify();bind(FULL/'main','main');base.verify_sources(FULL/'main')
    source_paths=[FULL/'main/factor_returns.parquet',FULL/'main/full_binding.json']
    for phase in range(5):source_paths += [FULL/f'main/attribution_daily_{phase}.parquet',base.ACCOUNT/f'accounts/history_QREG_{phase}/daily.parquet']
    old_benchmark=base.ROOT/'data/external/csi300_total_return_asof_20261013/H00300.json'
    source_paths += [old_benchmark,base.ROOT/'models/paper_futu/s6_qreg_20260930/model.txt',
        base.ROOT/'configs/futu_cn_paper.json',base.ROOT/'reports/paper_futu/s6_forward_v1/protocol.json']
    old=base.read(OUT/'protocol.json') if (OUT/'protocol.json').exists() else None
    p=dict(start=base.START,end=base.END,window=20,
        bank='pure industry bank factor from frozen full main model',
        value='mean of book-to-price and earnings-yield factor returns',
        regime='previous 20 completed sessions summed, lagged 1 session; weak <= 0',
        no_parameter_search=True,no_account_changes=True,calendar_hac_lags=20,
        implementation_sha256=base.sha(__file__),sources={str(x):base.sha(x) for x in source_paths})
    if old and old!=p:raise ValueError('Diagnostic protocol changed')
    base.save(OUT/'protocol.json',p)


def historical():
    f=pd.read_parquet(FULL/'main/factor_returns.parquet');regime=lagged_regimes(f)
    official=official_frame(base.read(base.ROOT/'data/external/csi300_total_return_asof_20261013/H00300.json'),'H00300')
    benchmark=official.close.pct_change(fill_method=None)
    benchmark.loc[pd.Timestamp(base.START)]=official.loc[base.START,'close']/official.loc[base.START,'open']-1
    rows=[];by_year=[];quarters=[]
    bank=[c for c in f if c.startswith('industry:') and c.endswith('\u94f6\u884c')][0]
    factors=f.loc[base.START:base.END]
    for phase in range(5):
        account=pd.read_parquet(base.ACCOUNT/f'accounts/history_QREG_{phase}/daily.parquet')
        residual=residual_parts(FULL/'main',phase);r=regime.reindex(account.index)
        assert r.known.all() and account.index.equals(residual.index)
        b=benchmark.reindex(account.index);assert b.notna().all()
        for code,label in [(-1,'全部'),*REGIMES.items()]:
            mask=pd.Series(True,index=r.index) if code==-1 else r.regime.eq(code).astype(bool)
            target=conditional_mean_hac(account.net_return,mask)
            active=conditional_mean_hac(account.net_return-b,mask)
            remaining=conditional_mean_hac(residual,mask)
            rows.append(dict(offset=phase,regime=label,days=target['observations'],
                strategy=target['annual_conditional_arithmetic_mean'],benchmark=float(b[mask].mean()*252),
                active=active['annual_conditional_arithmetic_mean'],net_residual=remaining['annual_conditional_arithmetic_mean'],
                residual_ci_low=remaining['ci_low'],residual_ci_high=remaining['ci_high']))
            for year,g in account.groupby(account.index.year):
                m=mask.reindex(g.index)
                if m.sum()==0:continue
                by_year.append(dict(offset=phase,year=year,regime=label,days=int(m.sum()),
                    strategy=float(g.loc[m,'net_return'].mean()*252),
                    benchmark=float(b.reindex(g.index)[m].mean()*252),
                    net_residual=float(residual.reindex(g.index)[m].mean()*252)))
        # Realized quarterly diagnostics are explicitly ex post and use all quarters.
        for quarter,g in account.groupby(account.index.to_period('Q')):
            ff=factors.reindex(g.index);br=float(ff[bank].sum());vr=float(((ff.book_to_price+ff.earnings_yield)/2).sum())
            net=g.net_return;bench=b.reindex(g.index);prior=(1+net).cumprod().shift(1,fill_value=1)
            quarters.append(dict(offset=phase,quarter=str(quarter),days=len(g),bank_factor_sum=br,value_factor_sum=vr,
                realized_both_weak=bool(br<=0 and vr<=0),strategy_return=float((1+net).prod()-1),
                benchmark_return=float((1+bench).prod()-1),
                linked_net_residual=float((residual.reindex(g.index)*prior).sum()),partial_quarter=str(quarter)=='2026Q4'))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'regime_per_phase.csv',index=False)
    summary=frame.groupby('regime',sort=False).agg(days=('days','first'),strategy=('strategy','mean'),
        benchmark=('benchmark','mean'),active=('active','mean'),net_residual=('net_residual','mean'),
        phase_min_residual_ci=('residual_ci_low','min'),phase_max_residual_ci=('residual_ci_high','max'))
    summary.to_csv(OUT/'regime_summary.csv')
    pd.DataFrame(by_year).to_csv(OUT/'regime_by_year.csv',index=False)
    pd.DataFrame(quarters).to_csv(OUT/'quarterly_realized_regimes.csv',index=False)
    regime.loc[base.START:base.END].to_csv(OUT/'daily_regime_labels.csv')
    return summary


def forward():
    # Copy immutable observations, never call a broker or relabel overnight records.
    first=base.read(base.ROOT/'reports/paper_futu/s6_forward_v1/protocol.json')['first_execution_date']
    today=str(china_now().date());dates=pd.DatetimeIndex(pd.to_datetime((HISTORICAL_PROVIDER/'calendars/day_future.txt').read_text().splitlines()))
    eligible=dates[(dates>=first)&(dates<=today)]
    if (china_now().hour,china_now().minute)<(15,35):eligible=eligible[eligible<today]
    rows=[];missing=[];sources={}
    for date in eligible:
        day=str(date.date());p=STATE_DIR/f'nav_saved_{day}.json'
        if not p.exists():missing.append(day);continue
        x=close_observation(p,day);dest=OUT/'observed_inputs'/p.name;dest.parent.mkdir(exist_ok=True)
        if dest.exists() and base.sha(dest)!=base.sha(p):raise ValueError('Closing observation changed')
        if not dest.exists():shutil.copy2(p,dest)
        sources[str(dest)]=base.sha(dest)
        rows.append(dict(datetime=day,assets=x['assets'],cash=x['cash'],holdings=x['holdings'],observed_at=x['time']))
    frame=pd.DataFrame(rows).set_index('datetime').sort_index();frame.index=pd.to_datetime(frame.index)
    frame['return_between_observations']=frame.assets.pct_change(fill_method=None)
    frame.to_csv(OUT/'observed_forward_nav.csv')
    benchmark_path=base.ROOT/'data/interim/s6_forward_benchmark_20261026/H00300.json'
    benchmark=official_frame(base.read(benchmark_path),'H00300');sources[str(benchmark_path)]=base.sha(benchmark_path)
    matches=frame.index.intersection(benchmark.index);comparison=None
    if len(matches)>=2:
        start,end=matches[0],matches[-1]
        strategy=float(frame.loc[end,'assets']/frame.loc[start,'assets']-1)
        bench=float(benchmark.loc[end,'close']/benchmark.loc[start,'close']-1)
        comparison=dict(start=str(start.date()),end=str(end.date()),strategy_return=strategy,benchmark_return=bench,
            geometric_excess=(1+strategy)/(1+bench)-1)
    result=dict(first_planned_s6_execution=first,first_valid_close=str(frame.index[0].date()),last_valid_close=str(frame.index[-1].date()),
        closing_observations=len(frame),observed_intervals=max(0,len(frame)-1),missing_closing_dates=missing,
        return_between_valid_endpoints=float(frame.assets.iloc[-1]/frame.assets.iloc[0]-1),
        latest_assets=float(frame.assets.iloc[-1]),latest_holdings=int(frame.holdings.iloc[-1]),
        benchmark_available_through=str(benchmark.index.max().date()),matched_benchmark_comparison=comparison,
        sources=sources,annualized_or_sharpe_estimated=False,
        limitation='Short actual-paper observation with execution/data repairs and an October20 correction; no clean uninterrupted frozen-code independent test. Missing closes not reconstructed.')
    base.save(OUT/'forward_summary.json',result);return result,frame


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from filelock import FileLock
    OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(OUT/'diagnostic.lock'),timeout=0):
        audit();summary=historical();observed,nav=forward()
        def pct(x):return f'{x:.2%}'
        def table(headers,rows):return '<table><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</table>'
        rows=[[label,int(r.days),pct(r.strategy),pct(r.benchmark),pct(r.active),pct(r.net_residual)] for label,r in summary.iterrows()]
        quarters=pd.read_csv(OUT/'quarterly_realized_regimes.csv').groupby('quarter').agg(
            bank_factor_sum=('bank_factor_sum','first'),value_factor_sum=('value_factor_sum','first'),
            both_weak=('realized_both_weak','first'),strategy_return=('strategy_return','mean'),
            benchmark_return=('benchmark_return','first'),net_residual=('linked_net_residual','mean'),partial=('partial_quarter','first'))
        qrows=[[q,pct(g.bank_factor_sum),pct(g.value_factor_sum),bool(g.both_weak),pct(g.strategy_return),pct(g.benchmark_return),pct(g.net_residual)] for q,g in quarters.iterrows()]
        text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>S6弱风格诊断与前瞻观察</title><style>body{{max-width:1250px;margin:32px auto;padding:0 22px;font:16px/1.7 system-ui;color:#17354b}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d2e0eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf4f8}}</style>
<h1>S6：银行与价值走弱时的表现</h1><p>冻结历史2022-01-04至2026-10-13，沿用S6五条原账户，不改选股、仓位、成本或执行。主定义：前20个完整交易日的银行纯行业因子累计收益、B/P与E/P因子平均累计收益，分别不大于零记为弱，滞后一个交易日。窗口固定，不挑参数。银行因子是控制市场与风格后的行业收益，不等于银行指数涨跌。</p>
<h2>前期风格状态下的随后收益</h2>{table(['前20日状态','交易日数','策略条件年化均值','H00300条件年化均值','主动收益条件均值','净残差条件均值'],rows)}
<p>年化均值是被归入该状态的日收益平均×252，不是连续策略复合年化。状态可能在当日反转。没有为不连续筛选日拼接一条虚假的可交易净值。HAC区间保留原始交易日日历和状态持续性，逐起点见CSV；历史区间已参与研究，结果不作为独立测试。</p>
<h2>全部季度：当季实际风格与账户表现（事后诊断）</h2>{table(['季度','银行因子累计','价值因子累计','当季双弱','实际策略收益','H00300收益','净残差贡献'],qrows)}
<p>该表使用季度结束后才知道的风格结果，只用于解释，不当作当季事前交易信号。2026Q4只有截至10月13日的部分数据。</p>
<h2>真实前瞻观察记录</h2><p>计划S6切换日{observed['first_planned_s6_execution']}；有效收盘观察为{observed['first_valid_close']}至{observed['last_valid_close']}，共{observed['closing_observations']}个点、{observed['observed_intervals']}个间隔，区间收益{pct(observed['return_between_valid_endpoints'])}。缺少收盘记录：{html.escape(str(observed['missing_closing_dates']))}；不补造数据，也不把午夜查询改称前一日收盘。</p>
{table(['实际收盘日','账户资产','现金','持股数'],[[str(day.date()),f'{r.assets:,.2f}',f'{r.cash:,.2f}',int(r.holdings)] for day,r in nav.iterrows()])}
<p>官方H00300目前取得至{observed['benchmark_available_through']}。共同日期对照：{html.escape(str(observed['matched_benchmark_comparison']))}。未发布的基准日期不补零或用价格指数冒充全收益指数。样本不足，不算年化、夏普或Alpha显著性；这段观察含执行/数据修复及10月20日一次纠偏，不是完全无改动的干净前瞻测试。</p>
<p>本报告不触发下单，不修改部署或调度。行业历史发布时间与股本修订档案仍有限，风格结果受所选风险模型定义影响。</p></html>'''
        (OUT/'report.html').write_text(text,encoding='utf-8')
        p=base.read(OUT/'protocol.json')
        for name,h in p['sources'].items():assert base.sha(name)==h
        base.save(OUT/'completion.json',dict(status='complete',historical_diagnostic_only=True,orders_submitted=0,
            independent_forward_test_sufficient=False,protocol_sha256=base.sha(OUT/'protocol.json'),
            artifacts={str(p):base.sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['completion.json','diagnostic.lock']}))
        print(summary.to_string());print(observed)


if __name__=='__main__':main()
