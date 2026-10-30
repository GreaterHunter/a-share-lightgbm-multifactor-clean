"""Extend sealed S6 Barra v1 with independently bound industry/capital metadata.

Main case: 93-calendar-day delayed industry changes and sqrt-cap WLS.
Controls: full factors with equal weighting; one-day industry timing sensitivity.
No LightGBM re-training or broker actions.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import html
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from paper_futu import research_barra as base
from src import barra_risk as core
from paper_futu.fetch_barra_metadata import OUT as METADATA

ROOT=base.ROOT
CASES={'main':dict(lag=93,equal=False),'equal':dict(lag=93,equal=True),'lag1':dict(lag=1,equal=False)}
OUT=ROOT/'reports/experiments/barra_s6_industry_size_asof_20261013'
REFERENCE=ROOT/'reports/experiments/barra_s6_v1_asof_20261013'


def hac_mean(values,lags=20):
    x=np.asarray(values,dtype=float)
    if len(x)<=lags+2 or not np.isfinite(x).all():raise ValueError('Insufficient finite mean observations')
    mean=float(x.mean());u=x-mean;n=len(x)
    omega=float(u@u/n)
    for k in range(1,lags+1):omega+=2*(1-k/(lags+1))*float(u[k:]@u[:-k]/n)
    se=float(np.sqrt(max(omega,0)/n))
    return dict(annual_arithmetic_mean=mean*252,ci_low=(mean-1.96*se)*252,ci_high=(mean+1.96*se)*252,
        nominal_p=math.erfc(abs(mean/se)/np.sqrt(2)) if se>0 else None,observations=n,hac_lags=lags)


def metadata_verify():
    m=base.read(METADATA/'manifest.json')
    for path,h in {**m['artifacts'],**m['sources'],**m['implementation']}.items():
        if base.sha(path)!=h:raise ValueError('Metadata provenance changed: '+path)
    return m


def bind(out,case):
    metadata_verify()
    p=dict(case=case,settings=CASES[case],metadata_manifest_sha256=base.sha(METADATA/'manifest.json'),
        base_protocol_sha256=base.sha(out/'protocol.json'),
        implementation={str(Path(__file__)):base.sha(__file__)},
        nonlinear_size='Cube standardized log A-share capitalization; weighted orthogonalization to intercept and size, restandardize',
        historical_publication_archive_complete=False,
        timing='Main industry timing delayed 93 calendar days; 1-day sensitivity. Historical change date is not a verified first-publication timestamp.',
        equal_case='Forces equal cross-section standardization, equal regression loss, count-weighted industry constraint')
    path=out/'full_binding.json'
    if path.exists() and base.read(path)!=p:
        old=base.read(path);before=dict(old);after=dict(p)
        before.pop('implementation');after.pop('implementation')
        if before!=after or (out/'factors_complete.json').exists() or (out/'analysis_complete.json').exists():
            raise ValueError('Full-case binding differs after fitted results')
        # Only the failed, not-yet-applied pre-fit size transform was repaired.
        if not base.stage_done(out,'data') or (out/'nonlinear_size_transform.json').exists():
            raise ValueError('Pre-fit cache compatibility not established')
        archive=out/('pre_fit_binding_'+base.sha(path)+'.json')
        if not archive.exists():base.save(archive,old)
        base.save(out/'binding_repair_audit.json',dict(old=old['implementation'],new=p['implementation'],
            compatible_data_cache=True,no_factor_fit_reused=True,reason='Preserve orthogonality after winsorization'))
    base.save(path,p)


def fixed_nonlinear_size(out,equal):
    marker=out/'nonlinear_size_transform.json';path=out/'exposures.parquet'
    if marker.exists():
        assert base.sha(path)==base.read(marker)['output_sha256'];return
    frame=pd.read_parquet(path);parts=[];checks=[]
    for day,g in frame.groupby(level='datetime',sort=True):
        w=g.capital if not equal and g.capital.notna().all() and g.capital.gt(0).all() else pd.Series(1.,index=g.index)
        x=g['size'].to_numpy();cubic,_=core.standardize(pd.Series(x**3,index=g.index),w)
        cubic=cubic.to_numpy();design=np.column_stack([np.ones(len(x)),x]);root=np.sqrt(w/w.mean()).to_numpy()
        b=np.linalg.lstsq(design*root[:,None],cubic*root,rcond=None)[0]
        residual=cubic-design@b;residual-=np.average(residual,weights=w)
        sd=np.sqrt(np.average(residual**2,weights=w))
        # Clipping after orthogonalization would reintroduce size correlation.
        g=g.copy();g['nonlinear_size']=residual/sd if sd>1e-12 else 0.
        cross=float(np.average(g['size']*g.nonlinear_size,weights=w))
        if abs(cross)>1e-8:raise ValueError('Nonlinear size not orthogonal to size')
        parts.append(g);checks.append(abs(cross))
    old=base.sha(path);pd.concat(parts).sort_index().to_parquet(path)
    previous=base.read(out/'data_complete.json')
    base.save(out/'data_before_size_transform.json',previous)
    previous['artifacts'][str(path)]=base.sha(path);base.save(out/'data_complete.json',previous)
    base.save(marker,dict(input_sha256=old,output_sha256=base.sha(path),maximum_weighted_cross_product=max(checks)))


def run_case(case):
    settings=CASES[case];out=OUT/case;out.mkdir(parents=True,exist_ok=True)
    base.audit(out,METADATA/f"industry_lag{settings['lag']}.parquet",METADATA/'capital_pit.parquet',True)
    bind(out,case)
    def standardize(values,weights=None,minimum=30):
        return core.standardize(values,None if settings['equal'] else weights,minimum)
    def fit(exposures,returns,regression_weights=None,industry_mass=None):
        if settings['equal']:
            regression_weights=None
            industry_mass=exposures[[c for c in exposures if c.startswith('industry:')]].sum()
        return core.fit_cross_section(exposures,returns,regression_weights,industry_mass)
    with ExitStack() as stack:
        stack.enter_context(patch.object(base,'standardize',standardize))
        stack.enter_context(patch.object(base,'fit_cross_section',fit))
        base.prepare(out);fixed_nonlinear_size(out,settings['equal'])
        base.estimate(out);base.analyze(out);base.report(out)
    bind(out,case);base.verify_sources(out)
    base.save(out/'full_complete.json',dict(status='complete',full_binding_sha256=base.sha(out/'full_binding.json'),
        base_completion_sha256=base.sha(out/'completion.json'),nonlinear_transform_sha256=base.sha(out/'nonlinear_size_transform.json'),
        historical_effective_dates_used=True,original_publication_times_fully_verified=False))


def residual_parts(directory,phase):
    comp=pd.read_parquet(directory/f'attribution_daily_{phase}.parquet').set_index('datetime')
    return comp.specific+comp.execution_timing+comp.fees


def report():
    metadata=metadata_verify()
    for case in CASES:
        directory=OUT/case;bind(directory,case)
        assert base.read(directory/'full_complete.json')['status']=='complete'
        for stage in ['data','factors','analysis','report']:assert base.stage_done(directory,stage)
    base.verify_sources(REFERENCE)
    rows=[];yearly=[];linked=[];deltas=[];definitions={'reference':REFERENCE,**{c:OUT/c for c in CASES}}
    for label,directory in definitions.items():
        for phase in range(5):
            residual=residual_parts(directory,phase)
            account=pd.read_parquet(base.ACCOUNT/f'accounts/history_QREG_{phase}/daily.parquet')
            assert residual.index.equals(account.index)
            for lag in [20,60]:rows.append(dict(case=label,offset=phase,**hac_mean(residual,lag)))
            for year,g in residual.groupby(residual.index.year):
                yearly.append(dict(case=label,offset=phase,year=year,annual_arithmetic_mean=float(g.mean()*252),
                    diagnostic_compound=float((1+g).prod()-1),days=len(g)))
            for group,g in [('all',residual),*[(str(y),g) for y,g in residual.groupby(residual.index.year)]]:
                actual=account.net_return.reindex(g.index);prior=(1+actual).cumprod().shift(1,fill_value=1)
                linked.append(dict(case=label,offset=phase,period=group,linked_net_residual_contribution=float((prior*g).sum())))
            if label!='reference':
                reference=residual_parts(REFERENCE,phase)
                assert residual.index.equals(reference.index)
                deltas.append(dict(case=label,offset=phase,**hac_mean(residual-reference,20)))
    means=pd.DataFrame(rows);years=pd.DataFrame(yearly);links=pd.DataFrame(linked)
    means.to_csv(OUT/'residual_mean_hac.csv',index=False);years.to_csv(OUT/'residual_yearly.csv',index=False)
    links.to_csv(OUT/'residual_linked_contributions.csv',index=False);pd.DataFrame(deltas).to_csv(OUT/'paired_residual_differences.csv',index=False)
    summary=[]
    for label,directory in definitions.items():
        p=means.loc[means.case.eq(label)&means.hac_lags.eq(20)]
        v=base.read(directory/'summary.json');profile=base.read(directory/'data_profile.json')
        summary.append(dict(case=label,mean_net_residual_annual_arithmetic=float(p.annual_arithmetic_mean.mean()),
            phase_min_ci_low=float(p.ci_low.min()),phase_max_ci_high=float(p.ci_high.max()),
            all_phases_ci_positive=bool(p.ci_low.gt(0).all()),mean_linked_net_residual_contribution=float(links.loc[links.case.eq(label)&links.period.eq('all'),'linked_net_residual_contribution'].mean()),
            predicted_volatility=v['mean_forecast_annual_volatility'],calibration_rms=v['mean_calibration_rms'],
            industry_coverage=profile['industry_observation_coverage'],capital_coverage=profile['capital_observation_coverage']))
    result=dict(status='complete',as_of=base.END,summary=summary,metadata_manifest_sha256=base.sha(METADATA/'manifest.json'),
        remaining_return_definition='net account return minus modeled market/industry/style contribution and stock risk-free component; includes specific, execution timing and actual strategy fees',
        comparison_caution='Main vs equal isolates weighting/standardization effects; equal vs old reference adds industry and size under equal-weight rules. Industry lag1 vs main checks timing.',
        residual_not_tradable_alpha=True,original_industry_publication_archive_incomplete=True,
        benchmark_historical_weights_not_obtained=True,selection_bias_not_corrected=True)
    base.save(OUT/'comparison.json',result)
    names=dict(reference='旧版：市场＋10风格/等权',equal='加入行业与规模/等权',main='行业与规模/市值加权/93日延迟',lag1='行业与规模/市值加权/1日延迟')
    def table(headers,values):return '<table><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in values)+'</table>'
    rows=[]
    for r in summary:
        rows.append([names[r['case']],f"{r['mean_net_residual_annual_arithmetic']:.2%}",
            f"{r['phase_min_ci_low']:.2%} 至 {r['phase_max_ci_high']:.2%}",f"{r['mean_linked_net_residual_contribution']:.2%}",
            f"{r['industry_coverage']:.2%}",f"{r['capital_coverage']:.2%}"])
    yy=years.groupby(['case','year']).annual_arithmetic_mean.mean().unstack('case')
    yrows=[[str(year),*[f"{g[label]:.2%}" for label in ['reference','equal','main','lag1']]] for year,g in yy.iterrows()]
    main_attr=pd.read_csv(OUT/'main/return_attribution.csv').groupby(['period','factor']).linked_contribution.mean().unstack()
    arows=[]
    for period,g in main_attr.iterrows():
        style_names=base.STYLES+['size','nonlinear_size']
        arows.append([period,f'{g.country:.2%}',f"{g.filter(like='industry:').sum():.2%}",f"{g.reindex(style_names).sum():.2%}",
            f'{g.specific:.2%}',f'{g.execution_timing:.2%}',f'{g.fees:.2%}',f'{g.stock_rf:.2%}',f'{g.sum():.2%}'])
    # Frozen input and account assertions are repeated after all experiments.
    for directory in definitions.values():base.verify_sources(directory)
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>S6行业与规模控制后的剩余收益</title><style>body{{max-width:1400px;margin:30px auto;padding:0 22px;font:16px/1.7 system-ui;color:#15354b}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d8e3eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf4f8}}.note{{background:#fff1d6;padding:18px}}</style>
<h1>S6：补齐历史行业与市值后的收益归因</h1><p>固定S6模型、账户、费用与2022-01-04至2026-10-13区间。新增历史行业、规模及非线性规模；不修改LightGBM和模拟盘。</p>
<p class="note">行业来自巨潮历史申万分类变更记录，按分类版本生效日及历史变更日向后对齐。它没有完整首次发布时间档案，主研究延迟93个自然日，另做1日延迟敏感性；不是已认证无修订的严格PIT档案。市值采用公开公告后才可用的A股股数×每日未复权A股收盘，不把H/B股按A股价估值。没有取得历史沪深300权重，暂不计算声称相对该指数的主动因子风险。</p>
<h2>控制后剩多少</h2>{table(['口径','净残差年化算术均值','各起点95%HAC区间最小下界至最大上界','全期净残差贡献','历史行业覆盖','历史市值覆盖'],rows)}
<p>净残差＝账户净收益−模型共同因子贡献−股票无风险部分，等于个股残差＋交易时点与仓位变化−实际交易费用。这是风险调整的描述性残差，不是可直接交易的对冲收益，不包含另行实施对冲的费用。年化均值是日均值×252，非复合年化。全期贡献使用实际账户净值复利衔接，不是年化收益。</p>
<p>HAC使用20阶；60阶结果另存CSV。区间按每个起点分别估计，表中只是所有区间的包络，不能当作合并后的95%置信区间；五条账户高度相关，也未修正多年历史参数搜索的选择偏差。正残差或名义显著都不能宣布已证明纯Alpha。</p>
<h2>逐年净残差的年化算术均值</h2>{table(['年份','旧版','行业规模/等权','行业规模/93日延迟','行业规模/1日延迟'],yrows)}<p>2026为截至10月13日的日均值年化，不是完整年度回报；各年的实际累计残差诊断及账户复利衔接贡献在CSV中分开列出。</p>
<h2>主研究实际账户收益归因</h2>{table(['年份','市场','行业','风格含规模','个股残差','交易时点','费用','股票无风险部分','账户净收益'],arows)}
<p>各项为该区间累计净收益的百分点贡献，逐起点复算后均值，合计等于账户回报。因子收益会随回归约束和权重变化，不将不同模型的单项系数当作天然可比的经济真值。</p>
<h2>数据、风险报告与复现</h2><p>完整来源响应与哈希在本地metadata manifest中，股本通知日期必须不越界，行业并未用当前归属回填整个历史。非线性规模由标准化对数市值的三次项构造，按本模型权重对规模和截距正交。各模型参数固定，没有根据S6历史收益选新冠军。</p>
<p><a href="main/report.html">主研究风险详情</a> · <a href="equal/report.html">等权公平对照</a> · <a href="lag1/report.html">1日时点敏感性</a>。旧v1报告保留原样。所有归因仍受财务修订档案、历史行业记录与股本源修订不完整、残差相关性及选优偏差影响。</p></html>'''
    (OUT/'report.html').write_text(body,encoding='utf-8')
    base.save(OUT/'completion.json',dict(status='complete',sources_unchanged=True,comparison_sha256=base.sha(OUT/'comparison.json'),
        report_sha256=base.sha(OUT/'report.html'),cases={c:base.sha(OUT/c/'full_complete.json') for c in CASES}))
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=list(CASES));p.add_argument('--stage',choices=['all','case','report'],default='all');args=p.parse_args()
    from filelock import FileLock
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='case':run_case(args.case);return
    with FileLock(str(OUT/'controller.lock'),timeout=0):
        if args.stage=='all':
            for case in CASES:
                if (OUT/case/'full_complete.json').exists():continue
                base.save(OUT/'status.json',dict(stage=case,state='running',pid=os.getpid()))
                with (OUT/f'{case}.log').open('a',encoding='utf-8') as log:
                    r=subprocess.run([sys.executable,'-X','utf8','-u','-m','paper_futu.research_barra_full','--stage','case','--case',case],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if r.returncode:
                    base.save(OUT/'status.json',dict(stage=case,state='failed',exit_code=r.returncode));raise SystemExit(r.returncode)
                print('Finished',case,flush=True)
        report();base.save(OUT/'status.json',dict(state='complete'))


if __name__=='__main__':main()
