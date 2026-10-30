"""Readable supplemental findings; preserve each hash-bound risk case."""
from __future__ import annotations
import shutil
import numpy as np
import pandas as pd
from paper_futu.research_barra_full import OUT,REFERENCE,CASES,metadata_verify,bind
from paper_futu import research_barra as base


def main():
    comparison=base.read(OUT/'comparison.json');assert comparison['status']=='complete'
    metadata_verify()
    for case in CASES:
        bind(OUT/case,case);base.verify_sources(OUT/case)
        for stage in ['data','factors','analysis','report']:assert base.stage_done(OUT/case,stage)
        audit=pd.read_csv(OUT/case/'regression_audit.csv')
        # The sealed base runner records pre-override cap availability. The
        # equal control deliberately ignores that vector; expose actual weights.
        audit=audit.rename(columns={'cap_weighted':'capital_vector_available'})
        audit['effective_cap_weighted']=audit.capital_vector_available & (not CASES[case]['equal'])
        audit['equal_override']=CASES[case]['equal']
        audit.to_csv(OUT/case/'effective_regression_weighting.csv',index=False)
    exposure=pd.read_parquet(OUT/'main/portfolio_exposures.parquet')
    bank=[c for c in exposure if c.startswith('industry:') and c.endswith('\u94f6\u884c')][0]
    bank_year=exposure.groupby(exposure.datetime.dt.year)[bank].agg(['mean','max','min'])
    bank_year.to_csv(OUT/'bank_exposure_by_year.csv')
    attr=pd.read_csv(OUT/'main/return_attribution.csv').groupby(['period','factor']).linked_contribution.mean().unstack()
    main=[s for s in comparison['summary'] if s['case']=='main'][0]
    reference=[s for s in comparison['summary'] if s['case']=='reference'][0]
    findings=dict(as_of=base.END,mean_bank_weight_by_year=bank_year['mean'].to_dict(),
        latest_bank_weight=float(exposure.loc[exposure.datetime.eq(base.END),bank].mean()),
        latest_bank_observation='mean of five historical phase accounts at cutoff, not current live account',
        mean_net_residual_annual=main['mean_net_residual_annual_arithmetic'],
        residual_reduction_vs_reference=1-main['mean_net_residual_annual_arithmetic']/reference['mean_net_residual_annual_arithmetic'],
        no_positive_lower_ci_in_all_phases=not main['all_phases_ci_positive'],
        independent_alpha_proven=False,linked_residual_by_year=(attr.specific+attr.execution_timing+attr.fees).to_dict())
    means=pd.read_csv(OUT/'residual_mean_hac.csv');m=means.loc[means.case.eq('main')&means.hac_lags.eq(20)]
    findings['all_main_phase_intervals_include_zero']=bool((m.ci_low.le(0)&m.ci_high.ge(0)).all())
    base.save(OUT/'findings.json',findings)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,1,figsize=(12,12),layout='constrained')
    order=['reference','equal','main','lag1'];labels=['原风格模型','行业规模/等权','行业规模/市值权重','1日延迟敏感性']
    values=[next(s for s in comparison['summary'] if s['case']==c)['mean_net_residual_annual_arithmetic']*100 for c in order]
    bars=axes[0].bar(labels,values,color=['#8796a2','#2776a2','#1b9880','#6d8bb7'])
    axes[0].bar_label(bars,fmt='%.2f%%',padding=3);axes[0].set_ylim(0,max(values)*1.25)
    axes[0].set_title('S6：扣除共同因子和实际费用后的净残差年化算术均值');axes[0].set_ylabel('%/年（不是复合年化）')
    years=[str(x) for x in range(2022,2027)];positions=np.arange(5)
    total=attr.loc[years].sum(axis=1)*100
    residual=(attr.loc[years,'specific']+attr.loc[years,'execution_timing']+attr.loc[years,'fees'])*100
    b1=axes[1].bar(positions-.19,total,.38,label='原账户实际净收益',color='#397ba2')
    b2=axes[1].bar(positions+.19,residual,.38,label='净残差的复利衔接贡献',color='#cf8b3c')
    axes[1].bar_label(b1,fmt='%.1f',padding=3);axes[1].bar_label(b2,fmt='%.1f',padding=3)
    axes[1].set_xticks(positions,['2022','2023','2024','2025','2026截至10月13日']);axes[1].axhline(0,color='gray',lw=.8)
    axes[1].set_ylabel('收益百分点');axes[1].legend();axes[1].set_title('主研究：原收益与行业、规模等控制后的净残差贡献')
    bank_daily=exposure.groupby('datetime')[bank].mean()*100
    axes[2].plot(bank_daily.index,bank_daily,color='#287d78');axes[2].set_ylim(0,100);axes[2].set_ylabel('占账户净值 %')
    axes[2].set_title('银行持仓比例：五个历史起点均值，有效风险快照日');axes[2].grid(alpha=.2)
    fig.savefig(OUT/'findings.png',dpi=150);plt.close(fig)
    report=OUT/'report.html';backup=OUT/'report_before_findings.html'
    if not backup.exists():shutil.copy2(report,backup)
    body=backup.read_text(encoding='utf-8')
    note=f'''<h2>主要发现</h2><p>主研究净残差年化算术均值{main['mean_net_residual_annual_arithmetic']:.2%}，原版为{reference['mean_net_residual_annual_arithmetic']:.2%}。所有五个主研究起点的95%HAC区间均包含零。原账户收益保持原样，这些结果支持大部分表面Alpha与共同风险暴露有关；仍不能证明真实Alpha恰为零。</p>
<p>截至2026年10月13日，五个历史账户平均银行仓位为{findings['latest_bank_weight']:.2%}；不是今天的模拟盘仓位。2024、2025、2026年有效风险快照上的平均银行仓位分别为{bank_year.loc[2024,'mean']:.2%}、{bank_year.loc[2025,'mean']:.2%}、{bank_year.loc[2026,'mean']:.2%}。缺少历史指数权重，不能把这些绝对仓位称为相对沪深300的超配比例。</p><img src="findings.png" alt="净残差与银行仓位" style="max-width:100%"><p>实际回归权重审计见各子目录effective_regression_weighting.csv。原引擎regression_audit.csv中的cap_weighted为覆盖向量可用性标记；equal对照会明确覆盖为等权，不能按原标记误解其实际回归权重。</p>'''
    body=body.replace('<h2>控制后剩多少</h2>',note+'<h2>控制后剩多少</h2>')
    report.write_text(body,encoding='utf-8')
    original=OUT/'completion_before_findings.json'
    if not original.exists():shutil.copy2(OUT/'completion.json',original)
    completed=base.read(original);completed['report_sha256']=base.sha(report)
    completed['supplement']={str(p):base.sha(p) for p in [OUT/'findings.json',OUT/'findings.png',OUT/'bank_exposure_by_year.csv',base.ROOT/'scripts/report_barra_full_findings.py']}
    completed['supplement'].update({str(OUT/c/'effective_regression_weighting.csv'):base.sha(OUT/c/'effective_regression_weighting.csv') for c in CASES})
    base.save(OUT/'completion.json',completed)
    print(findings)


if __name__=='__main__':main()
