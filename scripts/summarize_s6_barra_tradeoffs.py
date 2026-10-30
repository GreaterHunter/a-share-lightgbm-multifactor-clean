"""Descriptive paired diagnostics; no new fits, portfolio rules or simulations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from paper_futu.research_s6_barra_overlay import OUT, CASES, COSTS, read, save, digest, verify


def main():
    verify();pairs=[];yearly=[];all_rows=[]
    for cost in COSTS:
        for phase in range(5):
            accounts={c:read(OUT/f'accounts/history_{c}_f{cost}_{phase}/complete.json') for c in CASES}
            for case in CASES:
                root=OUT/f'accounts/history_{case}_f{cost}_{phase}'
                d=pd.read_parquet(root/'daily.parquet');risk=pd.read_parquet(root/'realized_holdings_risk.parquet')
                log=read(root/'selection.json');optimized=[r for r in log if r['status']=='optimized']
                row=dict(case=case,cost_bp=cost,offset=phase,
                    realized_annual_volatility=float(d.net_return.std()*np.sqrt(252)),
                    mean_equity_weight=float(risk.stock_weight.mean()),
                    mean_predicted_annual_volatility=float(risk.predicted_volatility.mean()),
                    mean_score_delta_to_same_day_original=(float(np.mean([r['target_score_relative_original'] for r in optimized])) if optimized else 0.),
                    optimized_rebalances=len(optimized),fallback_rebalances=sum(r['status'].startswith('fallback') for r in log),
                    **accounts[case]['periods']['all'])
                all_rows.append(row)
                for year in range(2022,2027):
                    m=accounts[case]['periods'][str(year)]
                    yearly.append(dict(case=case,cost_bp=cost,offset=phase,year=year,
                        total_return=m['total_return'],max_drawdown=m['max_drawdown'],
                        bank_weight=m['mean_bank_weight'],fees=m['fees'],
                        predicted_volatility=m['mean_predicted_volatility']))
                if case!='R0':
                    a=accounts[case]['periods']['all'];b=accounts['R0']['periods']['all']
                    pairs.append(dict(case=case,cost_bp=cost,offset=phase,
                        cagr_difference=a['cagr']-b['cagr'],drawdown_difference=a['max_drawdown']-b['max_drawdown'],
                        bank_weight_difference=a['mean_bank_weight']-b['mean_bank_weight'],fee_difference=a['fees']-b['fees']))
    pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_risk_details.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_rows).to_csv(OUT/'phase_diagnostics.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame(all_rows).groupby(['case','cost_bp']).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'risk_tradeoff_summary.csv',index=False,encoding='utf-8-sig')
    save(OUT/'additional_diagnostics.json',dict(status='complete',paired_count=len(pairs),
        summary=summary.to_dict('records'),
        interpretation='Five phases are correlated; paired differences are descriptive, not five independent experiments. Score units are percentile, not return.',
        script_sha256=digest(__file__),files={str(OUT/f):digest(OUT/f) for f in
            ('paired_differences.csv','yearly_risk_details.csv','phase_diagnostics.csv','risk_tradeoff_summary.csv')}))
    print(summary.loc[summary.cost_bp.eq(0),['case','cagr','max_drawdown','realized_annual_volatility',
          'mean_predicted_annual_volatility','mean_bank_weight','fees']].to_string(index=False),flush=True)


if __name__=='__main__':main()
