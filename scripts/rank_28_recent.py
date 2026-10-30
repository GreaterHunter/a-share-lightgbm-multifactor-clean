"""Descriptive 2025/2026 ranking, no fits, backtests, or deployment changes."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/experiments/top10_rank80_two_thresholds_asof_20261013'
OUT=ROOT/'reports/analysis/top28_recent_2025_2026'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    done=load(SOURCE/'completion.json');assert done['status']=='complete' and sha(SOURCE/'report_manifest.json')==done['manifest_sha256']
    for path,h in load(SOURCE/'report_manifest.json')['artifacts'].items():assert sha(path)==h,path
    years=pd.read_csv(SOURCE/'yearly.csv');account=pd.read_csv(SOURCE/'comparison.csv')
    cols=['total_return','max_drawdown','fees_over_opening_nav']
    data=years[years.year.isin([2025,2026])].pivot(index='name',columns='year',values=cols)
    assert len(data)==28 and data.notna().all().all()
    data.columns=[f'{metric}_{year}' for metric,year in data.columns]
    rankcols=[]
    for metric in ('total_return','max_drawdown'):
        for year in (2025,2026):
            col=f'{metric}_{year}';rcol=f'rank_{col}'
            data[rcol]=data[col].rank(ascending=metric=='max_drawdown',method='average');rankcols.append(rcol)
    data['mean_metric_rank']=data[rankcols].mean(axis=1)
    data['worst_recent_period_drawdown']=data[['max_drawdown_2025','max_drawdown_2026']].max(axis=1)
    data['mean_fee_rate_descriptive']=data[['fees_over_opening_nav_2025','fees_over_opening_nav_2026']].mean(axis=1)
    data['dominated_by']=''
    for name,row in data.iterrows():
        others=data.drop(index=name)
        dominates=(others.total_return_2025>=row.total_return_2025)&(others.total_return_2026>=row.total_return_2026)&(others.max_drawdown_2025<=row.max_drawdown_2025)&(others.max_drawdown_2026<=row.max_drawdown_2026)
        strictly=(others.total_return_2025>row.total_return_2025)|(others.total_return_2026>row.total_return_2026)|(others.max_drawdown_2025<row.max_drawdown_2025)|(others.max_drawdown_2026<row.max_drawdown_2026)
        data.loc[name,'dominated_by']=';'.join(others.index[dominates&strictly])
    past=account[account.period.eq('2022_2025')].set_index('name')
    current=account[account.period.eq('2026')].set_index('name')
    data=data.join(past[['rank_exit','strong_rank','mode','threshold','cagr','max_drawdown']].rename(columns={'cagr':'cagr_2022_2025','max_drawdown':'drawdown_2022_2025'}))
    data=data.join(current[['closed','fees','mean_holding_days']].rename(columns={c:c+'_2026' for c in ('closed','fees','mean_holding_days')}))
    data=data.sort_values(['mean_metric_rank','worst_recent_period_drawdown','mean_fee_rate_descriptive'],kind='stable').reset_index()
    data.insert(0,'priority',range(1,29));data['selected']=data.priority.le(5)
    assert data.head(5).dominated_by.eq('').all()
    # Scores are recomputed from exact values, not rounded presentation cells.
    np.testing.assert_allclose(data.mean_metric_rank,data[rankcols].sum(axis=1)/4,rtol=0,atol=0)
    OUT.mkdir(parents=True,exist_ok=True);data.to_csv(OUT/'ranking.csv',index=False,encoding='utf-8-sig')
    records=[]
    for r in data.itertuples():
        records.append({'关注顺序':r.priority,'排名规则':f'{r.rank_exit}/{r.strong_rank}',
            '退出':'直接清仓' if r.mode=='full' else '分段退出','止盈/止损':r.threshold.replace('_','% / ')+'%',
            '2025净收益':f'{r.total_return_2025:.2%}','2025回撤':f'{r.max_drawdown_2025:.2%}',
            '2026净收益*':f'{r.total_return_2026:.2%}','2026回撤*':f'{r.max_drawdown_2026:.2%}',
            '四项平均名次':f'{r.mean_metric_rank:.2f}','2022—25年化':f'{r.cagr_2022_2025:.2%}',
            '2022—25最大回撤':f'{r.drawdown_2022_2025:.2%}',
            '2026费用':f'{r.fees_2026/10000:.2f}万','2026完整周期':r.closed_2026,
            '被其他版本全面超过':'是' if r.dominated_by else '否'})
    table=pd.DataFrame(records)
    style='<style>body{font:16px/1.7 system-ui;color:#203047;max-width:1800px;padding:20px;margin:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f8;position:sticky;top:0}tbody tr:nth-child(-n+5){background:#eaf6ee}tbody tr:hover{background:#fff4ce}.scroll{overflow:auto;max-height:850px}p{max-width:1150px}a{color:#165aab}</style>'
    body='<html lang="zh"><meta charset="utf-8"><title>28个版本：2025—2026综合比较</title>'+style
    body+='<h1>28个版本：重点比较2025、2026</h1><p>按2025收益、2026收益、2025最大回撤、2026最大回撤四项分别排名，收益越高名次越靠前、回撤越小越靠前，再等权平均名次。并列先比较两段中较大的回撤，再比较费用占期初净值比例。该排序是本次描述性筛选，不是此前预注册的选模规则，不代表预期收益或显著性。</p>'
    body+='<p>2026*仅1月5日至10月13日185个交易日，年初100万元现金重新起跑；2025是2022—2025连续账户中的一年。两段收益不直接求平均、不复合为连续两年回报。回撤均在各段内部计算。费用已扣，不重复扣除；旧年份与费用只用于解释风险，未另加隐藏权重。</p>'
    body+='<div class="scroll">'+table.to_html(index=False)+'</div>'
    body+='<h2>前五个的用途与限制</h2><ol><li><b>80/2直接清仓10/5：</b>2025收益最高，近期回撤较小，作为均衡候选；2022—2025年化14.85%，回撤31.03%，仍超过30%历史筛选线。</li><li><b>80/2直接清仓8/3：</b>2026收益27.83%、回撤7.51%，作为近期收益候选；2022—2025年化10.99%、回撤35.06%，时期依赖明显。</li><li><b>65/2直接清仓8/4：</b>近两段收益都较强，保留原排名规则作收益对照；2026回撤9.85%，高于前两名。</li><li><b>80/2直接清仓6/2：</b>近期四指标排名靠前，但2026费用5.76万元、274个完整周期，旧区间年化8.88%、回撤37.27%，更适合作高频退出对照。</li><li><b>80/2分段退出8/3：</b>近期两段回撤均低于8%，用来检验分段退出是否值得；2022—2025年化10.88%、回撤34.91%，不是全历史低风险版本。</li></ol>'
    body+='<p>原65/2直接清仓10/5排第6，2022—2025年化18.84%，仍是有价值的长期基准。前5按近期四指标选出，不能据此删除该参照。五个候选全都未通过旧区间最大回撤≤30%的严格筛选线，因此此表只确定关注名单，不宣告可冻结或可部署。</p>'
    body+='<p>排名80/2意为跌到81名退出、新进入前2名触发替换。强势条件不限制正常空位补仓。策略保留H10季度模型、Top10、止盈冷却2日/止损3日、理想次日开盘成交；费用假设未获全部模拟账单逐笔验证。历史区间已多次研究，名单是事后筛选，不是独立测试结论。未新增训练、回测或模拟盘改动。</p>'
    body+='<p><a href="ranking.csv">完整数值与四项名次</a> · <a href="verification.json">来源核验</a></p></html>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',source_manifest_sha256=sha(SOURCE/'report_manifest.json'),
        configs=28,metrics=['2025 return','2026 return','2025 drawdown','2026 drawdown'],weight=[.25]*4,
        tie_break=['worst_recent_period_drawdown','mean_fee_rate_descriptive'],strict_past_drawdown_limit=.30,
        selected_names=data.head(5).name.tolist(),new_training=False,new_backtests=False,
        artifacts={n:sha(OUT/n) for n in ('ranking.csv','report.html')}),ensure_ascii=False,indent=2),encoding='utf-8')
    print(table.to_string(index=False))

if __name__=='__main__':main()
