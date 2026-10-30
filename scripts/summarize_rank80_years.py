"""Read-only ledger comparison; 2026 is a separate fresh-cash run."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAST=ROOT/'reports/experiments/top10_rank80_strong1_asof_20251231'
OLD=ROOT/'reports/experiments/top10_rank65_tp2_sl3_asof_20251231'
RECENT=ROOT/'reports/experiments/top10_rank80_strong1_2026_asof_20261013'
OUT=ROOT/'reports/analysis/rank80_annual_comparison_2022_2026'
KEY=['rank_exit','strong_rank','mode','threshold']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    sources={}
    for folder in (PAST,RECENT):
        complete=load(folder/'completion.json');assert complete['status']=='complete'
        assert complete['manifest_sha256']==sha(folder/'report_manifest.json')
        for path,expected in load(folder/'report_manifest.json')['artifacts'].items():
            assert sha(path)==expected,path;sources[path]=expected
    previous=pd.read_csv(PAST/'yearly.csv');recent=pd.read_csv(RECENT/'comparison.csv');recent['year']=2026
    expected=pd.concat([previous,recent],ignore_index=True)
    assert not expected.duplicated(KEY+['year']).any() and len(expected)==50
    rows=[]
    for cfg,g in expected.groupby(KEY,sort=True):
        rank,strong,mode,threshold=cfg
        name=f'r{rank}_s{strong}_{mode}_{threshold}'
        for folder,account,years in [((PAST if rank==80 else OLD),f'top10_{mode}_{threshold}',range(2022,2026)),
                                     (RECENT,name,[2026])]:
            directory=folder/'accounts'/account
            daily=pd.read_parquet(directory/'daily.parquet');daily['date']=pd.to_datetime(daily.date)
            orders=pd.read_parquet(directory/'orders.parquet');orders['date']=pd.to_datetime(orders.date)
            for f in ('daily','orders'):
                path=directory/f'{f}.parquet';sources[str(path)]=sha(path)
            for year in years:
                d=daily[daily.date.dt.year.eq(year)];o=orders[orders.date.dt.year.eq(year)]
                r=d.net_return.to_numpy();nav=(1+r).cumprod();opening=float(d.nav.iloc[0]/(1+r[0]))
                value=float(d.nav.iloc[-1]/opening-1);dd=float((1-nav/np.maximum.accumulate(np.r_[1,nav])[1:]).max())
                row=g[g.year.eq(year)].iloc[0]
                np.testing.assert_allclose([value,dd,d.fees.sum()],[row.total_return,row.max_drawdown,row.fees],atol=1e-10,rtol=0)
                np.testing.assert_allclose(d.fees.sum(),o.fee.sum(),atol=1e-7,rtol=0)
                rows.append(dict(zip(KEY,cfg))|dict(name=name,year=year,start=str(d.date.min().date()),end=str(d.date.max().date()),
                    days=len(d),total_return=value,max_drawdown=dd,opening_nav=opening,fees=float(d.fees.sum()),
                    fees_over_opening_nav=float(d.fees.sum()/opening),fills=len(o),
                    annualized_daily_vol=float(np.std(r,ddof=1)*np.sqrt(252)),
                    basis='fresh_2026_cash' if year==2026 else 'continuous_2022_2025'))
    table=pd.DataFrame(rows)
    summaries=[]
    for cfg,g in table.groupby(KEY,sort=True):
        by=g.set_index('year');past=by.loc[2022:2025];last=by.loc[[2025,2026]]
        summaries.append(dict(zip(KEY,cfg))|dict(name=g.name.iloc[0],
            return_2025=by.loc[2025,'total_return'],return_2026=by.loc[2026,'total_return'],
            drawdown_2025=by.loc[2025,'max_drawdown'],drawdown_2026=by.loc[2026,'max_drawdown'],
            recent_lower_return=float(last.total_return.min()),recent_worst_year_drawdown=float(last.max_drawdown.max()),
            complete_year_return_std=float(past.total_return.std(ddof=0)),
            complete_year_mean_return=float(past.total_return.mean()),
            worst_complete_year_return=float(past.total_return.min()),
            fee_rate_2025=by.loc[2025,'fees_over_opening_nav'],fee_rate_2026=by.loc[2026,'fees_over_opening_nav']))
    summary=pd.DataFrame(summaries)
    OUT.mkdir(parents=True,exist_ok=True)
    table.to_csv(OUT/'yearly_verified.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(OUT/'stability.csv',index=False,encoding='utf-8-sig')
    names={};display=[]
    for i,(cfg,g) in enumerate(table.groupby(KEY,sort=True),1):
        rank,strong,mode,threshold=cfg;ident=f'V{i}'
        label=f'{rank}/{strong} '+('直接清仓' if mode=='full' else '分段退出')+' '+threshold.replace('_','% / ')+'%'
        names[g.name.iloc[0]]=ident+' '+label
        by=g.set_index('year');display.append({'编号':ident,'版本':label}|{str(y)+('*' if y==2026 else ''):f'{by.loc[y,"total_return"]:.2%} / {by.loc[y,"max_drawdown"]:.2%}' for y in range(2022,2027)})
    disp=pd.DataFrame(display)
    focus=[]
    for r in summary.itertuples():
        focus.append({'版本':names[r.name],'2025净收益':f'{r.return_2025:.2%}','2026净收益*':f'{r.return_2026:.2%}',
          '2025回撤':f'{r.drawdown_2025:.2%}','2026回撤*':f'{r.drawdown_2026:.2%}',
          '2025费用/期初净值':f'{r.fee_rate_2025:.2%}','2026费用/期初净值*':f'{r.fee_rate_2026:.2%}'})
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,6.5))
    order=list(names)
    for ax,metric,title,cmap in zip(axes,['total_return','max_drawdown'],['Net return (%)','Within-year maximum drawdown (%)'],['RdYlGn','YlOrRd']):
        arr=table.pivot(index='name',columns='year',values=metric).reindex(order).to_numpy()*100
        im=ax.imshow(arr,aspect='auto',cmap=cmap)
        ax.set_xticks(range(5),['2022','2023','2024','2025','2026*']);ax.set_yticks(range(10),[names[n].split()[0] for n in order]);ax.set_title(title)
        for i in range(10):
            for j in range(5):ax.text(j,i,f'{arr[i,j]:.1f}',ha='center',va='center',fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('2026*: Jan 5 - Oct 13, fresh cash; 2022-2025: continuous account')
    fig.tight_layout();fig.savefig(OUT/'yearly_heatmap.png',dpi=160);plt.close(fig)
    body='<html lang="zh"><meta charset="utf-8"><title>2022—2026逐年稳定性比较</title><style>body{font:16px/1.7 system-ui;max-width:1500px;margin:35px auto;padding:0 16px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#eef3fa}img{width:100%}.scroll{overflow-x:auto}</style>'
    body+='<h1>Top10各版本逐年收益与回撤</h1><p>每格：扣费后区间收益 / 年内最大回撤。2022—2025是连续账户按年拆分；2026*为1月5日至10月13日的185个交易日，重新以100万元现金起跑。未把两段拼接成五年总收益，不将2026区间收益当作全年收益，也不以五个不等长区间的标准差排名。</p>'
    body+='<div class="scroll">'+disp.to_html(index=False)+'</div><h2>重点比较2025、2026</h2><div class="scroll">'+pd.DataFrame(focus).to_html(index=False)+'</div>'
    body+='<p>费用已从净收益扣除，费用/期初净值仅用于比较交易负担，不应再扣一次。年内最大回撤从每年期初净值计入峰值重新计算，不包括此前年度高点；两年回撤的较大值也不是连续两年最大回撤。</p>'
    body+='<h2>研究判断</h2><p>原65/2直接清仓10/5在2025、2026分别盈利14.59%、15.19%，回撤7.04%、8.66%，近两段收益更接近且回撤较低。原65/2直接清仓8/4近两段收益14.60%、26.21%，收益更强，但回撤7.58%、9.85%稍高、2022更弱。两者应分别视为稳定参照与近期收益候选，不能仅根据这一轮历史比较自动切换。</p>'
    body+='<p>新80/1分段10/5在2025仅7.62%，2026升至25.71%，并非持续领先。新80/1分段6/2在2025、2026较强，但2022亏损24.66%、回撤40.03%，2024也弱于宽止盈止损；更像时期依赖，而不是已证明更稳健。</p>'
    body+='<p>所有版本2022亏损，2024收益显著高于其他年度。全历史不存在跨年份都平稳的版本。唯一2022—2025连续回撤未超过30%的原65/2分段10/5，仍不代表未来回撤不会超过30%。样本已反复研究，2026也已观察，本表是描述性比较，无法证明某版本不存在过拟合。</p><img src="yearly_heatmap.png">'
    body+='<p>排名65/2表示跌出65退出、新进前2触发强势替换；80/1表示跌出80退出、新进第1替换。其余保持H10季度模型、Top10、止盈冷却2日/止损3日、保留仓位漂移。日线理想开盘成交，既定费用假设未经全部模拟盘账单逐笔核验。没有重训、回放新增账户或修改模拟盘。</p><p><a href="yearly_verified.csv">逐年核验明细</a> · <a href="stability.csv">稳定性描述指标</a></p></html>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    for path,h in sources.items():assert sha(path)==h
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',year_rows=50,configurations=10,
        source_hashes=sources,returns_drawdowns_fees_recomputed=True,no_continuous_2022_2026_claim=True),indent=2),encoding='utf-8')
    print(disp.to_string(index=False));print(pd.DataFrame(focus).to_string(index=False))

if __name__=='__main__':main()
