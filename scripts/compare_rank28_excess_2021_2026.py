"""Read-only, segmented performance comparison; never stitches account NAVs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAST = ROOT/'reports/experiments/top28_continuous_2021_2025'
RECENT = ROOT/'reports/experiments/top10_rank80_two_thresholds_asof_20261013'
DATA26 = ROOT/'reports/experiments/top10_rank80_strong1_2026_asof_20261013/data'
OUT = ROOT/'reports/analysis/top28_excess_2021_2026'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def stats(s):
    wealth = np.r_[1., (1+s).cumprod().to_numpy()]
    return dict(total_return=float(wealth[-1]-1),
                max_drawdown=float((1-wealth/np.maximum.accumulate(wealth)).max()))


def active_stats(r, b):
    assert r.index.equals(b.index) and r.notna().all() and b.notna().all()
    active = r-b
    sd = active.std(ddof=1)
    return dict(tracking_error=float(sd*np.sqrt(252)),
                arithmetic_active_annual=float(active.mean()*252),
                information_ratio=float(active.mean()/sd*np.sqrt(252)))


def main():
    sources = {}
    for folder in (PAST, RECENT):
        c = read(folder/'completion.json')
        assert c['status'] == 'complete' and c['manifest_sha256'] == sha(folder/'report_manifest.json')
        for p, h in read(folder/'report_manifest.json')['artifacts'].items():
            assert sha(p) == h, p
        sources[str(folder/'report_manifest.json')] = sha(folder/'report_manifest.json')
    past = pd.read_csv(PAST/'ranking.csv').set_index('name')
    current = pd.read_csv(RECENT/'comparison.csv').query("period == '2026'").set_index('name')
    b0 = pd.read_parquet(PAST/'data/benchmark.parquet').benchmark_return
    b1 = pd.read_parquet(DATA26/'benchmark.parquet').benchmark_return
    b0.index = pd.to_datetime(b0.index); b1.index = pd.to_datetime(b1.index)
    bench = pd.concat([b0,b1])
    assert len(bench)==1397 and not bench.index.duplicated().any()
    benchmark_rows = []
    for code in ('000300','H00300'):
        frames = []
        for d in (PAST/'data/benchmark', DATA26/'benchmark'):
            path = d/f'{code}.json'; sources[str(path)] = sha(path)
            x = pd.DataFrame(read(path)['data'])
            x.index = pd.to_datetime(x.tradeDate, format='%Y%m%d')
            frames.append(x.close.astype(float))
        closes = pd.concat(frames)
        duplicate = closes[closes.index.duplicated(keep=False)]
        assert duplicate.groupby(level=0).nunique().le(1).all()
        closes = closes[~closes.index.duplicated()].sort_index()
        r = closes.pct_change(fill_method=None).reindex(bench.index)
        assert r.notna().all()
        if code == 'H00300':
            np.testing.assert_allclose(r, bench, atol=1e-12)
        for year, g in r.groupby(r.index.year):
            benchmark_rows.append(dict(code=code, year=int(year), days=len(g), **stats(g)))
    yearly = []; records = []
    for name, row in past.iterrows():
        d0 = PAST/'accounts'/name
        d1 = RECENT/'accounts/2026'/name
        if not d1.exists():
            # Existing audited accounts were reused without copying.
            from paper_futu.research_rank80_two_thresholds import existing
            d1 = existing('2026', (int(row.rank_exit),int(row.strong_rank),row['mode'],row.threshold))
        returns = []
        for folder, b in ((d0,b0),(d1,b1)):
            path = folder/'daily.parquet'
            expected = read(folder/'complete.json')['artifacts'][str(path)]
            assert sha(path)==expected
            sources[str(path)] = expected
            daily = pd.read_parquet(path)
            r = pd.Series(daily.net_return.to_numpy(),index=pd.to_datetime(daily.date))
            assert r.index.equals(b.index)
            np.testing.assert_allclose((1+r).cumprod()*1e6,daily.nav,rtol=1e-10)
            returns.append(r)
            for year, g in r.groupby(r.index.year):
                bg = b.reindex(g.index)
                sr, br = stats(g), stats(bg)
                yearly.append(dict(name=name,year=int(year),days=len(g),**sr,
                    benchmark_return=br['total_return'],
                    geometric_excess=(1+sr['total_return'])/(1+br['total_return'])-1,
                    **active_stats(g,bg)))
        r0,r1 = returns
        np.testing.assert_allclose((1+r0).prod()-1,row.total_return)
        np.testing.assert_allclose((1+r1).prod()-1,current.loc[name,'total_return'])
        r = pd.concat(returns)
        allstats = active_stats(r,bench)
        yrs = [x['geometric_excess'] for x in yearly if x['name']==name and x['year']<=2025]
        records.append(dict(name=name,rank_exit=row.rank_exit,strong_rank=row.strong_rank,
            mode=row['mode'],threshold=row.threshold,cagr_2021_2025=row.cagr,
            annual_geometric_excess_2021_2025=row.annual_geometric_excess,
            drawdown_2021_2025=row.max_drawdown,
            return_2026=current.loc[name,'total_return'],
            excess_2026=current.loc[name,'geometric_excess'],
            drawdown_2026=current.loc[name,'max_drawdown'],
            annual_excess_std_2021_2025=float(np.std(yrs,ddof=1)),
            positive_excess_years_2021_2025=sum(v>0 for v in yrs),
            ir_2021_2025=active_stats(r0,b0)['information_ratio'],
            ir_2026=active_stats(r1,b1)['information_ratio'],**allstats))
    ranking = pd.DataFrame(records).sort_values(['information_ratio','drawdown_2021_2025'],ascending=[False,True])
    ranking.insert(0,'priority',range(1,29))
    annual = pd.DataFrame(yearly)
    assert len(annual)==168 and len(ranking)==28
    OUT.mkdir(parents=True,exist_ok=True)
    ranking.to_csv(OUT/'ranking.csv',index=False,encoding='utf-8-sig')
    annual.to_csv(OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(benchmark_rows).to_csv(OUT/'benchmark.csv',index=False,encoding='utf-8-sig')
    display = ranking[['priority','name','cagr_2021_2025','annual_geometric_excess_2021_2025',
        'return_2026','excess_2026','tracking_error','information_ratio','annual_excess_std_2021_2025',
        'drawdown_2021_2025','drawdown_2026']].copy()
    for col in display.columns:
        if col not in ('priority','name','information_ratio'): display[col]=display[col].map(lambda v:f'{v:.2%}')
    display.information_ratio=display.information_ratio.map(lambda v:f'{v:.3f}')
    display.columns=['顺序','配置','2021—25净年化','2021—25几何超额年化','2026净收益','2026几何超额',
        '日超额年化标准差','分段合并信息比率','2021—25年度超额标准差','2021—25最大回撤','2026最大回撤']
    body='''<html lang="zh"><meta charset="utf-8"><title>28版本：2021—2026超额比较</title>
    <style>body{font:15px/1.7 system-ui;margin:30px;color:#203047}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;border:1px solid #ddd;text-align:right}th{background:#edf3fa}.scroll{overflow:auto}p{max-width:1200px}</style>
    <h1>28版本：2021—2026分段表现与超额稳定性</h1>
    <p>2021—2025是连续账户；2026截至10月13日是100万元现金独立起跑。没有拼接净值、计算六年连续年化或六年连续回撤。2026不是完整年度。</p>
    <p>主基准为含分红的H00300。每日主动收益a=r策略−r基准；跟踪误差=样本标准差(a)×√252；信息比率=平均(a)×252/跟踪误差。以全部1397个交易日的分段观察合并计算，日等权，五年历史占比86.8%。按信息比率降序排列，回撤作为独立风险约束；不是原预注册选模规则。该指标不是风险因子调整后Alpha。</p>
    <p>年度几何超额=(1+策略年度收益)/(1+基准年度收益)−1。年度超额标准差只用2021—2025五个完整年度，不能将2026不足一年数据混入同尺度计算。五年年化沿用252交易日口径。基准首日因缺少全收益开盘指数，采用前一年末收盘起算，含策略尚未持仓的首夜收益，存在首日时点差异。</p>
    <p>全部28版五年回撤超过30%，因此表中优先级仅为研究排序，无风险合格替代版本。历史数据经过多次筛选，不是独立验证；低跟踪误差不等于低绝对回撤。2024年集中贡献应结合逐年表判断。</p>'''
    body+='<div class="scroll">'+display.to_html(index=False)+'</div><h2>沪深300基准</h2>'
    bt=pd.DataFrame(benchmark_rows)
    for c in ('total_return','max_drawdown'):bt[c]=bt[c].map(lambda v:f'{v:.2%}')
    body+=bt.to_html(index=False)+'<h2>逐年策略与超额</h2><div class="scroll">'+annual.to_html(index=False,float_format=lambda v:f'{v:.6f}')+'</div>'
    body+='<p><a href="ranking.csv">28版数值</a> · <a href="yearly.csv">逐年数值</a> · <a href="benchmark.csv">基准数值</a></p></html>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',sources=sources,configs=28,
        years=168,days=1397,continuous_six_year_account=False,benchmark='H00300',ddof=1,
        ranking='pooled daily active information ratio descending; historical drawdown tie break',
        artifacts={f:sha(OUT/f) for f in ('ranking.csv','yearly.csv','benchmark.csv','report.html')}),indent=2),encoding='utf-8')
    print(display.head(10).to_string(index=False))
    print(bt.to_string(index=False))


if __name__=='__main__':main()
