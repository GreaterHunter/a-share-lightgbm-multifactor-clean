"""Fee-only S6 replay in an independent directory; frozen sources stay unchanged."""
import ast
import inspect
import textwrap
import numpy as np
import pandas as pd
from paper_futu import research_momentum_s6 as source
from paper_futu import final_ablation_account as account
from paper_futu.research_rank_buffer_futu_fees import fee,verify
from paper_futu.top5_buffer_data import ROOT,read,save,digest,init_qlib
from paper_futu.top5_buffer_report import metrics

OUT=ROOT/'reports/experiments/s6_futu_fees_asof_20251231'

def cash_cap(cash,friction):
    lo,hi=0.,max(0.,cash)
    for _ in range(65):
        mid=(lo+hi)/2
        if mid+fee(mid,True,friction)<=cash:lo=mid
        else:hi=mid
    return lo

def exchange_class():
    # Preserve the sealed exchange's execution method except fee/cash capacity.
    tree=ast.parse(textwrap.dedent(inspect.getsource(account.AuditedExchange._calc_trade_info_by_order)))
    matches=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='affordable' for t in node.targets):
            node.value=ast.parse('cash_cap(cash, self.friction) / price',mode='eval').body;matches+=1
    assert matches==1
    env=dict(account.__dict__,fee=lambda value,direction,friction=0:fee(value,direction==1,friction),cash_cap=cash_cap)
    exec(compile(ast.fix_missing_locations(tree),'<isolated S6 fee adapter>','exec'),env)
    return type('FutuPublishedFeeExchange',(account.AuditedExchange,),{'_calc_trade_info_by_order':env['_calc_trade_info_by_order']})

def replay_function():
    code=inspect.getsource(source.replay)
    old='from paper_futu.final_ablation_account import AuditedExchange,AuditedStrategy'
    assert code.count(old)==1
    code=code.replace(old,'from paper_futu.final_ablation_account import AuditedStrategy')
    old='expected_fees=np.where(filled.direction.eq(1),np.maximum(filled.notional*.0005,5.),np.maximum(filled.notional*.0015,5.))'
    assert code.count(old)==1
    code=code.replace(old,'expected_fees=np.array([futu_fee(v,d==1) for v,d in zip(filled.notional,filled.direction)])')
    env=dict(source.__dict__,OUT=OUT,AuditedExchange=exchange_class(),futu_fee=fee)
    exec(compile(code,'<isolated S6 fee replay>','exec'),env)
    return env['replay']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=read(source.OUT/'protocol.json')
    bindings={**old['sources'],**old['implementation'],**read(source.OUT/'report_manifest.json')['artifacts'],
              __file__:digest(__file__),str(ROOT/'paper_futu/research_rank_buffer_futu_fees.py'):digest(ROOT/'paper_futu/research_rank_buffer_futu_fees.py')}
    verify(bindings);verify(old['protected'])
    protocol=dict(start=source.START,end=source.END,case=source.CASES['s6_main'],sources=bindings,protected=old['protected'],
        fee_source='https://support.futunn.com/topic692',fee_date='2026-10-28',
        limitations=['Published transaction fees provisionally applied to all historical orders; not exact broker bill replication.',
        'Custody fee excluded pending clarification; full precision arithmetic; no additional slippage.',
        'Same legacy S6 sizing, rounding and ideal-open fill assumptions; not the newer live integer execution logic.',
        'Observed actual single-order fee differs slightly from published schedule. Historical research, not independent testing.'])
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    calendar=pd.DatetimeIndex(pd.to_datetime((source.SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    assert calendar.max()==pd.Timestamp(source.END)
    pred=pd.read_parquet(source.SOURCE/'data/predictions.parquet');assert pred.execution_date.max()<=pd.Timestamp(source.END)
    benchmark=pd.read_parquet(source.SOURCE/'data/benchmark.parquet').benchmark_return
    members=pd.read_csv(source.SOURCE/'data/provider/instruments/csi300.txt',sep='\t',header=None)
    init_qlib(source.SOURCE/'data/provider')
    replay_function()('s6_main',source.CASES['s6_main'],pred[['prediction']],calendar,sorted(members[0].unique()),benchmark)
    directory=OUT/'accounts/s6_main';verify(read(directory/'complete.json')['artifacts'])
    daily=pd.read_parquet(directory/'daily.parquet');new=metrics(daily,benchmark)
    previous=pd.read_csv(source.OUT/'comparison.csv').query("case=='s6_main'").iloc[0]
    new['final_nav']=float(daily.nav.iloc[-1]);new['net_profit']=new['final_nav']-1e6
    save(OUT/'summary.json',new)
    yearly=pd.DataFrame([dict(year=int(y),**metrics(g,benchmark)) for y,g in daily.groupby(daily.date.dt.year)])
    yearly.to_csv(OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    table=pd.DataFrame([dict(fee_model='旧成本',**{k:previous[k] for k in ('total_return','cagr','max_drawdown','fees')}),dict(fee_model='富途公开交易费暂定',**{k:new[k] for k in ('total_return','cagr','max_drawdown','fees')})])
    table.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    show=table.copy()
    for c in ('total_return','cagr','max_drawdown'):show[c]=show[c].map(lambda x:f'{x:.2%}')
    page='<html lang="zh"><meta charset="utf-8"><title>S6富途费用回测</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto}td,th{padding:10px}</style><h1>S6主策略：富途公开交易费用暂定口径</h1><p>2022—2025，初始100万元；季度重训、预测10日、Top20前6名40%、每5日调仓、最多换6只。无新增止盈止损。</p>'+show.to_html(index=False)
    page+=f'<p>期末资产{new["final_nav"]:,.2f}元；净收益{new["net_profit"]:,.2f}元。费用已经扣除，不能再次扣除。</p><p>沿用公开交易费表，未核实证券组合费未扣，未另加滑点；实际模拟盘扣费与官网有小幅差异，因此不是精确复刻账单。理想开盘成交、保留原回测取整规则，历史结果不是实际模拟账户收入。</p><h2>逐年指标（收益率为小数）</h2>'+yearly[['year','total_return','max_drawdown','fees']].to_html(index=False)+'</html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    verify(bindings);verify(old['protected'])
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json','report_manifest.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',source_and_live_unchanged=True,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(new,flush=True)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
