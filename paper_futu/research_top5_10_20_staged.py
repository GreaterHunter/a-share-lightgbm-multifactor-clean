"""Confirmed staged 10/5 and proportionally scaled 8/4; 2022-2025 only."""
import sys
import types
from dataclasses import replace
from paper_futu import research_top5_head2_full_exit as full
from paper_futu import research_rank_buffer_futu_fees as fees
from paper_futu.top5_buffer_data import ROOT,START,END,read,save,digest

OUT=ROOT/'reports/experiments/top5_10_20_staged_asof_20251231'
VARIANTS={'staged_10_5':(.10,.05),'staged_8_4':(.08,.04)}

def engine():
    name='_top5_10_20_staged_private'
    if name in sys.modules:return sys.modules[name]
    path=ROOT/'paper_futu/rank_buffer_10_20_engine.py';code=path.read_text(encoding='utf-8')
    for old,new in {'if self.top_k not in (10, 20):':'if self.top_k not in (5, 10, 20):',
                    'nav_open*self.rules.entry_weight':'nav_open*entry_weight(self, code, ranks)'}.items():
        assert code.count(old)==1;code=code.replace(old,new)
    mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    exec(compile(code,str(path),'exec'),mod.__dict__)
    mod.fee=fees.fee;mod.buy_quantity=fees.buy_quantity;mod.liquidation_breakeven=fees.liquidation_breakeven
    mod.entry_weight=full.entry_weight
    return mod

def make_account(k,tp,sl):
    if (tp,sl) not in VARIANTS.values():raise ValueError('Only the two confirmed threshold pairs')
    a=engine().Account(k)
    a.rules=replace(a.rules,profit=tp,loss_budget=sl,loss_half=sl*.8,trailing_drawdown=sl)
    return a

def run():
    import json
    import numpy as np
    import pandas as pd
    from paper_futu.top5_buffer_report import metrics,trade_metrics
    p=read(full.OUT/'protocol.json');done=read(full.OUT/'completion.json')
    assert done['status']=='complete' and done['manifest_sha256']==digest(full.OUT/'report_manifest.json')
    bindings={**p['source'],**read(full.OUT/'report_manifest.json')['artifacts'],
              str(ROOT/'paper_futu/research_top5_head2_full_exit.py'):digest(ROOT/'paper_futu/research_top5_head2_full_exit.py'),
              __file__:digest(__file__)}
    fees.verify(bindings);fees.verify(p['protected'])
    protocol=dict(start=START,end=END,variants=VARIANTS,top_k=[5,10,20],initial_cash=1e6,
        source=bindings,protected=p['protected'],fee_policy=p['fee_policy'],entry_weights=p['entry_weights'],retained=p['retained'],
        confirmation='Both threshold pairs confirmed; 8/4 loss half=3.2%, profit=8%, trailing=4%, second profit=half fill*1.08',
        priorities='Fee-inclusive whole-episode loss budget first, then staged price logic, then rank/strong-entry exits',
        staged='If not yet halved and rank<=K: profit TP or loss 80% of SL -> half. Otherwise profit TP -> full, loss waits SL/budget. After loss-half original TP retained. After profit-half fee-adjusted remaining-share breakeven/trailing and second TP. At most one half; no topups.',
        boundary='Completed close signal, next ideal open; ranking exit>50, D+3 cooldown, no 2026, no refit, year-end mark only',
        workers=1,threads=4,memory_gate=False)
    protocol=json.loads(json.dumps(protocol))
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    panel=pd.read_parquet(fees.SOURCE/'data/market.parquet').reset_index();assert panel.datetime.max()<=pd.Timestamp(END)
    market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')};del panel
    frame=pd.read_parquet(fees.SOURCE/'data/predictions.parquet').reset_index();assert frame.execution_date.max()<=pd.Timestamp(END)
    predictions=dict(tuple(frame.groupby('execution_date')))
    calendar=pd.DatetimeIndex(pd.to_datetime((fees.SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    assert calendar.max()==pd.Timestamp(END)
    days=calendar[(calendar>=START)&(calendar<=END)]
    benchmark=pd.read_parquet(fees.SOURCE/'data/benchmark.parquet').benchmark_return
    rows=[];years=[];trades=[];exits=[];checks={};reproduced={};curves={}
    for k in (5,10,20):
        for v,(tp,sl) in VARIANTS.items():
            name=f'top{k}_{v}';d=OUT/'accounts'/name;d.mkdir(parents=True,exist_ok=True)
            if not (d/'complete.json').exists():
                a=make_account(k,tp,sl)
                for step,day in enumerate(days):
                    prev=calendar[calendar.get_loc(day)-1];r=predictions[day];assert r.datetime.eq(prev).all()
                    a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],
                           {c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
                a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
                for n,data in dict(daily=a.daily,orders=a.orders,holdings=a.holdings,episodes=a.episodes,open_episodes=opened,events=a.events).items():
                    pd.DataFrame(data).to_parquet(d/f'{n}.parquet',index=False)
                save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
            c=read(d/'complete.json');assert c['protocol_sha256']==digest(OUT/'protocol.json');fees.verify(c['artifacts'])
            if k!=5 and v=='staged_10_5':
                for n in ('daily','orders','holdings','episodes','open_episodes','events'):
                    pd.testing.assert_frame_equal(pd.read_parquet(d/f'{n}.parquet'),pd.read_parquet(fees.OUT/'accounts'/name/f'{n}.parquet'),check_exact=True)
                reproduced[name]=True
            daily,orders,held,closed,opened,events=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes','events')]
            for f in (daily,orders,held):f['date']=pd.to_datetime(f.date)
            assert pd.DatetimeIndex(daily.date).equals(benchmark.index)
            checks[name]=fees.ledger_audit(daily,orders,held,closed,opened,k)
            half_events=events.loc[events.event.eq('half_complete')]
            if len(half_events):assert half_events.groupby('episode_id').size().max()==1
            episodes=pd.concat([closed,opened],ignore_index=True)
            np.testing.assert_allclose(episodes.budget,np.minimum(episodes.entry_nav*.02,episodes.initial_cost*sl),rtol=0,atol=1e-8)
            info=dict(top_k=k,variant=v);rows.append(dict(**info,**metrics(daily,benchmark)))
            for y,g in daily.groupby(daily.date.dt.year):years.append(dict(**info,year=int(y),**metrics(g,benchmark)))
            trades.append(dict(**info,**trade_metrics(closed,opened),fills=len(orders),mean_cycle_return=float(closed.return_on_cost.mean()),
                               loss_half_episodes=int(episodes.half_mode.eq('loss').sum()),profit_half_episodes=int(episodes.half_mode.eq('profit').sum())))
            for reason,g in closed.groupby('exit_reason'):exits.append(dict(**info,reason=reason,count=len(g),mean_return=float(g.return_on_cost.mean())))
            curves[name]=daily
            print(f'COMPLETE {name}: NAV {daily.nav.iloc[-1]:.2f}',flush=True)
    staged=pd.DataFrame(rows);direct=pd.read_csv(full.OUT/'comparison.csv')
    paired=staged.assign(threshold=staged.variant.str.replace('staged_','',regex=False)).merge(
        direct.assign(threshold=direct.variant.str.replace('full_','',regex=False)),on=['top_k','threshold'],suffixes=('','_full'))
    paired['cagr_delta']=paired.cagr-paired.cagr_full;paired['drawdown_delta']=paired.max_drawdown-paired.max_drawdown_full
    annual=pd.DataFrame(years)
    for n,t in dict(comparison=staged,paired_comparison=paired,yearly=annual,trade_summary=pd.DataFrame(trades),exit_summary=pd.DataFrame(exits)).items():
        t.to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(14,12),sharex=True)
    for i,k in enumerate((5,10,20)):
        for j,tag in enumerate(('10_5','8_4')):
            ax=axes[i,j];d=curves[f'top{k}_staged_{tag}']
            f=pd.read_parquet(full.OUT/'accounts'/f'top{k}_full_{tag}'/'daily.parquet')
            ax.plot(pd.to_datetime(d.date),d.nav/1e6,label='Staged');ax.plot(pd.to_datetime(f.date),f.nav/1e6,label='Full exit')
            ax.set_title(f'Top{k} | TP/SL {tag}');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'paired_nav.png',dpi=150);plt.close(fig)
    show=paired[['top_k','threshold','total_return','cagr','cagr_full','max_drawdown','max_drawdown_full','fees']].copy()
    for c in ('total_return','cagr','cagr_full','max_drawdown','max_drawdown_full'):show[c]=show[c].map(lambda x:f'{x:.2%}')
    show.fees=show.fees.map(lambda x:f'{x:,.2f}')
    yearshow=annual.pivot(index=['top_k','variant'],columns='year',values='total_return').map(lambda x:f'{x:.2%}')
    page='<html lang="zh"><meta charset="utf-8"><title>Top5/10/20分段退出</title><style>body{font:16px/1.7 system-ui;max-width:1400px;margin:35px auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>Top5/10/20：分段退出与直接清仓</h1><p>2022-01-04至2025-12-31；初始100万元。新增六组分段账户，对照六组已核验直接清仓账户。无重新训练、不使用2026数据、不修改S6模拟盘。</p>'
    page+='<p>Top5入场前两只合计股票仓位50%，其余三只50%；Top10/20入场等权，总股票预算95%。保留股允许漂移，不每日配平，不补回减半股数。头部是入场时持仓加新候选的排名。</p>'
    page+='<h2>分段规则</h2><p>10%／5%：未减半且仍在前K名，涨10%或跌4%减半；不在前K名时，涨10%全出、跌至5%或触发预算全出。8%／4%：对应改为涨8%、跌3.2%减半，止损4%。含费累计亏损预算始终优先。</p><p>亏损减半后，保留原始10%／8%止盈线；再次触发累计亏损预算即全部退出。盈利减半后，跟踪线为剩余股份含费保本价与已知最高价回撤5%／4%的较高者，只上移；第二次止盈为减半实际成交价再涨10%／8%。每周期最多减半一次。整手不能合法拆分则全出。</p>'
    page+='<p>所有条件由已完成收盘价判断、次日理想开盘执行。排名超50退出、新进前2换股、风险清仓D+3冷却等不变。未成交全退出意图持续保留。止损预算非成交保证，跳空可能超预算。</p>'
    page+='<h2>扣费后对照</h2>'+show.to_html(index=False)+'<h2>分段版逐年收益</h2>'+yearshow.to_html()+'<img src="paired_nav.png"><h2>周期和减半次数</h2>'+pd.DataFrame(trades).round(5).to_html(index=False)
    page+='<p>费用沿用富途公开交易费表：证券组合费未核实未计入，未额外加滑点，不能视为精确复刻模拟账户账单。历史数据已反复研究，不是独立测试。Top10/20的10%／5%分段版与此前账户逐项一致。</p><p><a href="paired_comparison.csv">配对指标</a> · <a href="yearly.csv">逐年</a> · <a href="trade_summary.csv">周期统计</a> · <a href="exit_summary.csv">退出原因</a> · <a href="verification.json">验收</a></p></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    fees.verify(bindings);fees.verify(p['protected'])
    save(OUT/'verification.json',dict(status='passed',checks=checks,original_staged_reproduced=reproduced,at_most_one_half=True,budget_never_reset=True,sources_and_live_unchanged=True))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json','report_manifest.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',new_accounts=6,comparison_accounts=6,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(show.to_string(index=False),flush=True);print(yearshow.to_string(),flush=True)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
