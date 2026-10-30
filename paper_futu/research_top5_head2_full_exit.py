"""Top5 head-two 50% and Top10/20 full-exit comparison, bounded at 2025."""
import sys
import types
import math
from paper_futu import research_rank_buffer_futu_fees as fees
from paper_futu.top5_buffer_data import ROOT,START,END,read,save,digest

OUT=ROOT/'reports/experiments/top5_head2_50_full_exit_asof_20251231'
VARIANTS={'full_10_5':(.10,.05),'full_8_4':(.08,.04)}

def entry_weight(account,code,ranks):
    if account.rules.top_k!=5:return .95/account.rules.top_k
    # Retained holdings drift: tiers are set at entry, not rebalanced each day.
    candidates=set(account.positions)|{code}
    ordered=sorted(candidates,key=lambda c:(ranks.get(c,math.inf),c))
    return .95*(.5/2 if ordered.index(code)<2 else .5/3)

def engine():
    name='_top5_head2_full_exit_private'
    if name in sys.modules:return sys.modules[name]
    path=ROOT/'paper_futu/rank_buffer_full_exit_engine.py'
    code=path.read_text(encoding='utf-8')
    edits={'if self.top_k not in (10, 20):':'if self.top_k not in (5, 10, 20):',
           'nav_open*self.rules.entry_weight':'nav_open*entry_weight(self, code, ranks)'}
    for old,new in edits.items():
        assert code.count(old)==1
        code=code.replace(old,new)
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(code,str(path),'exec'),module.__dict__)
    module.fee=fees.fee;module.buy_quantity=fees.buy_quantity
    module.liquidation_breakeven=fees.liquidation_breakeven;module.entry_weight=entry_weight
    return module

def run():
    import numpy as np
    import pandas as pd
    from paper_futu.top5_buffer_report import metrics,trade_metrics
    p=read(fees.OUT/'protocol.json');c=read(fees.OUT/'completion.json')
    assert c['status']=='complete' and c['manifest_sha256']==digest(fees.OUT/'report_manifest.json')
    bindings={**p['source'],**read(fees.OUT/'report_manifest.json')['artifacts'],
              str(ROOT/'paper_futu/research_rank_buffer_futu_fees.py'):digest(ROOT/'paper_futu/research_rank_buffer_futu_fees.py'),
              __file__:digest(__file__)}
    fees.verify(bindings);fees.verify(p['protected'])
    protocol=dict(start=START,end=END,top_k=[5,10,20],variants=VARIANTS,source=bindings,protected=p['protected'],
        fee_policy={k:p[k] for k in ('fee_source','fee_checked_date','rounding','custody_fee','actual_fee_evidence')},
        entry_weights='Top5: entry-time first two among current holdings plus candidate 23.75% NAV each, others 15.8333%; Top10/20 95%/K',
        retained='No daily equalization, topups or reweighting on rank changes; actual tier weights drift',
        unchanged='Daily close decision / next ideal open; exit rank>50; new top2 entrance; D+3 risk exit cooldown; no halves; fee-inclusive loss budget',
        no_training=True,no_2026=True,workers=1,threads=4,memory_gate=False)
    # JSON canonical form for tuple variants.
    import json
    protocol=json.loads(json.dumps(protocol))
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    panel=pd.read_parquet(fees.SOURCE/'data/market.parquet').reset_index();assert panel.datetime.max()<=pd.Timestamp(END)
    market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')};del panel
    frame=pd.read_parquet(fees.SOURCE/'data/predictions.parquet').reset_index();assert frame.execution_date.max()<=pd.Timestamp(END)
    pred=dict(tuple(frame.groupby('execution_date')))
    calendar=pd.DatetimeIndex(pd.to_datetime((fees.SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    assert calendar.max()==pd.Timestamp(END)
    days=calendar[(calendar>=START)&(calendar<=END)]
    benchmark=pd.read_parquet(fees.SOURCE/'data/benchmark.parquet').benchmark_return
    rows=[];years=[];trades=[];checks={};curves={};same={}
    for k in (5,10,20):
        for v,(tp,sl) in VARIANTS.items():
            name=f'top{k}_{v}';d=OUT/'accounts'/name;d.mkdir(parents=True,exist_ok=True)
            if not (d/'complete.json').exists():
                a=engine().Account(k,profit=tp,stop=sl)
                for step,day in enumerate(days):
                    prev=calendar[calendar.get_loc(day)-1];r=pred[day];assert r.datetime.eq(prev).all()
                    a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],
                           {c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
                a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
                assert all(not e['half_mode'] for e in a.episodes+opened)
                assert all(e['full'] for e in a.events if e.get('event')=='intent')
                for n,data in dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,episodes=a.episodes,open_episodes=opened).items():
                    pd.DataFrame(data).to_parquet(d/f'{n}.parquet',index=False)
                save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
            done=read(d/'complete.json');assert done['protocol_sha256']==digest(OUT/'protocol.json');fees.verify(done['artifacts'])
            daily,orders,held,closed,opened=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes')]
            if k!=5:
                for n in ('daily','orders','holdings','episodes','open_episodes','events'):
                    pd.testing.assert_frame_equal(pd.read_parquet(d/f'{n}.parquet'),pd.read_parquet(fees.OUT/'accounts'/name/f'{n}.parquet'))
                same[name]=True
            for f in (daily,orders,held):f['date']=pd.to_datetime(f.date)
            assert pd.DatetimeIndex(daily.date).equals(benchmark.index)
            checks[name]=fees.ledger_audit(daily,orders,held,closed,opened,k)
            info=dict(top_k=k,variant=v);rows.append(dict(**info,**metrics(daily,benchmark)))
            for y,g in daily.groupby(daily.date.dt.year):years.append(dict(**info,year=int(y),**metrics(g,benchmark)))
            trades.append(dict(**info,**trade_metrics(closed,opened),fills=len(orders),mean_cycle_return=float(closed.return_on_cost.mean())))
            assert trades[-1]['half_episodes']==0
            curves[name]=daily
            print(f'COMPLETE {name}: NAV {daily.nav.iloc[-1]:.2f}',flush=True)
    table=pd.DataFrame(rows);annual=pd.DataFrame(years)
    for n,t in dict(comparison=table,yearly=annual,trade_summary=pd.DataFrame(trades)).items():t.to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
    for name,daily in curves.items():
        nav=daily.nav.to_numpy()/1e6;axes[0].plot(daily.date,nav,label=name)
        axes[1].plot(daily.date,(nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1)*100,label=name)
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Net wealth');axes[1].set_ylabel('Drawdown (%)')
    fig.tight_layout();fig.savefig(OUT/'nav_drawdown.png',dpi=150);plt.close(fig)
    show=table[['top_k','variant','total_return','cagr','max_drawdown','fees','max_stock_weight']].copy()
    for col in ('total_return','cagr','max_drawdown','max_stock_weight'):show[col]=show[col].map(lambda x:f'{x:.2%}')
    show.fees=show.fees.map(lambda x:f'{x:,.2f}')
    yearshow=annual.pivot(index=['top_k','variant'],columns='year',values='total_return').map(lambda x:f'{x:.2%}')
    page='<html lang="zh"><meta charset="utf-8"><title>Top5前两名50%与Top10/20全退出</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:35px auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>Top5前两名50%：全退出止盈止损对照</h1><p>2022-01-04至2025-12-31，初始100万元。沿用16个季度模型，不训练、不使用2026数据、不更改模拟盘。</p>'
    page+='<p>每日收盘判断、次日理想开盘执行。排名超过50退出，新进入前2触发替换；风险清仓后D+3可再买。全部取消减半。Top5新开仓按入场时在持仓中的排名，前2只各占净值23.75%，其余各15.8333%；Top10/20新开仓按95%/K。保留股权重漂移，不每日配平、不因排名升降重配。因此50%是入场分配规则，并非全程维持实际前两只50%。</p>'
    page+='<p>交易费用采用富途公开模拟盘费表，尚未核实证券组合费未计入、未另加滑点；不是精确复刻模拟账户账单。亏损预算仍计入买卖费用，可能略早于简单价格跌幅触发；跳空可能使成交损失超过阈值。</p>'
    page+=show.to_html(index=False)+'<h2>逐年净收益</h2>'+yearshow.to_html()+'<img src="nav_drawdown.png"><h2>完整持仓周期</h2>'+pd.DataFrame(trades).round(5).to_html(index=False)
    page+='<p>Top10/20四个账户已完整重跑，订单、持仓、周期和每日净值与上一轮同费用结果逐项一致。此前历史已反复研究，不是独立测试。<a href="comparison.csv">指标</a> · <a href="yearly.csv">逐年</a> · <a href="verification.json">验收</a></p></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    fees.verify(bindings);fees.verify(p['protected'])
    save(OUT/'verification.json',dict(status='passed',checks=checks,top10_top20_reproduction=same,no_halves=True,sources_and_live_unchanged=True))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json','report_manifest.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',accounts=6,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(show.to_string(index=False),flush=True);print(yearshow.to_string(),flush=True)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
