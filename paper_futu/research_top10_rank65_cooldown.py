"""Top10 rank65 with TP D+2 / SL D+3; four confirmed exit policies."""
import sys
import types
from dataclasses import replace
from paper_futu import research_top5_head2_full_exit as full
from paper_futu import research_top5_10_20_staged as staged
from paper_futu import research_rank_buffer_futu_fees as fees
from paper_futu.top5_buffer_data import ROOT,START,END,read,save,digest

OUT=ROOT/'reports/experiments/top10_rank65_tp2_sl3_asof_20251231'
PROFIT_REASONS={'profit_full','profit_half','second_profit','profit_after_loss_half'}

def cooldown_delay(reason):
    if reason in ('rank_exit','strong_replacement'):return 0
    return 2 if reason in PROFIT_REASONS else 3

def make_account(mode,tp,sl):
    if mode not in ('full','staged') or (tp,sl) not in ((.10,.05),(.08,.04)):raise ValueError('Unconfirmed setting')
    name=f'_rank65_cooldown_{mode}'
    if name not in sys.modules:
        path=ROOT/'paper_futu'/('rank_buffer_full_exit_engine.py' if mode=='full' else 'rank_buffer_10_20_engine.py')
        code=path.read_text(encoding='utf-8');old='self.cooldown[ep.code] = step+3'
        assert code.count(old)==1;code=code.replace(old,'self.cooldown[ep.code] = step+cooldown_delay(reason)')
        m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
        exec(compile(code,str(path),'exec'),m.__dict__)
        m.fee=fees.fee;m.buy_quantity=fees.buy_quantity;m.liquidation_breakeven=fees.liquidation_breakeven;m.cooldown_delay=cooldown_delay
    m=sys.modules[name]
    a=m.Account(10,profit=tp,stop=sl) if mode=='full' else m.Account(10)
    a.rules=replace(a.rules,rank_exit=65,profit=tp,loss_budget=sl,loss_half=0 if mode=='full' else sl*.8,trailing_drawdown=0 if mode=='full' else sl)
    return a

def audit_cooldown(orders,closed,calendar):
    import pandas as pd
    buys=orders[orders.direction.eq('BUY')]
    checked=0
    for e in closed.itertuples():
        exit_day=pd.Timestamp(e.exit_day)
        later=buys[(buys.code==e.code)&(buys.date>=exit_day)]
        if len(later):
            next_day=later.date.min();gap=calendar.get_loc(next_day)-calendar.get_loc(exit_day)
            assert gap>=max(1,cooldown_delay(e.exit_reason))
            checked+=1
    return checked

def run():
    import json
    import numpy as np
    import pandas as pd
    from paper_futu.top5_buffer_report import metrics,trade_metrics
    sources={};protected={}
    for base in (full.OUT,staged.OUT):
        c=read(base/'completion.json');assert c['status']=='complete' and c['manifest_sha256']==digest(base/'report_manifest.json')
        p=read(base/'protocol.json');sources.update(p['source']);sources.update(read(base/'report_manifest.json')['artifacts']);protected.update(p['protected'])
    sources[__file__]=digest(__file__)
    fees.verify(sources);fees.verify(protected)
    protocol=dict(start=START,end=END,top_k=10,rank_exit=65,profit_cooldown=2,loss_cooldown=3,
        profit_reasons=sorted(PROFIT_REASONS),profit_trailing_cooldown=3,rank_and_strong_no_extra_cooldown=True,
        half_exit_no_cooldown=True,variants=['full_10_5','full_8_4','staged_10_5','staged_8_4'],
        entry_weight=.095,sources=sources,protected=protected,fee_policy=read(full.OUT/'protocol.json')['fee_policy'],
        unchanged='Quarterly H10 S6 predictions; daily close decision / next open; new top2 strong entry max2; drift retained weights, no topups; full episode loss budget; staged scaled rules',
        changes='Rank exit50->65 AND take-profit cooldown3->2 together; cannot attribute difference to one change',
        no_training=True,no_2026=True,workers=1,threads=4)
    protocol=json.loads(json.dumps(protocol));OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    panel=pd.read_parquet(fees.SOURCE/'data/market.parquet').reset_index();assert panel.datetime.max()<=pd.Timestamp(END)
    market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')};del panel
    frame=pd.read_parquet(fees.SOURCE/'data/predictions.parquet').reset_index();assert frame.execution_date.max()<=pd.Timestamp(END)
    pred=dict(tuple(frame.groupby('execution_date')))
    calendar=pd.DatetimeIndex(pd.to_datetime((fees.SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()));assert calendar.max()==pd.Timestamp(END)
    days=calendar[(calendar>=START)&(calendar<=END)];benchmark=pd.read_parquet(fees.SOURCE/'data/benchmark.parquet').benchmark_return
    rows=[];years=[];trades=[];exits=[];checks={};curves={}
    for mode,base in [('full',full.OUT),('staged',staged.OUT)]:
        for threshold,tp,sl in [('10_5',.10,.05),('8_4',.08,.04)]:
            variant=f'{mode}_{threshold}';name=f'top10_{variant}';d=OUT/'accounts'/name;d.mkdir(parents=True,exist_ok=True)
            if not (d/'complete.json').exists():
                a=make_account(mode,tp,sl)
                for step,day in enumerate(days):
                    prev=calendar[calendar.get_loc(day)-1];r=pred[day];assert r.datetime.eq(prev).all()
                    a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],{c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
                a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
                if mode=='full':assert all(not x['half_mode'] for x in a.episodes+opened)
                for n,r in dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,episodes=a.episodes,open_episodes=opened).items():pd.DataFrame(r).to_parquet(d/f'{n}.parquet',index=False)
                save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
            c=read(d/'complete.json');assert c['protocol_sha256']==digest(OUT/'protocol.json');fees.verify(c['artifacts'])
            daily,orders,held,closed,opened=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes')]
            for f in (daily,orders,held):f['date']=pd.to_datetime(f.date)
            assert pd.DatetimeIndex(daily.date).equals(benchmark.index)
            check=fees.ledger_audit(daily,orders,held,closed,opened,10);check['reentries_checked']=audit_cooldown(orders,closed,calendar);checks[name]=check
            olddaily=pd.read_parquet(base/'accounts'/name/'daily.parquet');oldmetrics=metrics(olddaily,benchmark)
            new=metrics(daily,benchmark);info=dict(variant=variant)
            rows.append(dict(**info,**new,old_cagr=oldmetrics['cagr'],old_max_drawdown=oldmetrics['max_drawdown'],old_fees=oldmetrics['fees'],cagr_delta=new['cagr']-oldmetrics['cagr']))
            for y,g in daily.groupby(daily.date.dt.year):years.append(dict(**info,year=int(y),**metrics(g,benchmark)))
            trades.append(dict(**info,**trade_metrics(closed,opened),mean_holding_days=float(closed.holding_days.mean()),median_holding_days=float(closed.holding_days.median()),fills=len(orders)))
            for reason,g in closed.groupby('exit_reason'):exits.append(dict(**info,reason=reason,count=len(g),share=len(g)/len(closed)))
            curves[variant]=(daily,olddaily)
            print(f'COMPLETE {name} NAV={daily.nav.iloc[-1]:.2f}',flush=True)
    table=pd.DataFrame(rows);annual=pd.DataFrame(years)
    for n,t in dict(comparison=table,yearly=annual,trade_summary=pd.DataFrame(trades),exit_summary=pd.DataFrame(exits)).items():t.to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,(name,(new,old)) in zip(axes.flat,curves.items()):
        ax.plot(new.date,new.nav/1e6,label='Rank65 / TP D+2');ax.plot(pd.to_datetime(old.date),old.nav/1e6,label='Rank50 / TP D+3')
        ax.set_title(name);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'paired_nav.png',dpi=150);plt.close(fig)
    show=table[['variant','total_return','cagr','old_cagr','max_drawdown','old_max_drawdown','fees']].copy()
    for col in ('total_return','cagr','old_cagr','max_drawdown','old_max_drawdown'):show[col]=show[col].map(lambda x:f'{x:.2%}')
    show.fees=show.fees.map(lambda x:f'{x:,.2f}')
    yr=annual.pivot(index='variant',columns='year',values='total_return').map(lambda x:f'{x:.2%}')
    page='<html lang="zh"><meta charset="utf-8"><title>Top10排名65与差异冷却</title><style>body{font:16px/1.7 system-ui;max-width:1300px;margin:35px auto}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%}img{max-width:100%}</style><h1>Top10：排名退出65、止盈冷却2日、止损3日</h1><p>2022—2025，初始100万元，S6季度重训H10预测不变。Top10入场等权、95%股票预算，保留仓权重漂移、不补回减半仓位。</p><p>排名第65保留，第66起退出；新入前2强势替换规则不变。止盈最终清仓日D后最早D+2开盘可买回，止损最早D+3。盈利跟踪止损归入止损3日；因整手不能拆分的止盈减半实际全清也归止盈2日。单纯减半不设冷却。排名退出/强势替换无额外冷却，但禁止同日买回。</p>'
    page+=show.to_html(index=False)+'<h2>逐年净收益</h2>'+yr.to_html()+'<img src="paired_nav.png"><h2>持有期与成交</h2>'+pd.DataFrame(trades).round(5).to_html(index=False)+'<h2>最终退出原因</h2>'+pd.DataFrame(exits).round(5).to_html(index=False)
    page+='<p>本次同时修改排名门槛和止盈冷却，结果是联合效果，不能单独归因。交易费用沿用富途公开费表，未核实组合费不计、未额外加滑点。收盘判断次日理想开盘成交，历史不是独立测试。不修改模拟盘、模型和冻结源文件。</p><p><a href="comparison.csv">对照</a> · <a href="yearly.csv">逐年</a> · <a href="verification.json">验收</a></p></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    fees.verify(sources);fees.verify(protected)
    save(OUT/'verification.json',dict(status='passed',checks=checks,source_and_live_unchanged=True,no_training=True,no_2026=True))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json','report_manifest.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',accounts=4,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(show.to_string(index=False),flush=True);print(yr.to_string(),flush=True)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
