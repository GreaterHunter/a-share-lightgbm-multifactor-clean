"""Rank80/strong1 and TP6/SL2 offline comparisons, unchanged H10 signals."""
import sys
import types
from dataclasses import replace
from paper_futu import research_top10_rank65_cooldown as base
from paper_futu import research_rank_buffer_futu_fees as fees
from paper_futu.top5_buffer_data import ROOT,START,END,read,save,digest

OUT=ROOT/'reports/experiments/top10_rank80_strong1_asof_20251231'
THRESHOLDS={'10_5':(.10,.05),'8_4':(.08,.04),'6_2':(.06,.02)}

def engine(mode):
    if mode not in ('full','staged'):raise ValueError(mode)
    name=f'_rank80_strong1_{mode}'
    if name in sys.modules:return sys.modules[name]
    path=ROOT/'paper_futu'/('rank_buffer_full_exit_engine.py' if mode=='full' else 'rank_buffer_10_20_engine.py')
    code=path.read_text(encoding='utf-8')
    edits={'self.cooldown[ep.code] = step+3':'self.cooldown[ep.code] = step+cooldown_delay(reason)',
        'for i in (2, self.rules.top_k, self.rules.rank_exit):':'for i in (1, self.rules.top_k, self.rules.rank_exit):',
        'ordered[:2]':'ordered[:1]','self.previous_ranks.get(c, math.inf) > 2':'self.previous_ranks.get(c, math.inf) > 1',
        'replacements < 2':'replacements < 1'}
    if mode=='full':edits['((.10,.05),(.08,.04))']='((.10,.05),(.08,.04),(.06,.02))'
    for old,new in edits.items():
        assert code.count(old)==1;code=code.replace(old,new)
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(code,str(path),'exec'),m.__dict__)
    m.fee=fees.fee;m.buy_quantity=fees.buy_quantity;m.liquidation_breakeven=fees.liquidation_breakeven
    m.cooldown_delay=base.cooldown_delay
    return m

def make_account(mode,threshold):
    tp,sl=THRESHOLDS[threshold];m=engine(mode)
    a=m.Account(10,profit=tp,stop=sl) if mode=='full' else m.Account(10)
    a.rules=replace(a.rules,rank_exit=80,profit=tp,loss_budget=sl,loss_half=0 if mode=='full' else sl*.8,trailing_drawdown=0 if mode=='full' else sl)
    return a

def run():
    import pandas as pd
    from paper_futu.top5_buffer_report import metrics,trade_metrics
    c=read(base.OUT/'completion.json');assert c['status']=='complete' and c['manifest_sha256']==digest(base.OUT/'report_manifest.json')
    p=read(base.OUT/'protocol.json')
    sources={**p['sources'],**read(base.OUT/'report_manifest.json')['artifacts'],
        str(ROOT/'paper_futu/research_top10_rank65_cooldown.py'):digest(ROOT/'paper_futu/research_top10_rank65_cooldown.py'),__file__:digest(__file__)}
    fees.verify(sources);fees.verify(p['protected'])
    protocol=dict(start=START,end=END,sources=sources,protected=p['protected'],fee_policy=p['fee_policy'],
        rank_exit=80,strong='Previously outside rank1, now rank1 and unheld; strictly higher score; at most one strong replacement daily',
        thresholds={k:list(v) for k,v in THRESHOLDS.items()},top_k=10,entry_weight=.095,
        cooldown='TP full exit D+2; SL including profit trailing D+3; ranking and strong replacement no extra cooldown; half no cooldown',
        staged='First half if still top10: TP or 0.8*SL. After profit half trailing drawdown SL, second TP from half fill. Whole-episode budget unchanged.',
        model='Original H10 quarterly QREG 16 pre2026 models; no fit',
        unchanged='Daily close decision, next ideal open; weights drift; no topups; lots/T+1/fee-inclusive budget/cash and unfilled intent rules preserved',
        limits='Only six new accounts; original rank65/strong2 four accounts reused. No extra grid.',workers=1,threads=4)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    panel=pd.read_parquet(fees.SOURCE/'data/market.parquet').reset_index();assert panel.datetime.max()<=pd.Timestamp(END)
    market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')};del panel
    frame=pd.read_parquet(fees.SOURCE/'data/predictions.parquet').reset_index();assert frame.execution_date.max()<=pd.Timestamp(END)
    pred=dict(tuple(frame.groupby('execution_date')))
    cal=pd.DatetimeIndex(pd.to_datetime((fees.SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()));assert cal.max()==pd.Timestamp(END)
    days=cal[(cal>=START)&(cal<=END)];benchmark=pd.read_parquet(fees.SOURCE/'data/benchmark.parquet').benchmark_return
    bindings=[]
    for mode in ('full','staged'):
        for threshold in THRESHOLDS:
            name=f'top10_{mode}_{threshold}';d=OUT/'accounts'/name;d.mkdir(parents=True,exist_ok=True)
            if not (d/'complete.json').exists():
                a=make_account(mode,threshold)
                for step,day in enumerate(days):
                    prev=cal[cal.get_loc(day)-1];r=pred[day];assert r.datetime.eq(prev).all()
                    a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],{c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
                a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
                if mode=='full':assert all(not e['half_mode'] for e in a.episodes+opened)
                for n,r in dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,episodes=a.episodes,open_episodes=opened).items():pd.DataFrame(r).to_parquet(d/f'{n}.parquet',index=False)
                save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
                print(f'COMPLETE {name} NAV {a.daily[-1]["nav"]:.2f}',flush=True)
            assert read(d/'complete.json')['protocol_sha256']==digest(OUT/'protocol.json')
            bindings.append((80,1,mode,threshold,d))
            if threshold!='6_2':bindings.append((65,2,mode,threshold,base.OUT/'accounts'/name))
    rows=[];years=[];exits=[];checks={};curves={}
    for rank,strong,mode,threshold,d in bindings:
        fees.verify(read(d/'complete.json')['artifacts'])
        daily,orders,held,closed,opened,events=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes','events')]
        for f in (daily,orders,held):f['date']=pd.to_datetime(f.date)
        assert pd.DatetimeIndex(daily.date).equals(benchmark.index)
        name=f'r{rank}_s{strong}_{mode}_{threshold}';check=fees.ledger_audit(daily,orders,held,closed,opened,10)
        check['reentries_checked']=base.audit_cooldown(orders,closed,cal);checks[name]=check
        if rank==80:
            incoming=orders[orders.direction.eq('BUY') & orders.reason.eq('strong_entry')]
            for o in incoming.itertuples():
                r=pred[o.date];scores=dict(zip(r.instrument,r.prediction));ordered,ranks=engine(mode).ranking(scores)
                assert ranks[o.code]==1
                previous_execution=cal[cal.get_loc(o.date)-1]
                if previous_execution in pred:
                    past=pred[previous_execution];_,past_ranks=engine(mode).ranking(dict(zip(past.instrument,past.prediction)))
                    assert past_ranks.get(o.code,float('inf'))>1
            strong_sells=orders[orders.reason.eq('strong_replacement')]
            if len(strong_sells):assert strong_sells.groupby('date').size().max()<=1
        info=dict(rank_exit=rank,strong_rank=strong,mode=mode,threshold=threshold)
        rows.append(dict(**info,**metrics(daily,benchmark),**trade_metrics(closed,opened),
            fills=len(orders),buy_fills=int(orders.direction.eq('BUY').sum()),sell_fills=int(orders.direction.eq('SELL').sum()),
            mean_holding_days=float(closed.holding_days.mean()),median_holding_days=float(closed.holding_days.median()),
            mean_cycle_return=float(closed.return_on_cost.mean()),net_profit=float(daily.nav.iloc[-1]-1e6)))
        for y,g in daily.groupby(daily.date.dt.year):
            years.append(dict(**info,year=int(y),**metrics(g,benchmark),fills=int(orders.date.dt.year.eq(y).sum())))
        for reason,g in closed.groupby('exit_reason'):exits.append(dict(**info,reason=reason,count=len(g),share=len(g)/len(closed)))
        curves[name]=daily
    table=pd.DataFrame(rows).sort_values(['mode','threshold','rank_exit']);annual=pd.DataFrame(years)
    for n,t in dict(comparison=table,yearly=annual,exit_summary=pd.DataFrame(exits)).items():t.to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
    for mode,ax in zip(('full','staged'),axes):
        for name,daily in curves.items():
            if f'_{mode}_' in name:ax.plot(daily.date,daily.nav/1e6,label=name)
        ax.set_title(mode);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'nav_comparison.png',dpi=150);plt.close(fig)
    show=table[['rank_exit','strong_rank','mode','threshold','cagr','max_drawdown','total_return','closed','fills','mean_holding_days','fees','win_rate','avg_gain','avg_loss']].copy()
    for c in ('cagr','max_drawdown','total_return','win_rate','avg_gain','avg_loss'):show[c]=show[c].map(lambda x:f'{x:.2%}')
    show=show.round(2)
    yearly=annual[annual.rank_exit==80].pivot(index=['mode','threshold'],columns='year',values='total_return').map(lambda x:f'{x:.2%}')
    page='<html lang="zh"><meta charset="utf-8"><title>Top10排名80强势第1与6/2</title><style>body{font:16px/1.7 system-ui;max-width:1500px;margin:35px auto}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}.scroll{overflow-x:auto}</style><h1>Top10：排名80退出、强势仅第1名，新增6%／2%</h1><p>2022—2025，初始100万元，沿用H10季度模型，无重新训练。前80保留、81起退出；未持有股票从前1之外进入第1才触发强势替换，每日最多因此换1只。普通空位仍按排名补位，不要求所有新买入均为第1。</p><p>止盈清仓D+2、止损与盈利跟踪止损D+3。新开仓账户9.5%，保留股漂移不配平。所有条件收盘判断，次日理想开盘执行，2%止损不是成交损失上限。含费累计亏损预算优先。</p><p>6%／2%分段版：尚未减半且前10，涨6%或跌1.6%减半，否则相应全清/等待预算。盈利减半后跟踪回撤2%，第二次止盈为减半成交价再涨6%；亏损减半后保留原6%止盈。最多减半一次、不补回。</p>'
    page+='<h2>扣费后表现、交易次数与交易质量</h2><div class="scroll">'+show.to_html(index=False)+'</div><p>closed为完整持仓周期数；fills为买卖成交笔数，减半会增加成交笔数但不增加完整周期数。mean_holding_days只统计已退出周期。费用已扣，不能再次扣除。</p><h2>新规则逐年净收益</h2>'+yearly.to_html()+'<img src="nav_comparison.png">'
    page+='<p>排名80+强势第1作为联合修改，未拆分两者单独影响；6/2与8/4、10/5在相同新排名规则下比较。交易次数增加只有在新增交易扣费后仍有正贡献时才有价值，本表不把次数当优化目标。</p><p>费率沿用富途公开模拟交易费表，未核实组合费不计、无额外滑点。历史数据已反复研究，不是独立测试；不修改S6模拟盘。<a href="comparison.csv">指标</a> · <a href="yearly.csv">逐年</a> · <a href="exit_summary.csv">退出原因</a> · <a href="verification.json">验收</a></p></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    fees.verify(sources);fees.verify(p['protected'])
    save(OUT/'verification.json',dict(status='passed',checks=checks,accounts=10,new_accounts=6,no_training=True,no_2026=True,sources_and_live_unchanged=True))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json','report_manifest.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',accounts=10,new_accounts=6,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(show.to_string(index=False),flush=True);print(yearly.to_string(),flush=True)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
