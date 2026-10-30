"""Bounded add-on: rank80/strong2 and TP10/SL4, TP8/SL3, no new fitting."""
import os,sys,types
from dataclasses import replace
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
from paper_futu.top5_buffer_data import ROOT,read,save,digest
from paper_futu import research_top10_rank65_cooldown as old
from paper_futu import research_top10_rank80_strong1 as prior
from paper_futu import research_rank80_2026 as recent
from paper_futu import research_rank_buffer_futu_fees as fees
OUT=ROOT/'reports/experiments/top10_rank80_two_thresholds_asof_20261013'
THRESHOLDS={'10_5':(.10,.05),'8_4':(.08,.04),'6_2':(.06,.02),'10_4':(.10,.04),'8_3':(.08,.03)}
CONFIGS=[(rank,strong,mode,t) for rank,strong in ((65,2),(80,2),(80,1)) for mode in ('full','staged')
         for t in THRESHOLDS if not (rank==65 and t=='6_2')]

def engine(mode,strong):
    if mode not in ('full','staged') or strong not in (1,2):raise ValueError('Unexpected rule')
    name=f'_rank80_thresholds_{mode}_{strong}'
    if name in sys.modules:return sys.modules[name]
    path=ROOT/'paper_futu'/('rank_buffer_full_exit_engine.py' if mode=='full' else 'rank_buffer_10_20_engine.py')
    code=path.read_text(encoding='utf-8')
    edits={'self.cooldown[ep.code] = step+3':'self.cooldown[ep.code] = step+cooldown_delay(reason)'}
    if strong==1:edits.update({'for i in (2, self.rules.top_k, self.rules.rank_exit):':'for i in (1, self.rules.top_k, self.rules.rank_exit):',
        'ordered[:2]':'ordered[:1]','self.previous_ranks.get(c, math.inf) > 2':'self.previous_ranks.get(c, math.inf) > 1','replacements < 2':'replacements < 1'})
    if mode=='full':edits['((.10,.05),(.08,.04))']='((.10,.05),(.08,.04),(.06,.02),(.10,.04),(.08,.03))'
    for before,after in edits.items():assert code.count(before)==1;code=code.replace(before,after)
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(code,str(path),'exec'),module.__dict__)
    module.fee=fees.fee;module.buy_quantity=fees.buy_quantity;module.liquidation_breakeven=fees.liquidation_breakeven
    module.cooldown_delay=old.cooldown_delay
    return module

def make_account(rank,strong,mode,threshold):
    if (rank,strong,mode,threshold) not in CONFIGS:raise ValueError('Out of scope')
    tp,sl=THRESHOLDS[threshold];m=engine(mode,strong)
    a=m.Account(10,profit=tp,stop=sl) if mode=='full' else m.Account(10)
    a.rules=replace(a.rules,rank_exit=rank,profit=tp,loss_budget=sl,
        loss_half=0 if mode=='full' else .8*sl,trailing_drawdown=0 if mode=='full' else sl)
    return a

def existing(period,cfg):
    rank,strong,mode,t=cfg
    if t in ('10_4','8_3') or (rank,strong)==(80,2):return None
    if period=='2026':return recent.OUT/'accounts'/f'r{rank}_s{strong}_{mode}_{t}'
    return (old.OUT if rank==65 else prior.OUT)/'accounts'/f'top10_{mode}_{t}'

def run():
    import gc
    import numpy as np
    import pandas as pd
    from paper_futu.top5_buffer_report import metrics,trade_metrics
    sources={};protected={}
    for folder in (old.OUT,prior.OUT,recent.OUT):
        c=read(folder/'completion.json');assert c['status']=='complete' and c['manifest_sha256']==digest(folder/'report_manifest.json')
        manifest=read(folder/'report_manifest.json')['artifacts'];fees.verify(manifest);sources.update(manifest)
        p=read(folder/'protocol.json');protected.update(p['protected'])
    for f in ('research_rank80_two_thresholds.py','rank_buffer_full_exit_engine.py','rank_buffer_10_20_engine.py','research_rank_buffer_futu_fees.py','top5_buffer_report.py'):
        path=ROOT/'paper_futu'/f;sources[str(path)]=digest(path)
    for d in (fees.SOURCE/'data',recent.OUT/'data'):
        for f in ('market.parquet','predictions.parquet','benchmark.parquet'):sources[str(d/f)]=digest(d/f)
    protocol=dict(configs=[list(c) for c in CONFIGS],accounts=56,new_accounts=36,reused_accounts=20,sources=sources,protected=protected,
        periods={'2022_2025':['2022-01-04','2025-12-31'],'2026':['2026-01-05','2026-10-13']},
        unchanged='H10 quarterly QREG models; 1000000 cash at each period start; Top10 9.5% entry, drift/no topups; TP cooldown2 SL3; close trigger next ideal open.',
        staged='SL*0.8 first half when rank<=10, SL whole-episode fee-inclusive budget; TP first half if rank<=10; profit half trailing drawdown SL and next TP from half fill; max one half.',
        fee_policy=read(prior.OUT/'protocol.json')['fee_policy'],no_new_training=True,no_live_changes=True)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    fees.verify(protected);records=[];annual=[];checks={};newcount=0;reused=0
    for period,(start,end) in protocol['periods'].items():
        datadir=(fees.SOURCE if period=='2022_2025' else recent.OUT)/'data'
        calpath=(fees.SOURCE/'data/provider/calendars/day.txt') if period=='2022_2025' else (recent.SOURCE/'provider/calendars/day.txt')
        cal=pd.DatetimeIndex(pd.to_datetime(calpath.read_text().splitlines()));days=cal[(cal>=start)&(cal<=end)]
        panel=pd.read_parquet(datadir/'market.parquet').reset_index()
        market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')};del panel
        pred=dict(tuple(pd.read_parquet(datadir/'predictions.parquet').reset_index().groupby('execution_date')))
        benchmark=pd.read_parquet(datadir/'benchmark.parquet').benchmark_return
        for cfg in CONFIGS:
            rank,strong,mode,t=cfg;name=f'r{rank}_s{strong}_{mode}_{t}';source=existing(period,cfg)
            d=source or OUT/'accounts'/period/name
            if source is not None:reused+=1
            else:
                newcount+=1;d.mkdir(parents=True,exist_ok=True)
                if not (d/'complete.json').exists():
                    a=make_account(*cfg)
                    for step,day in enumerate(days):
                        prev=cal[cal.get_loc(day)-1];r=pred[day];assert r.datetime.eq(prev).all()
                        assert (r.model_cutoff<=r.datetime).all()
                        a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],
                               {c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
                    a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
                    if mode=='full':assert all(not r['half_mode'] for r in a.episodes+opened)
                    for n,value in dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,episodes=a.episodes,open_episodes=opened).items():
                        pd.DataFrame(value).to_parquet(d/f'{n}.parquet',index=False)
                    save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
                assert read(d/'complete.json')['protocol_sha256']==digest(OUT/'protocol.json')
            fees.verify(read(d/'complete.json')['artifacts'])
            daily,orders,held,closed,opened=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes')]
            for f in (daily,orders,held):f['date']=pd.to_datetime(f.date)
            assert pd.DatetimeIndex(daily.date).equals(days) and not daily.date.duplicated().any()
            key=period+'_'+name;checks[key]=fees.ledger_audit(daily,orders,held,closed,opened,10)
            checks[key]['reentry_checks']=old.audit_cooldown(orders,closed,cal)
            info=dict(period=period,name=name,rank_exit=rank,strong_rank=strong,mode=mode,threshold=t,reused=source is not None)
            records.append(dict(**info,**metrics(daily,benchmark),**trade_metrics(closed,opened),fills=len(orders),
                mean_holding_days=float(closed.holding_days.mean()),net_profit=float(daily.nav.iloc[-1]-1e6)))
            for year,g in daily.groupby(daily.date.dt.year):
                opening=float(g.nav.iloc[0]/(1+g.net_return.iloc[0]))
                annual.append(dict(**info,year=int(year),**metrics(g,benchmark),
                    fees_over_opening_nav=float(g.fees.sum()/opening),fills=int(orders.date.dt.year.eq(year).sum())))
            print('COMPLETE',period,name,'reused' if source else 'new',round(daily.nav.iloc[-1],2),flush=True)
        del market,pred;gc.collect()
    assert newcount==36 and reused==20 and len(checks)==56
    table=pd.DataFrame(records);years=pd.DataFrame(annual)
    assert len(years)==140 and not years.duplicated(['name','year']).any()
    table.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig');years.to_csv(OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    # Paired deltas change only the named rule; never rank the mixed intervals as a continuous account.
    pairs=[]
    def pair(kind,left,right):
        a=years[years.name.eq(left)].set_index('year');b=years[years.name.eq(right)].set_index('year')
        assert a.index.equals(b.index)
        for year in a.index:
            pairs.append(dict(kind=kind,base=left,changed=right,year=int(year),
                return_delta=float(b.loc[year,'total_return']-a.loc[year,'total_return']),
                drawdown_delta=float(b.loc[year,'max_drawdown']-a.loc[year,'max_drawdown']),
                fee_rate_delta=float(b.loc[year,'fees_over_opening_nav']-a.loc[year,'fees_over_opening_nav'])))
    for mode in ('full','staged'):
        for t in THRESHOLDS:
            if t!='6_2':pair('rank65_to80',f'r65_s2_{mode}_{t}',f'r80_s2_{mode}_{t}')
            pair('strong2_to1',f'r80_s2_{mode}_{t}',f'r80_s1_{mode}_{t}')
        for rank,strong in ((65,2),(80,2),(80,1)):
            for before,after in (('10_5','10_4'),('8_4','8_3'),('8_4','10_4')):
                pair('threshold_'+before+'_to_'+after,f'r{rank}_s{strong}_{mode}_{before}',f'r{rank}_s{strong}_{mode}_{after}')
    pd.DataFrame(pairs).to_csv(OUT/'paired_deltas.csv',index=False,encoding='utf-8-sig')
    show=[]
    for cfg in CONFIGS:
        rank,strong,mode,t=cfg;name=f'r{rank}_s{strong}_{mode}_{t}';g=years[years.name.eq(name)].set_index('year')
        show.append({'排名':f'{rank}/{strong}','退出':'直接清仓' if mode=='full' else '分段退出','止盈/止损':t.replace('_','% / ')+'%'}|
                    {str(y)+('*' if y==2026 else ''):f'{g.loc[y,"total_return"]:.2%} / {g.loc[y,"max_drawdown"]:.2%}' for y in range(2022,2027)})
    display=pd.DataFrame(show)
    body='<html lang="zh"><meta charset="utf-8"><title>80/2与10/4、8/3消融</title><style>body{font:16px/1.7 system-ui;max-width:1500px;margin:30px auto;padding:0 15px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}thead{background:#eef3fa}.scroll{overflow-x:auto}</style><h1>Top10：80/2与10%/4%、8%/3%补充消融</h1>'
    body+='<p>每格为扣费后收益 / 年内最大回撤。2022—2025连续账户，2026*截至10月13日另以100万元现金起跑；不是连续五年净值。28配置共56个区间账户，复用20个、新回放36个，未重训。</p>'
    body+='<div class="scroll">'+display.to_html(index=False)+'</div>'
    body+='<p>80/2：跌至81名排名退出，新进前2名触发强势替换。比较65/2与80/2只改变退出名次，比较80/2与80/1只改变强势入场规则。强势条件不限制空位正常补仓。</p><p>直接清仓触及时全卖；分段版仍在前10时，盈利达到目标或亏损达到止损幅度80%首次减半。10/4、8/3的亏损减半线为3.2%、2.4%，盈利减半后跟踪幅度为4%、3%，第二次止盈仍从减半成交价按10%、8%计算。含费整笔预算优先、跳空可能突破止损，减半后不加仓。止盈冷却2日、止损3日。</p>'
    body+='<p>沿用H10季度模型与固定费用假设，日线理想次日开盘成交，无额外滑点。2026季度拟合允许使用当时已成熟标签；历史已反复研究，不是独立测试。H00300在2026缺开盘值，基准从上年末收盘起算，与策略首次开盘建仓有隔夜差异。不修改模拟盘。</p><p><a href="comparison.csv">全期与交易指标</a> · <a href="yearly.csv">逐年指标</a> · <a href="paired_deltas.csv">配对消融差异</a> · <a href="verification.json">验收</a></p></html>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    fees.verify(sources);fees.verify(protected)
    save(OUT/'verification.json',dict(status='passed',new_accounts=newcount,reused_accounts=reused,checks=checks,no_training=True,no_live_change=True))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json','report_manifest.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',accounts=56,new_accounts=36,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(display.to_string(index=False),flush=True)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
