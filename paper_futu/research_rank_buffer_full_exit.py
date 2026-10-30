"""Top10/20 full-exit price-rule ablation, strictly bounded at 2025-12-31."""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
import html
import os
from pathlib import Path
import time
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'

from paper_futu.top5_buffer_data import ROOT,START,END,read,save,digest
from paper_futu.rank_buffer_full_exit_engine import Account

SOURCE=ROOT/'reports/experiments/top5_rank_buffer_asof_20251231'
BASE=ROOT/'reports/experiments/rank_buffer_top10_top20_asof_20251231'
OUT=ROOT/'reports/experiments/rank_buffer_full_exit_asof_20251231'
VARIANTS={'full_10_5':(.10,.05),'full_8_4':(.08,.04)}
LABELS={'staged_10_5':'原减半10/5','full_10_5':'直接清仓10/5','full_8_4':'直接清仓8/4'}


def verify(files):
    for p,h in files.items():
        if digest(p)!=h:raise ValueError(f'Changed artifact: {p}')


def implementation():
    names=('paper_futu/research_rank_buffer_full_exit.py','paper_futu/rank_buffer_full_exit_engine.py',
           'paper_futu/rank_buffer_10_20_engine.py','paper_futu/rank_buffer_10_20_report.py',
           'paper_futu/top5_buffer_report.py','paper_futu/top5_buffer_data.py')
    return {str(ROOT/p):digest(ROOT/p) for p in names}


def verify_engine_scope():
    """Only price_action and Account.__init__ may differ from the staged engine."""
    def normalize(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        tree.body=[x for x in tree.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))
                   and not (isinstance(x,ast.FunctionDef) and x.name=='price_action')]
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=='Account':
                node.body=[x for x in node.body if not (isinstance(x,ast.FunctionDef) and x.name=='__init__')]
        return ast.dump(tree,include_attributes=False)
    assert normalize(ROOT/'paper_futu/rank_buffer_10_20_engine.py')==normalize(ROOT/'paper_futu/rank_buffer_full_exit_engine.py')


@contextmanager
def lock():
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def audit():
    import numpy as np
    import pandas as pd
    verify_engine_scope()
    b=read(BASE/'completion.json')
    assert b['status']=='complete' and b['manifest_sha256']==digest(BASE/'report_manifest.json')
    source_p=read(BASE/'protocol.json')
    bindings={**read(BASE/'report_manifest.json')['artifacts'],**source_p['sources'],
              **source_p['implementation'],**read(SOURCE/'audit_complete.json')['artifacts']}
    bindings[str(BASE/'completion.json')]=digest(BASE/'completion.json')
    bindings[str(BASE/'report_manifest.json')]=digest(BASE/'report_manifest.json')
    for k in (10,20):
        for bp in (0,10,20):
            d=BASE/'accounts'/f'top{k}_{bp:02d}bp';c=read(d/'complete.json')
            assert c['protocol_sha256']==digest(BASE/'protocol.json')
            bindings[str(d/'complete.json')]=digest(d/'complete.json');bindings.update(c['artifacts'])
    verify(bindings);verify(source_p['protected'])
    models=read(SOURCE/'models_audit.json')['periods'];assert len(models)==16
    for m in models:
        assert m['label_end_max']<=m['cutoff']<m['start'] and m['end']<=END and m['reload_equal']
    calendar=pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    assert calendar.max()==pd.Timestamp(END)
    pred=pd.read_parquet(SOURCE/'data/predictions.parquet')
    days=calendar[(calendar>=START)&(calendar<=END)];signals=pred.index.get_level_values('datetime')
    pos=calendar.get_indexer(signals)
    assert not pred.index.has_duplicates and (pos>=0).all() and (pos+1<len(calendar)).all()
    assert np.array_equal(calendar.to_numpy()[pos+1],pred.execution_date.to_numpy())
    assert (pred.model_cutoff<=signals).all() and np.isfinite(pred.prediction).all()
    assert set(pred.execution_date)==set(days)
    p=dict(start=START,end=END,top_k=[10,20],variants=VARIANTS,friction_bp=[0,10,20],
        initial_cash=1e6,price_exit='all remaining shares; never half; no trailing or second take-profit',
        signal='completed close, execute following ideal open; NOT intraday barrier fills',
        loss_budget='min(entry NAV * .02, initial fee-inclusive cost * stop); fees included; budget priority retained',
        preserved=['ranking >50 exit','new entry into top2 replacement, maximum2/day','D+3 risk-exit cooldown',
                   '95% stock budget','entry equal weight; retained positions drift, no topups','T+1, lots, cash, limits'],
        rule_sets={f'{k}_{v}':asdict(Account(k,profit=tp,stop=sl).rules) for k in (10,20) for v,(tp,sl) in VARIANTS.items()},
        half_fields_inactive=True,no_training=True,quarters_reused=16,accounts_new=12,baseline_accounts_reused=6,
        workers=1,threads=4,memory_gate_enabled=False,sources=bindings,protected=source_p['protected'],
        implementation=implementation(),limitations=source_p['limitations'])
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==read_json_safe(p),'Protocol changed'
    else:save(OUT/'protocol.json',p)
    save(OUT/'audit.json',dict(status='passed',engine_changes_only_price_action_and_init=True,
         source_verified=True,no_training=True,models=16,prediction_rows=len(pred),end=END))
    print('AUDIT passed: only exit policy changed; 16 causal models reused',flush=True)


def read_json_safe(value):
    import json
    return json.loads(json.dumps(value,default=str))


def protocol():
    p=read(OUT/'protocol.json');assert p['implementation']==implementation()
    verify(p['protected']);return p


def replay(k,variant,bp,market,predictions,days,prior):
    import pandas as pd
    name=f'top{k}_{variant}_{bp:02d}bp';d=OUT/'accounts'/name;done=d/'complete.json'
    if done.exists():
        c=read(done);assert c['protocol_sha256']==digest(OUT/'protocol.json');verify(c['artifacts']);return
    tp,sl=VARIANTS[variant];a=Account(k,bp/10000,profit=tp,stop=sl)
    for step,day in enumerate(days):
        if step%100==0:save(OUT/'progress.json',dict(account=name,step=step,total=len(days),date=str(day.date())))
        rows=predictions[day];prev=prior[day];assert rows.datetime.eq(prev).all()
        a.step(str(day.date()),step,dict(zip(rows.instrument,rows.prediction)),market[day],
               {c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
    checks=a.reconcile()
    opened=a.open_episodes(str(days[-1].date()),len(days)-1)
    assert all(not x['half_mode'] for x in a.episodes+opened)
    assert all(x.get('event')!='half_complete' for x in a.events)
    assert all(x['full'] for x in a.events if x.get('event')=='intent')
    records=dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,episodes=a.episodes,open_episodes=opened)
    d.mkdir(parents=True,exist_ok=True)
    for n,r in records.items():pd.DataFrame(r).to_parquet(d/(n+'.parquet'),index=False)
    save(d/'reconciliation.json',dict(**checks,no_half_exits=True))
    save(done,dict(status='complete',top_k=k,variant=variant,friction_bp=bp,
         protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.iterdir() if f.name!='complete.json'}))
    print(f'COMPLETE {name} NAV={a.daily[-1]["nav"]:.2f} cycles={len(a.episodes)}',flush=True)


def backtest():
    import pandas as pd
    verify(protocol()['sources'])
    panel=pd.read_parquet(SOURCE/'data/market.parquet').reset_index()
    assert panel.datetime.max()<=pd.Timestamp(END)
    market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')}
    del panel
    frame=pd.read_parquet(SOURCE/'data/predictions.parquet').reset_index();predictions=dict(tuple(frame.groupby('execution_date')))
    calendar=pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    days=calendar[(calendar>=START)&(calendar<=END)];prior={d:calendar[calendar.get_loc(d)-1] for d in days}
    for k in (10,20):
        for v in VARIANTS:
            for bp in (0,10,20):replay(k,v,bp,market,predictions,days,prior)


def report():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_futu.rank_buffer_10_20_report import independent_audit
    from paper_futu.top5_buffer_report import metrics,trade_metrics
    p=protocol();benchmark=pd.read_parquet(SOURCE/'data/benchmark.parquet').benchmark_return
    rows=[];years=[];trades=[];exits=[];checks={};fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    for k in (10,20):
        for variant in ('staged_10_5',*VARIANTS):
            for bp in (0,10,20):
                d=BASE/'accounts'/f'top{k}_{bp:02d}bp' if variant=='staged_10_5' else OUT/'accounts'/f'top{k}_{variant}_{bp:02d}bp'
                c=read(d/'complete.json');verify(c['artifacts'])
                expected=BASE/'protocol.json' if variant=='staged_10_5' else OUT/'protocol.json'
                assert c['protocol_sha256']==digest(expected)
                daily,orders,holding,closed,opened,events=[pd.read_parquet(d/(n+'.parquet')) for n in ('daily','orders','holdings','episodes','open_episodes','events')]
                for f in (daily,orders,holding):f['date']=pd.to_datetime(f.date)
                assert pd.DatetimeIndex(daily.date).equals(benchmark.index)
                key=f'{k}_{variant}_{bp}';checks[key]=independent_audit(daily,orders,holding,closed,opened,k,bp/10000)
                info=dict(top_k=k,variant=variant,friction_bp=bp)
                m=metrics(daily,benchmark);rows.append(dict(**info,**m))
                annual=[]
                for year,g in daily.groupby(daily.date.dt.year):
                    row=dict(**info,year=int(year),**metrics(g,benchmark));years.append(row);annual.append(row['total_return'])
                np.testing.assert_allclose(np.prod(1+np.array(annual))-1,m['total_return'],atol=1e-10)
                t=trade_metrics(closed,opened)
                if variant!='staged_10_5':assert t['half_episodes']==0
                trades.append(dict(**info,**t,buy_fills=int(orders.direction.eq('BUY').sum()),sell_fills=int(orders.direction.eq('SELL').sum()),
                    average_cycle_pnl=float(closed.pnl.mean()),average_cycle_return=float(closed.return_on_cost.mean()),average_holding_days=float(closed.holding_days.mean())))
                for reason,g in closed.groupby('exit_reason'):exits.append(dict(**info,reason=reason,count=len(g),pnl=float(g.pnl.sum())))
                if bp==0:
                    col=0 if k==10 else 1;nav=daily.nav.to_numpy()/1e6
                    axes[0,col].plot(daily.date,nav,label=variant)
                    axes[1,col].plot(daily.date,100*(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1),label=variant)
        col=0 if k==10 else 1
        axes[0,col].plot(benchmark.index,(1+benchmark).cumprod(),label='H00300',color='black',linestyle='--')
        axes[0,col].set_title(f'Top{k}: staged vs full exits');axes[0,col].set_ylabel('Net wealth')
        axes[1,col].set_ylabel('Drawdown (%)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlim(pd.Timestamp(START),pd.Timestamp(END))
    fig.tight_layout();fig.savefig(OUT/'nav_drawdown.png',dpi=150);plt.close(fig)
    tables=dict(comparison=pd.DataFrame(rows),yearly=pd.DataFrame(years),trade_summary=pd.DataFrame(trades),exit_summary=pd.DataFrame(exits))
    for n,f in tables.items():f.to_csv(OUT/(n+'.csv'),index=False,encoding='utf-8-sig')
    base=tables['comparison'].query('friction_bp==0').copy()
    display=base[['top_k','variant','total_return','cagr','max_drawdown','sharpe','information_ratio','fees']].copy()
    display.variant=display.variant.map(LABELS)
    for col in ('total_return','cagr','max_drawdown'):display[col]=display[col].map(lambda x:f'{x:.2%}')
    for col in ('sharpe','information_ratio'):display[col]=display[col].round(3)
    display.fees=display.fees.map(lambda x:f'{x:,.2f}')
    display.columns=['持仓数','版本','累计净收益','年化','最大回撤','夏普(无风险1%)','信息比率','费用(元)']
    annual=tables['yearly'].query('friction_bp==0').pivot(index=['top_k','variant'],columns='year',values='total_return').map(lambda x:f'{x:.2%}')
    stress=tables['comparison'].query('friction_bp>0')[['top_k','variant','friction_bp','cagr','max_drawdown','fees']].copy()
    for col in ('cagr','max_drawdown'):stress[col]=stress[col].map(lambda x:f'{x:.2%}')
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Top10/20直接清仓消融</title>',
      '<style>body{font:16px/1.65 Arial,"Microsoft Yahei",sans-serif;max-width:1450px;margin:32px auto;padding:20px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border:1px solid #ddd;text-align:right}th{background:#eef3f8}img{max-width:100%}</style>',
      '<h1>Top10/20：取消减半，触发即提交全部退出</h1><p>2022-01-04至2025-12-31；初始100万元；同一S6十日预测与16个季度模型。12个新账户、6个旧账户复用。基础买入0.05%、卖出0.15%、每笔最低5元；另加每边10bp/20bp压力。无2026数据，无重新训练，无模拟盘变更。</p>',
      '<h2>基础成本</h2>',display.to_html(index=False),'<h2>逐年净收益</h2>',annual.to_html(),
      '<img src="nav_drawdown.png"><h2>费用压力</h2>',stress.to_html(index=False),
      '<h2>完整持仓周期统计</h2>',tables['trade_summary'].query('friction_bp==0').round(5).to_html(index=False),
      '<h2>退出原因</h2>',tables['exit_summary'].query('friction_bp==0').to_html(index=False),
      '<h2>本轮唯一变化与保留条件</h2><ul><li>10/5直接退出取消提前4%减半、盈利减半、跟踪止损与第二段止盈。8/4版本再将止盈、止损阈值分别改为8%、4%。无论股票排名是否仍在前K，价格退出均为全退出意图。</li>',
      '<li>触及指已完成收盘价检查，次日开盘执行，并非盘中触价立即成交；跳空、停牌、跌停可能造成超额亏损。未成交的全退出意图持续保留。</li>',
      '<li>盈利阈值相对初始含买入费用的单位成本；原有含预计卖出费用的累计亏损预算保持不变，因此止损可能比单纯价格跌5%/4%稍早。预算为初始含费成本×止损比例与入场净值2%的较小者，不随净值扩大。</li>',
      '<li>排名第50保留，第51退出；未持有股票从前2外进入前2仍可替换较弱持仓，每日最多2只。风险退出D日，D+3最早可重买。保留股不每日配平，不能同日卖出再买回。</li>',
      '<li>新开仓预算为账户净值95%/K，保持现金和整手约束、T+1及原不可成交处理。收盘估值结束，不虚构年末清仓。</li>',
      '<li>完整持仓周期胜率不把部分成交拆成多个周期。30%回撤线仍是历史标记，非未来风险保证。年化按252交易日，夏普无风险1%，信息比率使用H00300全收益基准。</li>',
      *['<li>'+html.escape(x)+'</li>' for x in p['limitations']],'</ul>',
      '<p><a href="comparison.csv">全部18组指标</a> · <a href="yearly.csv">逐年指标</a> · <a href="trade_summary.csv">交易统计</a> · <a href="exit_summary.csv">退出统计</a> · <a href="final_verification.json">验收记录</a></p></html>']
    (OUT/'report.html').write_text('\n'.join(page),encoding='utf-8')
    verify(p['sources']);verify(p['protected']);assert p['implementation']==implementation()
    save(OUT/'final_verification.json',dict(status='passed',account_checks=checks,new_accounts=12,reused_accounts=6,
         no_half_exits_in_new_accounts=True,source_and_live_unchanged=True,no_training=True,end=END))
    artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('run.lock','status.json','progress.json','completion.json','report_manifest.json')}
    save(OUT/'report_manifest.json',dict(artifacts=artifacts))
    save(OUT/'completion.json',dict(status='complete',new_accounts=12,baseline_accounts=6,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(base[['top_k','variant','total_return','cagr','max_drawdown','fees']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--stage',choices=['audit','backtest','report','all'],default='all');args=parser.parse_args()
    with lock():
        start=time.time();save(OUT/'status.json',dict(status='running',pid=os.getpid(),stage=args.stage))
        try:
            if args.stage in ('audit','all'):audit()
            if args.stage in ('backtest','all'):backtest()
            if args.stage in ('report','all'):report()
            save(OUT/'status.json',dict(status='complete',elapsed=time.time()-start,stage=args.stage))
        except BaseException as e:
            save(OUT/'status.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
