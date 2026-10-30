"""Independent ledger reconciliation and Top10/20 reports; no live deployment."""
from __future__ import annotations
import html
from paper_futu.top5_buffer_data import read,save,digest,START,END
from paper_futu.top5_buffer_report import metrics,trade_metrics


def independent_audit(daily, orders, holdings, closed, opened, top_k, friction):
    import numpy as np
    cash = 1e6; quantities = {}; previous_nav = 1e6
    order_groups = dict(tuple(orders.groupby('date',sort=False)))
    holding_groups = dict(tuple(holdings.groupby('date',sort=False)))
    for row in daily.itertuples():
        g = order_groups.get(row.date)
        fees = 0.
        if g is not None:
            buys, sells = set(g.loc[g.direction.eq('BUY'),'code']),set(g.loc[g.direction.eq('SELL'),'code'])
            if buys.intersection(sells): raise ValueError('Same-day sell then buy')
            for o in g.itertuples():
                from paper_futu.rank_buffer_10_20_engine import fee
                np.testing.assert_allclose(o.fee,fee(o.notional,o.direction=='BUY',friction),atol=1e-7,rtol=0)
                if str(o.trigger_day) >= str(row.date.date()): raise ValueError('Signal not before execution')
                np.testing.assert_allclose(o.cash_before,cash,atol=1e-6,rtol=0)
                if o.direction=='BUY':
                    if quantities.get(o.code,0)>1e-6: raise ValueError('Retained position topped up')
                    cash -= o.notional+o.fee; quantities[o.code]=quantities.get(o.code,0)+o.quantity
                else:
                    cash += o.notional-o.fee; quantities[o.code]=quantities.get(o.code,0)-o.quantity
                    if quantities[o.code]<-1e-6: raise ValueError('Oversold')
                np.testing.assert_allclose(o.cash_after,cash,atol=1e-6,rtol=0)
                fees += o.fee
        h = holding_groups.get(row.date)
        value = float(h.value.sum()) if h is not None else 0.
        actual = set(h.code) if h is not None else set()
        expected = {c for c,q in quantities.items() if q>1e-6}
        if actual != expected or len(actual)>top_k: raise ValueError('Holdings mismatch')
        if h is not None:
            for p in h.itertuples():
                np.testing.assert_allclose(p.quantity,quantities[p.code],atol=1e-6,rtol=0)
        np.testing.assert_allclose([row.nav,row.cash,row.fees,row.net_return],
                                   [cash+value,cash,fees,(cash+value)/previous_nav-1],atol=1e-6,rtol=0)
        previous_nav=row.nav
    profit = closed.pnl.sum()+(opened.pnl.sum() if len(opened) else 0.)
    np.testing.assert_allclose(previous_nav,1e6+profit,atol=1e-5,rtol=0)
    return dict(status='passed',daily_rows=len(daily),orders=len(orders),account_profit=float(profit))


def report(out):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_futu.research_rank_buffer_10_20 import protocol,verify,SOURCE
    p=protocol();verify(p['sources'])
    benchmark=pd.read_parquet(SOURCE/'data/benchmark.parquet').benchmark_return
    rows,years,trades,exits,contribs,checks=[],[],[],[],[],{}
    closed_all,open_all,breaches=[],[],[]
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for k in (20,10):
        for bp in (0,10,20):
            name=f'top{k}_{bp:02d}bp';d=out/'accounts'/name
            c=read(d/'complete.json');verify(c['artifacts'])
            if c['protocol_sha256']!=digest(out/'protocol.json'):raise ValueError('Account binding mismatch')
            daily,orders,holdings,closed,opened,events=[pd.read_parquet(d/(n+'.parquet')) for n in
                ('daily','orders','holdings','episodes','open_episodes','events')]
            for f in (daily,orders,holdings):f['date']=pd.to_datetime(f.date)
            if not pd.DatetimeIndex(daily.date).equals(benchmark.index):raise ValueError('Different account interval')
            checks[name]=independent_audit(daily,orders,holdings,closed,opened,k,bp/10000)
            rows.append(dict(top_k=k,friction_bp=bp,**metrics(daily,benchmark)))
            for y,g in daily.groupby(daily.date.dt.year):years.append(dict(top_k=k,friction_bp=bp,year=int(y),**metrics(g,benchmark)))
            trades.append(dict(top_k=k,friction_bp=bp,**trade_metrics(closed,opened),
                buy_fills=int(orders.direction.eq('BUY').sum()),sell_fills=int(orders.direction.eq('SELL').sum()),
                avg_cycle_pnl=float(closed.pnl.mean()),mean_cycle_return=float(closed.return_on_cost.mean()),
                average_holding_days=float(closed.holding_days.mean()),
                cooldown_blocks=int(events.event.eq('cooldown_blocks_entry').sum()),
                loss_half_cycles=int(closed.half_mode.eq('loss').sum()+opened.half_mode.eq('loss').sum()),
                profit_half_cycles=int(closed.half_mode.eq('profit').sum()+opened.half_mode.eq('profit').sum())))
            for reason,g in closed.groupby('exit_reason'):
                exits.append(dict(top_k=k,friction_bp=bp,reason=reason,count=len(g),
                    mean_return=float(g.return_on_cost.mean()),budget_exceed=int(g.budget_overshoot.gt(1e-7).sum())))
            closed_all.append(closed.assign(top_k=k,friction_bp=bp));open_all.append(opened.assign(top_k=k,friction_bp=bp))
            breach=holdings.loc[holdings.liquidation_pnl < -holdings.budget-1e-7].copy()
            breach['excess_loss']=-breach.liquidation_pnl-breach.budget
            breaches.append(breach.assign(top_k=k,friction_bp=bp))
            values=holdings.pivot(index='date',columns='code',values='value').reindex(daily.date).fillna(0.)
            flow=orders.assign(flow=np.where(orders.direction.eq('BUY'),-orders.notional-orders.fee,orders.notional-orders.fee))
            flow=flow.pivot_table(index='date',columns='code',values='flow',aggfunc='sum').reindex(index=values.index,columns=values.columns).fillna(0.)
            attribution=values-values.shift(1).fillna(0.)+flow
            np.testing.assert_allclose(attribution.sum(axis=1),daily.nav.diff().fillna(daily.nav.iloc[0]-1e6),atol=1e-6,rtol=0)
            for y,g in [('all',attribution),*list(attribution.groupby(attribution.index.year))]:
                for code,pnl in g.sum().items():
                    if abs(pnl)>1e-7:contribs.append(dict(top_k=k,friction_bp=bp,year=y,code=code,pnl=float(pnl)))
            if bp==0:
                wealth=daily.nav.to_numpy()/1e6
                axes[0].plot(daily.date,wealth,label=f'Top{k}')
                axes[1].plot(daily.date,100*(wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1),label=f'Top{k}')
    axes[0].plot(benchmark.index,(1+benchmark).cumprod(),label='H00300',color='black',linestyle='--')
    axes[0].set_title('Top10/20 | 2022-2025 | rank exit >50 | staged TP 10%, loss budget 5%')
    axes[0].set_ylabel('Net wealth (initial = 1)');axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:ax.legend();ax.grid(alpha=.2);ax.set_xlim(pd.Timestamp(START),pd.Timestamp(END))
    fig.tight_layout();fig.savefig(out/'nav_drawdown.png',dpi=160);plt.close(fig)
    tables=dict(comparison=pd.DataFrame(rows),yearly=pd.DataFrame(years),trade_summary=pd.DataFrame(trades),
                exit_summary=pd.DataFrame(exits),stock_contribution=pd.DataFrame(contribs),
                closed_episodes=pd.concat(closed_all,ignore_index=True),open_episodes=pd.concat(open_all,ignore_index=True),
                budget_breach_marks=pd.concat(breaches,ignore_index=True))
    for n,t in tables.items():t.to_csv(out/(n+'.csv'),index=False,encoding='utf-8-sig')
    base=tables['comparison'].query('friction_bp == 0').set_index('top_k')
    display=base[['total_return','cagr','max_drawdown','sharpe','information_ratio','geometric_excess','fees','average_holdings','cash_ratio','max_stock_weight']].copy()
    for col in ('total_return','cagr','max_drawdown','geometric_excess','cash_ratio','max_stock_weight'):display[col]=display[col].map(lambda x:f'{x:.2%}')
    for col in ('sharpe','information_ratio','average_holdings'):display[col]=display[col].round(3)
    display['fees']=display.fees.map(lambda x:f'{x:,.2f}')
    display=display.rename(columns=dict(total_return='累计收益',cagr='年化',max_drawdown='最大回撤',sharpe='夏普(无风险1%)',
        information_ratio='信息比率',geometric_excess='H00300几何超额',fees='费用(元)',average_holdings='平均持仓数',cash_ratio='平均现金比',max_stock_weight='最大单股权重'))
    yearly=tables['yearly'].query('friction_bp == 0')[['top_k','year','total_return','max_drawdown','geometric_excess','fees']].copy()
    for col in ('total_return','max_drawdown','geometric_excess'):yearly[col]=yearly[col].map(lambda x:f'{x:.2%}')
    risk='两组均超过30%历史回撤线。' if base.risk_flag_30.all() else '至少一组未超过30%历史回撤线，但不代表未来风险受此限制。'
    stress=tables['comparison'].query('friction_bp == 20')
    cost='增加每边20bp后，两组年化均为负。' if (stress.cagr<0).all() else '每边20bp压力结果见下表，不能仅依据基础费用判断可执行性。'
    findings=dict(risk=risk,cost=cost,training_reused=True,model_changes=False,
                  warning='Historical screened data, not an independent test; no automatic deployment.')
    save(out/'findings.json',findings)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Top20/Top10排名缓冲实验</title>',
        '<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#17344a}table{border-collapse:collapse;font-size:14px;width:100%;margin:18px 0}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#e9f1f7}img{max-width:100%}.note{background:#fff3df;padding:16px}a{color:#1766a2}</style>',
        '<h1>Top20与Top10：排名缓冲和分段止盈止损</h1><p>2022-01-04至2025-12-31，初始100万元；同一S6十日预测、季度重训。没有新增模型训练。收盘决策、次日理想开盘成交。</p>',
        '<p class="note">'+risk+cost+' 本轮不自动选冠军，不改变现行模拟盘。2022—2025已反复研究，不是独立测试。</p>',
        '<h2>基础成本结果</h2>',display.to_html(),'<img src="nav_drawdown.png" alt="净值与最大回撤">',
        '<h2>逐年表现</h2>',yearly.to_html(index=False),
        '<h2>6组完整费用压力对照</h2>',tables['comparison'].round(5).to_html(index=False),
        '<h2>完整持仓周期与交易</h2>',tables['trade_summary'].round(5).to_html(index=False),
        '<h2>退出原因</h2>',tables['exit_summary'].round(5).to_html(index=False),
        '<h2>规则及解释</h2><ul>',
        '<li>Top20/Top10新开仓预算分别为净值4.75%/9.5%，股票预算95%。保留仓允许漂移，不每日配平、不补回减半份额。</li>',
        '<li>排名第50保留，第51退出。新跨入前2的强势股仍可替换前50内的持仓，每日最多两只；同分不强制替换。</li>',
        '<li>跌4%且仍在本组前K名减半，其他持仓等待5%退出；累计亏损预算含买卖费用，预算优先、整笔周期不重置。</li>',
        '<li>涨10%且未减半、仍在本组前K名则减半，否则清仓。亏损减半后仍以原成本涨10%为全退出目标。</li>',
        '<li>止盈减半后，以实际减半成交价再涨10%为第二次止盈；跟踪止损取剩余份额含费保本与减半后最高已知价格回撤5%的较高者，只上移。</li>',
        '<li>风险清仓D日，D+3开盘起可再买；普通排名/强势替换不另加冷却。未成交退出持续保留，不占用不存在的空位。</li>',
        '<li>单笔5%是初始含费成本的累计亏损触发预算，非次日开盘成交的保证；跳空、停牌等超额损失未截断。</li>',
        '<li>年化采用252个交易日；逐年回撤在各年期初重置。夏普使用1%年无风险利率，信息比率使用与H00300日收益差。基准首日开盘到收盘，其后收盘到收盘。</li>',
        '<li>年末仅估值，未平仓周期单列；胜率与平均每次收益只统计完整周期，减半不重复计为新周期。</li>',
        *['<li>'+html.escape(x)+'</li>' for x in p['limitations']],'</ul>',
        '<h2>与上一轮的关系</h2><p>上一轮Top5同时采用不同排名退出门槛和止盈止损参数。本轮改变多项规则，因此不能把与Top5的差异单独归因于分散持仓；本轮Top10/20还同时改变与K绑定的减半排名门槛。</p>',
        '<p><a href="comparison.csv">6组指标</a> · <a href="yearly.csv">逐年指标</a> · <a href="trade_summary.csv">交易统计</a> · <a href="exit_summary.csv">退出统计</a> · <a href="closed_episodes.csv">完整周期</a> · <a href="open_episodes.csv">未平仓周期</a> · <a href="stock_contribution.csv">个股贡献</a> · <a href="budget_breach_marks.csv">预算突破</a> · <a href="final_verification.json">验收</a></p></html>']
    (out/'report.html').write_text('\n'.join(body),encoding='utf-8')
    verify(p['protected']);verify(p['frozen_source_manifest']);verify(p['sources'])
    save(out/'final_verification.json',dict(status='passed',accounts=checks,account_count=6,quarters_reused=16,
         source_and_live_files_unchanged=True,no_training=True,end=END))
    files=[f for f in out.iterdir() if f.is_file() and f.suffix in ('.json','.csv','.html','.png')
           and f.name not in ('status.json','progress.json','completion.json','report_manifest.json')]
    save(out/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in files}))
    save(out/'completion.json',dict(status='complete',accounts=6,manifest_sha256=digest(out/'report_manifest.json')))

