"""Recompute metrics from saved ledgers; static plots and a self-contained report."""
from __future__ import annotations

import html
import math

from paper_futu.top5_buffer_data import read, save, digest, START, END

LABELS = {'A':'仅排名缓冲', 'B':'硬止损10%', 'C':'8%提前退出',
          'D':'8%前五减半／其他10%退出', 'E':'D取消第二次固定止盈'}


def metrics(daily, benchmark):
    import numpy as np
    r = daily.net_return.to_numpy()
    b = benchmark.reindex(daily.date).to_numpy()
    if not np.isfinite(r).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite daily returns')
    nav = np.cumprod(1+r); base = np.prod(1+b)
    dd = 1-nav/np.maximum.accumulate(np.r_[1.,nav])[1:]
    active = r-b; rf = 1.01**(1/252)-1
    sharpe = (r-rf).mean()/r.std(ddof=1)*math.sqrt(252) if len(r)>1 and r.std(ddof=1)>0 else None
    ir = active.mean()/active.std(ddof=1)*math.sqrt(252) if len(active)>1 and active.std(ddof=1)>0 else None
    return dict(days=len(r),total_return=float(nav[-1]-1),cagr=float(nav[-1]**(252/len(r))-1),
                max_drawdown=float(dd.max()),benchmark_return=float(base-1),
                geometric_excess=float(nav[-1]/base-1),
                annual_geometric_excess=float((nav[-1]/base)**(252/len(r))-1),sharpe=sharpe,information_ratio=ir,
                fees=float(daily.fees.sum()),two_sided_turnover=float(daily.turnover.sum()),
                average_holdings=float(daily.holdings.mean()),cash_ratio=float(daily.cash_ratio.mean()),
                max_stock_weight=float(daily.max_weight.max()),risk_flag_30=bool(dd.max()>.3))


def trade_metrics(closed, opened):
    def mean(s): return float(s.mean()) if len(s) else None
    if not len(closed): return dict(closed=0,open=len(opened))
    wins, losses = closed.loc[closed.pnl>1e-7],closed.loc[closed.pnl < -1e-7]
    return dict(closed=len(closed),open=len(opened),win_rate=len(wins)/len(closed),
                flat=int(closed.pnl.abs().le(1e-7).sum()),avg_gain=mean(wins.return_on_cost),
                avg_loss=mean(losses.return_on_cost),avg_winning_days=mean(wins.holding_days),
                avg_losing_days=mean(losses.holding_days),
                max_gain=float(closed.return_on_cost.max()),max_loss=float(closed.return_on_cost.min()),
                half_episodes=int(closed.half_mode.ne('').sum()+opened.half_mode.ne('').sum()) if len(opened) else int(closed.half_mode.ne('').sum()),
                closed_budget_exceed_count=int(closed.budget_overshoot.gt(1e-7).sum()),
                closed_budget_exceed_total=float(closed.budget_overshoot.sum()),
                max_budget_exceed=float(closed.budget_overshoot.max()))


def independent_audit(daily, orders, holdings, closed, opened):
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
        if actual != expected or len(actual)>5: raise ValueError('Holdings mismatch')
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
    from paper_futu.research_top5_rank_buffer import protocol,verify_files
    p=protocol(); verify_files(read(out/'audit_complete.json')['artifacts'])
    benchmark=pd.read_parquet(out/'data/benchmark.parquet').benchmark_return
    rows,annual,trade_rows,contributions,exit_rows,checks=[],[],[],[],[],{}
    annual_contributions, all_closed, all_open, all_breaches = [], [], [], []
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    curves={}
    for case in 'ABCDE':
        for bp in (0,10,20):
            name=f'{case}_{bp:02d}bp'; directory=out/'accounts'/name
            c=read(directory/'complete.json');verify_files(c['artifacts'])
            if c['protocol_sha256'] != digest(out/'protocol.json'):raise ValueError('Mixed account protocol')
            frames={n:pd.read_parquet(directory/(n+'.parquet')) for n in
                    ('daily','orders','holdings','episodes','open_episodes','events')}
            daily,orders,holdings,closed,opened,events=[frames[n] for n in
                    ('daily','orders','holdings','episodes','open_episodes','events')]
            for f in (daily,orders,holdings):f['date']=pd.to_datetime(f.date)
            if daily.date.iloc[0] != pd.Timestamp(START) or daily.date.iloc[-1]!=pd.Timestamp(END):
                raise ValueError('Comparison interval changed')
            checks[name]=independent_audit(daily,orders,holdings,closed,opened)
            rows.append(dict(case=case,friction_bp=bp,**metrics(daily,benchmark)))
            for y,g in daily.groupby(daily.date.dt.year):
                annual.append(dict(case=case,friction_bp=bp,year=int(y),**metrics(g,benchmark)))
            trade_rows.append(dict(case=case,friction_bp=bp,**trade_metrics(closed,opened),
                                   cooldown_blocks=int(events.event.eq('cooldown_blocks_entry').sum())))
            all_closed.append(closed.assign(case=case,friction_bp=bp))
            all_open.append(opened.assign(case=case,friction_bp=bp))
            breach=holdings.loc[holdings.liquidation_pnl < -holdings.budget-1e-7].copy()
            breach['excess_loss']=-breach.liquidation_pnl-breach.budget
            all_breaches.append(breach.assign(case=case,friction_bp=bp,risk_rule_active=case!='A'))
            # Episode P&L, with open cycles marked to final close, sums to NAV profit.
            all_profit=pd.concat([closed[['code','pnl']],opened[['code','pnl']]],ignore_index=True) if len(opened) else closed[['code','pnl']]
            for code,g in all_profit.groupby('code'):
                contributions.append(dict(case=case,friction_bp=bp,code=code,pnl=float(g.pnl.sum()),
                                          contribution_on_initial_cash=float(g.pnl.sum()/1e6)))
            # Daily mark-to-market contribution: end value - prior end value +
            # sale proceeds - buys - fees. Includes all partial and open episodes.
            values=holdings.pivot(index='date',columns='code',values='value').reindex(daily.date).fillna(0.)
            cashflow=orders.assign(flow=np.where(orders.direction.eq('BUY'),-orders.notional-orders.fee,
                                                 orders.notional-orders.fee))
            flow=cashflow.pivot_table(index='date',columns='code',values='flow',aggfunc='sum').reindex(
                index=values.index,columns=values.columns).fillna(0.)
            attribution=values-values.shift(1).fillna(0.)+flow
            np.testing.assert_allclose(attribution.sum(axis=1),daily.nav.diff().fillna(daily.nav.iloc[0]-1e6),atol=1e-6,rtol=0)
            for year,g in attribution.groupby(attribution.index.year):
                for code,pnl in g.sum().items():
                    if abs(pnl)>1e-7:
                        annual_contributions.append(dict(case=case,friction_bp=bp,year=int(year),code=code,pnl=float(pnl)))
            for reason,g in closed.groupby('exit_reason'):
                exit_rows.append(dict(case=case,friction_bp=bp,reason=reason,count=len(g),
                                      mean_return=float(g.return_on_cost.mean()),budget_exceed=int(g.budget_overshoot.gt(1e-7).sum())))
            if bp==0:
                wealth=daily.nav.to_numpy()/1e6
                dd=wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1
                curves[case]=daily
                axes[0].plot(daily.date,wealth,label=case)
                axes[1].plot(daily.date,dd*100,label=case)
    axes[0].plot(benchmark.index,(1+benchmark).cumprod(),color='black',linestyle='--',label='H00300')
    axes[0].set_ylabel('Net wealth (initial = 1)');axes[1].set_ylabel('Drawdown (%)')
    axes[0].set_title('Top5 rank buffer | 2022-2025 | base costs | ideal next-open execution')
    for ax in axes:ax.legend(ncol=6);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'nav_drawdown.png',dpi=160);plt.close(fig)
    tables={'comparison':pd.DataFrame(rows),'yearly':pd.DataFrame(annual),
            'episode_summary':pd.DataFrame(trade_rows),'stock_contribution':pd.DataFrame(contributions),
            'yearly_stock_contribution':pd.DataFrame(annual_contributions),
            'closed_episodes':pd.concat(all_closed,ignore_index=True),
            'open_episodes':pd.concat(all_open,ignore_index=True),
            'budget_breach_marks':pd.concat(all_breaches,ignore_index=True),
            'exit_summary':pd.DataFrame(exit_rows)}
    for n,t in tables.items():t.to_csv(out/(n+'.csv'),index=False,encoding='utf-8-sig')
    base=tables['comparison'].query('friction_bp == 0').copy()
    display=base[['case','total_return','cagr','max_drawdown','geometric_excess','sharpe','information_ratio',
                  'fees','two_sided_turnover','average_holdings','cash_ratio','max_stock_weight']].copy()
    for col in ('total_return','cagr','max_drawdown','geometric_excess','cash_ratio','max_stock_weight'):
        display[col]=display[col].map(lambda x:f'{x:.2%}')
    for col in ('sharpe','information_ratio','two_sided_turnover','average_holdings'):display[col]=display[col].round(3)
    display['fees']=display.fees.map(lambda x:f'{x:,.2f}')
    display=display.rename(columns=dict(case='组别',total_return='累计收益',cagr='年化',max_drawdown='最大回撤',
        geometric_excess='H00300几何超额',sharpe='夏普(无风险1%)',information_ratio='信息比率',fees='费用(元)',
        two_sided_turnover='累计双边换手',average_holdings='平均持仓数',cash_ratio='平均现金比',max_stock_weight='最大单股权重'))
    yearly=tables['yearly'].query('friction_bp == 0')[['case','year','total_return','max_drawdown','geometric_excess','fees']].copy()
    for col in ('total_return','max_drawdown','geometric_excess'):yearly[col]=yearly[col].map(lambda x:f'{x:.2%}')
    text=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Top5排名缓冲实验（2022—2025）</title>',
          '<style>body{font-family:system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 24px;color:#183047}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#e9f1f7}h1,h2{color:#123b58}.note{background:#fff3db;padding:16px}img{max-width:100%}a{color:#1766a2}</style>',
          '<h1>Top5排名缓冲与分段止盈止损</h1><p>区间2022-01-04—2025-12-31；初始100万元；S6十日预测、季度重训；每天收盘决策，次日理想开盘成交。</p>',
          '<p class="note">本轮没有使用2026训练、验证、预测或绩效行。但此前已观察过2026，不能重新称为独立测试。排名缓冲A是必要对照，本报告不按最高历史收益自动选新冠军。</p>',
          '<p>'+html.escape('；'.join(f'{k}：{v}' for k,v in LABELS.items()))+'</p>',
          '<h2>基础成本对照</h2>',display.to_html(index=False,escape=True),
          '<img src="nav_drawdown.png" alt="净值与回撤">',
          '<h2>逐年结果（重点观察2024、2025）</h2>',yearly.to_html(index=False,escape=True),
          '<h2>交易摩擦压力：全部15组</h2>',tables['comparison'].round(5).to_html(index=False),
          '<h2>完整持仓周期与未平仓周期</h2>',tables['episode_summary'].round(5).to_html(index=False),
          '<h2>最终退出原因</h2>',tables['exit_summary'].round(5).to_html(index=False),
          '<h2>口径与限制</h2><ul>',
          '<li>年化=(期末净值/期初净值)^(252/交易日数)−1；最大回撤含初始净值。逐年回撤在各年期初重新起算。</li>',
          '<li>夏普使用日收益减去1%年利率换算的日无风险收益；信息比率使用策略与H00300日收益差的均值/样本标准差×√252。</li>',
          '<li>第一天基准从开盘起算，随后收盘到收盘。年末只按收盘估值，不虚构平仓，不用2026完成标签。</li>',
          '<li>亏损预算按整笔持仓周期锁定，减半不重置。收盘触发、次日开盘成交可能因跳空或无法成交而突破预算；未截断成交价。</li>',
          '<li>止盈减半后的保本线针对剩余份额原始含费成本；已实现盈利不用于降低该保本线。亏损减半则继续用整笔累计亏损预算。</li>',
          '<li>强势入场只对新跨入前2的未持仓股票生效。同分按代码排序，但同分不强制换股。持仓自然漂移可使单股权重高于19%，不为恢复等权而交易。</li>',
          '<li>A无价格风险控制，其超过假设预算的交易仅作比较，不代表执行了预算止损。</li>',
          *['<li>'+html.escape(x)+'</li>' for x in p['limitations']],'</ul>',
          '<p>完整数据：<a href="comparison.csv">15组指标</a> · <a href="yearly.csv">逐年指标</a> · <a href="episode_summary.csv">周期统计</a> · <a href="stock_contribution.csv">个股贡献</a> · <a href="exit_summary.csv">退出审计</a> · <a href="models_audit.json">16个季度模型边界</a> · <a href="final_verification.json">验收</a></p></html>']
    (out/'report.html').write_text('\n'.join(text),encoding='utf-8')
    verify_files(p['protected'])
    save(out/'final_verification.json',dict(status='passed',accounts=checks,account_count=len(checks),
         end=END,models=16,protected_unchanged=True,financial_revision_archive_complete=False))
    artifacts={str(f):digest(f) for f in out.iterdir() if f.is_file() and f.suffix in ('.csv','.html','.png','.json')
               and f.name not in ('status.json','progress.json','completion.json','report_manifest.json')}
    save(out/'report_manifest.json',dict(artifacts=artifacts))
    save(out/'completion.json',dict(status='complete',accounts=15,report_manifest_sha256=digest(out/'report_manifest.json')))
