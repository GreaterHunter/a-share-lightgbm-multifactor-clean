"""Descriptive held-period and stopped-share counterfactuals; no strategy search."""
import math
import numpy as np
import pandas as pd
from paper_futu.top5_buffer_data import ROOT,END,read,save,digest
from paper_futu import research_top5_head2_full_exit as full
from paper_futu import research_top5_10_20_staged as staged
from paper_futu.research_rank_buffer_futu_fees import SOURCE,fee,verify

OUT=ROOT/'reports/analysis/exit_horizon_diagnostics_asof_20251231'

def future_session(calendar,day,n=10):
    i=calendar.get_indexer([pd.Timestamp(day)])[0]
    return calendar[i+n] if i>=0 and i+n<len(calendar) else None

def valid(x):return x is not None and math.isfinite(x) and x>0

def kept_minus_sold(quantity,sold_price,future_price,sold_fee):
    """Same adjusted shares; no cash interest, reinvestment or new entry fee."""
    value=quantity*future_price
    return value-fee(value,False)-(quantity*sold_price-sold_fee)

def describe_holds(g):
    return dict(closed_cycles=len(g),mean_holding_days=float(g.holding_days.mean()),median_holding_days=float(g.holding_days.median()),
        p25=float(g.holding_days.quantile(.25)),p75=float(g.holding_days.quantile(.75)),p90=float(g.holding_days.quantile(.9)),
        share_lt10=float(g.holding_days.lt(10).mean()),share_eq10=float(g.holding_days.eq(10).mean()),share_gt10=float(g.holding_days.gt(10).mean()),
        mean_win_hold=float(g.loc[g.pnl>0,'holding_days'].mean()),mean_loss_hold=float(g.loc[g.pnl<0,'holding_days'].mean()))

def stop_summary(g):
    good=g[g.future_10_valid];up=good[good.forward_return>1e-10];down=good[good.forward_return < -1e-10]
    early=g[g.exited_before_original_horizon];mature=early[early.original_10_valid]
    return dict(stop_events=len(g),unique_episodes=g.episode_id.nunique(),observed_10=len(good),missing_10=len(g)-len(good),
        rebound_count=len(up),decline_count=len(down),flat_count=len(good)-len(up)-len(down),
        rebound_rate=len(up)/len(good) if len(good) else np.nan,
        average_rebound=float(up.forward_return.mean()),average_further_loss=float(down.forward_return.mean()),
        average_forward_return=float(good.forward_return.mean()),
        weighted_forward_return=float((good.quantity*good.future_open).sum()/good.notional.sum()-1) if len(good) else np.nan,
        net_avoided_losses=float((-good.keep_minus_sell_net.clip(upper=0)).sum()),
        net_missed_gains=float(good.keep_minus_sell_net.clip(lower=0).sum()),
        stop_advantage_net=float(-good.keep_minus_sell_net.sum()),
        net_rebound_rate=float(good.keep_minus_sell_net.gt(1e-8).mean()),
        full_cost_recovered_count=int(good.hypothetical_whole_position_profit.gt(0).sum()),
        full_cost_recovered_rate=float(good.hypothetical_whole_position_profit.gt(0).mean()),
        early_stop_events=len(early),early_original_observed=len(mature),
        early_original_hold_profitable=int(mature.original_whole_position_profit.gt(0).sum()),
        early_original_hold_profitable_rate=float(mature.original_whole_position_profit.gt(0).mean()),
        early_original_rebound_rate=float(mature.original_return_from_exit.gt(0).mean()),
        early_original_mean_delta=float(mature.original_return_from_exit.mean()))

def run():
    sources={};protected={}
    for base in (full.OUT,staged.OUT):
        c=read(base/'completion.json');assert c['status']=='complete' and c['manifest_sha256']==digest(base/'report_manifest.json')
        p=read(base/'protocol.json');sources.update(p['source']);sources.update(read(base/'report_manifest.json')['artifacts']);protected.update(p['protected'])
    sources[__file__]=digest(__file__)
    verify(sources);verify(protected)
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(end=END,accounts=12,no_2026=True,sources=sources,protected=protected,
        main_stop_reasons=['loss_budget','hard_stop','loss_half if minimum lot forces full exit'],separate_reasons=['actual partial loss_half','profit_trailing'],
        forward='sale-day adjusted open to 10 trading sessions later adjusted open; exact endpoint required',
        original='entry-day open to entry+10 open; only stops before this endpoint diagnose premature exit',
        pnl='Sold quantity hypothetical future net liquidation proceeds minus actual net proceeds. No cash reinvestment, cash interest, future execution constraints or full account counterfactual.',
        warnings=['Repeated episodes may overlap forward windows; sums are not attainable account P&L changes.',
        'Rebound is return above sale price, not recovery of original purchase cost.',
        'Endpoint price observation is not evidence an actual future order could execute.',
        'The model predicts cross-sectional ranks, not a guaranteed positive individual 10-day return.',
        'Already researched historical interval; descriptive associations do not establish causation.'])
    save(OUT/'protocol.json',protocol)
    calendar=pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    assert calendar.max()==pd.Timestamp(END)
    market=pd.read_parquet(SOURCE/'data/market.parquet');assert market.index.get_level_values('datetime').max()<=pd.Timestamp(END)
    opens=market.open.unstack('instrument');closes=market.close.unstack('instrument')
    def at(frame,day,code):
        if day is None or day not in frame.index or code not in frame.columns:return np.nan
        value=frame.at[day,code];return float(value) if valid(value) else np.nan
    holdings=[];allclosed=[];opened_rows=[];events=[];reason_holds=[]
    for base,mode in ((full.OUT,'full'),(staged.OUT,'staged')):
        for k in (5,10,20):
            for threshold in ('10_5','8_4'):
                name=f'top{k}_{mode}_{threshold}';d=base/'accounts'/name
                closed=pd.read_parquet(d/'episodes.parquet');opened=pd.read_parquet(d/'open_episodes.parquet')
                orders=pd.read_parquet(d/'orders.parquet');orders.date=pd.to_datetime(orders.date)
                daily=pd.read_parquet(d/'daily.parquet');daily.date=pd.to_datetime(daily.date)
                navs=daily.set_index('date').nav
                info=dict(strategy=name,top_k=k,mode=mode,threshold=threshold)
                for year,g in [('all',closed),*list(closed.groupby(pd.to_datetime(closed.exit_day).dt.year))]:
                    holdings.append(dict(**info,year=str(year),**describe_holds(g)))
                for reason,g in closed.groupby('exit_reason'):reason_holds.append(dict(**info,reason=reason,**describe_holds(g)))
                allclosed.append(closed.assign(**info));opened_rows.append(opened.assign(**info))
                lookup=pd.concat([closed,opened],ignore_index=True).set_index('episode_id')
                assert not lookup.index.has_duplicates
                buys=orders[orders.direction.eq('BUY')].set_index('episode_id')
                selected=orders[orders.direction.eq('SELL') & orders.reason.isin(['loss_budget','hard_stop','loss_half','profit_trailing'])]
                for o in selected.itertuples():
                    ep=lookup.loc[o.episode_id];buy=buys.loc[o.episode_id]
                    future=future_session(calendar,o.date);endprice=at(opens,future,o.code)
                    original=future_session(calendar,buy.date);originalprice=at(opens,original,o.code)
                    early=original is not None and o.date<original
                    if original is None:
                        early=calendar.get_loc(o.date)-calendar.get_loc(buy.date)<10
                    complete=valid(endprice);original_valid=valid(originalprice)
                    hypothetical=lambda p:ep.initial_qty*p-fee(ep.initial_qty*p,False)-ep.initial_cost
                    indices=calendar[(calendar>o.date)&(calendar<=future)] if future is not None else []
                    subsequent=closes.reindex(index=indices)[o.code].dropna() if o.code in closes else pd.Series(dtype=float)
                    lot_forced_full=o.reason=='loss_half' and str(ep.get('exit_day',''))==str(o.date.date())
                    reason_class='loss_clear' if o.reason in ('loss_budget','hard_stop') or lot_forced_full else o.reason
                    events.append(dict(**info,episode_id=int(o.episode_id),code=o.code,reason=o.reason,reason_class=reason_class,
                        entry_date=str(buy.date.date()),sale_date=str(o.date.date()),year=int(o.date.year),
                        holding_days=int(calendar.get_loc(o.date)-calendar.get_loc(buy.date)),quantity=o.quantity,notional=o.notional,sale_price=o.price,
                        initial_cost=ep.initial_cost,initial_quantity=ep.initial_qty,actual_sale_fee=o.fee,
                        future_date=str(future.date()) if future is not None else '',future_10_valid=bool(complete),
                        missing_reason='' if complete else ('past_cutoff' if future is None else 'missing_endpoint_quote'),
                        future_open=endprice,forward_return=endprice/o.price-1 if complete else np.nan,
                        keep_minus_sell_net=kept_minus_sold(o.quantity,o.price,endprice,o.fee) if complete else np.nan,
                        hypothetical_whole_position_profit=hypothetical(endprice) if complete else np.nan,
                        within10_max_close_return=float(subsequent.max()/o.price-1) if complete and len(subsequent) else np.nan,
                        original_horizon_date=str(original.date()) if original is not None else '',exited_before_original_horizon=bool(early),
                        original_10_valid=bool(original_valid),original_return_from_exit=originalprice/o.price-1 if original_valid else np.nan,
                        original_whole_position_return=originalprice/buy.price-1 if original_valid else np.nan,
                        original_whole_position_profit=hypothetical(originalprice) if original_valid else np.nan,
                        sale_account_nav=float(navs.loc[o.date])))
                print(f'ANALYZED {name}',flush=True)
    event=pd.DataFrame(events);hold=pd.DataFrame(holdings)
    summaries=[]
    for (name,reason_class),g in event.groupby(['strategy','reason_class']):
        first=g.iloc[0];info={k:first[k] for k in ('strategy','top_k','mode','threshold','reason_class')}
        for year,sub in [('all',g),*list(g.groupby('year'))]:summaries.append(dict(**info,year=str(year),**stop_summary(sub)))
    summary=pd.DataFrame(summaries)
    tables=dict(holding_periods=hold,holding_periods_by_exit=pd.DataFrame(reason_holds),stop_events=event,
                stop_summary=summary,closed_cycles=pd.concat(allclosed,ignore_index=True),open_cycles=pd.concat(opened_rows,ignore_index=True))
    for n,t in tables.items():t.to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig')
    main=hold.query("year=='all'").merge(summary.query("year=='all' and reason_class=='loss_clear'"),on=['strategy','top_k','mode','threshold','year'])
    main.to_csv(OUT/'main_comparison.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main=main.sort_values(['top_k','threshold','mode'])
    fig,axs=plt.subplots(1,2,figsize=(14,7))
    labels=main.strategy.str.replace('staged','stage',regex=False)
    axs[0].barh(labels,main.mean_holding_days,label='Mean');axs[0].scatter(main.median_holding_days,labels,color='black',label='Median')
    axs[0].axvline(10,color='red',linestyle='--');axs[0].set_xlabel('Trading days held');axs[0].legend()
    axs[1].barh(labels,main.rebound_rate*100);axs[1].axvline(50,color='red',linestyle='--');axs[1].set_xlabel('Above stop sale price at +10 open (%)')
    fig.tight_layout();fig.savefig(OUT/'holding_and_rebound.png',dpi=150);plt.close(fig)
    display=main[['strategy','closed_cycles','mean_holding_days','median_holding_days','share_lt10','stop_events','observed_10','rebound_rate','average_rebound','average_further_loss','full_cost_recovered_rate','early_original_hold_profitable_rate','net_avoided_losses','net_missed_gains']].copy()
    for c in ('share_lt10','rebound_rate','average_rebound','average_further_loss','full_cost_recovered_rate','early_original_hold_profitable_rate'):display[c]=display[c].map(lambda x:f'{x:.2%}')
    display=display.round(2)
    display=display.rename(columns=dict(strategy='策略',closed_cycles='完整周期数',mean_holding_days='平均持有日',median_holding_days='中位持有日',share_lt10='少于10日占比',stop_events='亏损清仓数',observed_10='后10日可评价数',rebound_rate='卖后10日上涨比例',average_rebound='上涨组平均涨幅',average_further_loss='下跌组平均跌幅',full_cost_recovered_rate='后10日恢复原整笔成本比例',early_original_hold_profitable_rate='提前止损后原第10日可盈利比例',net_avoided_losses='避免后续净损失元',net_missed_gains='错过后续净收益元'))
    annual=summary.query("year in ['2024','2025'] and reason_class=='loss_clear'").copy()
    page='<html lang="zh"><meta charset="utf-8"><title>持有期与止损后走势诊断</title><style>body{font:15px/1.7 system-ui;max-width:1700px;margin:30px auto;padding:15px}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}img{max-width:100%}.scroll{overflow-x:auto}</style><h1>12组策略：持有期、止损反弹与原10日目标</h1><p>仅使用2022—2025年数据；不重训、不回测新策略、不修改模拟盘。平均与中位持有期统计已完整退出周期，部分减半不另算新周期，未结束周期单列。</p>'
    page+='<p>主要“止损”包括loss_budget与hard_stop亏损全清，以及loss_half因整手不能拆分而实际全清的情形。实际亏损减半与盈利后的profit_trailing单列于明细。收益为止损成交当日开盘到第10个后续交易日开盘，按复权单位；年末越界或缺少准确端点不补齐、不算零。</p><p>卖后反弹只是超过卖出价，并不一定回本。“原第10日”从首次买入日计算，只在其之前止损的样本上诊断提前退出；该反事实假设原始全部股数一直持有，并扣相同卖出费用。模型预测的是相对排名而非保证盈利。</p>'
    page+='<div class="scroll">'+display.to_html(index=False)+'</div><img src="holding_and_rebound.png"><h2>2024、2025止损诊断（比例为小数）</h2><div class="scroll">'+annual[['strategy','year','stop_events','observed_10','rebound_rate','average_rebound','average_further_loss','net_avoided_losses','net_missed_gains','early_original_observed','early_original_hold_profitable_rate']].round(5).to_html(index=False)+'</div>'
    page+='<p>避免损失／错过收益：以各笔已卖出的实际股数，比较当日净卖出所得与10日后假设净卖出所得。未计释放资金的再投资与利息，未模拟未来可成交性；不同持仓周期可能共享未来日期。金额之和不是可实现的账户收益改善，不能直接加到净值中。</p><p>原成本恢复指标假设未曾减半并继续持有最初整笔股份，只是统一对照，不是分段账本的实际盈亏。日线中间最高收盘价仅作补充，不代表盘中最高价。所有结果是已反复研究历史的描述性分析。</p><p><a href="main_comparison.csv">主表</a> · <a href="stop_summary.csv">分类逐年统计</a> · <a href="stop_events.csv">逐笔反事实</a> · <a href="holding_periods.csv">逐年持有期</a> · <a href="open_cycles.csv">年末未结束周期</a></p></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    verify(sources);verify(protected)
    assert (pd.to_datetime(event.sale_date)<=pd.Timestamp(END)).all()
    assert (pd.to_datetime(event.loc[event.future_date.ne(''),'future_date'])<=pd.Timestamp(END)).all()
    save(OUT/'verification.json',dict(status='passed',accounts=12,source_and_live_unchanged=True,no_2026=True,
        full_stops=int(event.reason_class.eq('loss_clear').sum()),missing_endpoints=event.groupby(['reason_class','missing_reason']).size().to_dict().__str__()))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.iterdir() if f.is_file() and f.name not in ('completion.json','report_manifest.json')}))
    save(OUT/'completion.json',dict(status='complete',manifest_sha256=digest(OUT/'report_manifest.json')))
    print(main[['strategy','mean_holding_days','median_holding_days','share_lt10','stop_events','observed_10','rebound_rate','average_rebound','average_further_loss','early_original_observed','early_original_hold_profitable_rate','net_avoided_losses','net_missed_gains']].to_string(index=False),flush=True)

if __name__=='__main__':run()
