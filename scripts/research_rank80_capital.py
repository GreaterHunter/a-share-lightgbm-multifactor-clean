"""Isolated capital-only replay of the sealed 2022-2025 rank80/strong1 study."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from paper_futu import research_top10_rank80_strong1 as study
from paper_futu import research_rank_buffer_futu_fees as fees
from paper_futu.top5_buffer_data import ROOT, START, END, read, save, digest
from paper_futu.top5_buffer_report import metrics, trade_metrics

OUT = ROOT/'reports/experiments/rank80_capital_10k_asof_20251231'


def audit(daily, orders, held, capital):
    cash = previous = float(capital)
    quantities = {}
    og = dict(tuple(orders.groupby('date')))
    hg = dict(tuple(held.groupby('date')))
    for row in daily.itertuples():
        g = og.get(row.date)
        charges = 0.
        if g is not None:
            assert not set(g.loc[g.direction.eq('BUY'), 'code']) & set(g.loc[g.direction.eq('SELL'), 'code'])
            for o in g.itertuples():
                buying = o.direction == 'BUY'
                expected = max(3., o.notional*.0003)+15.+max(.01, o.notional*.0000487)+3*max(.01, o.notional*.00002)+(0 if buying else o.notional*.001)
                np.testing.assert_allclose(o.fee, expected, atol=1e-8)
                np.testing.assert_allclose(o.cash_before, cash, atol=1e-6, rtol=0)
                assert o.trigger_day < str(row.date.date())
                if buying:
                    assert quantities.get(o.code, 0) < 1e-6
                    if o.code.startswith('SH688'):
                        assert o.raw_quantity >= 200-1e-6
                    else:
                        np.testing.assert_allclose(o.raw_quantity/100, round(o.raw_quantity/100), atol=1e-7)
                sign = 1 if buying else -1
                cash -= sign*o.notional + o.fee
                quantities[o.code] = quantities.get(o.code, 0)+sign*o.quantity
                assert cash >= -1e-6 and quantities[o.code] >= -1e-6
                np.testing.assert_allclose(o.cash_after, cash, atol=1e-6, rtol=0)
                charges += o.fee
        h = hg.get(row.date)
        value = 0 if h is None else h.value.sum()
        assert {c for c,q in quantities.items() if q>1e-6} == (set() if h is None else set(h.code))
        if h is not None:
            assert len(h) <= 10
            for pos in h.itertuples():
                np.testing.assert_allclose(pos.quantity, quantities[pos.code], atol=1e-6, rtol=0)
        np.testing.assert_allclose([row.nav,row.cash,row.fees,row.net_return], [cash+value,cash,charges,(cash+value)/previous-1], atol=1e-6, rtol=0)
        previous = row.nav
    return dict(status='passed', rows=len(daily), fills=len(orders), initial_cash=capital)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    old = read(study.OUT/'protocol.json')
    source_names = [fees.SOURCE/'data'/n for n in ('market.parquet','predictions.parquet','benchmark.parquet','provider/calendars/day.txt')]
    source_names += [ROOT/'paper_futu'/n for n in ('research_top10_rank80_strong1.py','research_top10_rank65_cooldown.py','rank_buffer_full_exit_engine.py','research_rank_buffer_futu_fees.py')]
    sources = {}
    for path in source_names:
        h = digest(path)
        assert h == old['sources'][str(path)], str(path)
        sources[str(path)] = h
    baseline = study.OUT/'accounts/top10_full_10_5'
    completed = read(baseline/'complete.json')
    assert completed['protocol_sha256'] == digest(study.OUT/'protocol.json')
    fees.verify(completed['artifacts'])
    sources.update(completed['artifacts'])
    from paper_futu.cn_runtime_protocol import verify_active_protocol
    live = verify_active_protocol()
    protected = dict(live['immutable_bindings'])
    protected[str(ROOT/'configs/futu_cn_forward_protocol.json')] = digest(ROOT/'configs/futu_cn_forward_protocol.json')
    protocol = dict(start=START,end=END,initial_cash=[1000000,10000],source_hashes=sources,
        implementation_sha256=digest(__file__),protected=protected,training=False,
        rules='H10 quarterly QREG; Top10 R80/S1 full TP10/SL5; TP cooldown2 SL3; retained weights drift',
        fee_policy=old['fee_policy'],comparison='Only initial cash changes; both accounts freshly replayed',
        limitation='Historical adjusted-share/open-fill engine; unaffordable candidates skipped in rank order. Not an exact V10 execution/dividend-ledger simulation; published fee approximation, not broker-verified actual fees.')
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json') == protocol
    else:
        save(OUT/'protocol.json', protocol)
    panel = pd.read_parquet(fees.SOURCE/'data/market.parquet').reset_index()
    assert panel.datetime.max() <= pd.Timestamp(END)
    market = {d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')}
    del panel
    frame = pd.read_parquet(fees.SOURCE/'data/predictions.parquet').reset_index()
    assert frame.execution_date.max() <= pd.Timestamp(END)
    predictions = dict(tuple(frame.groupby('execution_date')))
    cal = pd.DatetimeIndex(pd.to_datetime((fees.SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))
    assert cal.max()==pd.Timestamp(END)
    days = cal[(cal>=START)&(cal<=END)]
    benchmark = pd.read_parquet(fees.SOURCE/'data/benchmark.parquet').benchmark_return
    mod = study.engine('full')
    original_buy = mod.buy_quantity
    rows, annual, curves, checks = [], [], {}, {}
    for capital in (1000000, 10000):
        account = study.make_account('full','10_5')
        account.initial_cash = account.cash = float(capital)
        attempts = []
        context = {}
        def observed_buy(code,money,price,factor,friction=0):
            quantity = original_buy(code,money,price,factor,friction)
            attempts.append(dict(date=context['day'],code=code,rank=context['ranks'].get(code),budget=money,
                raw_price=price/factor,quantity=quantity,blocked=quantity<=0))
            return quantity
        mod.buy_quantity = observed_buy
        try:
            for step,day in enumerate(days):
                prev = cal[cal.get_loc(day)-1]
                r = predictions[day]
                assert r.datetime.eq(prev).all()
                scores = dict(zip(r.instrument,r.prediction))
                context.update(day=str(day.date()),ranks=mod.ranking(scores)[1])
                account.step(str(day.date()),step,scores,market[day],{c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
                if step % 240 == 0:
                    print(f'{capital}: {day.date()} NAV={account.daily[-1]["nav"]:.2f}',flush=True)
        finally:
            mod.buy_quantity = original_buy
        check = account.reconcile()
        opened = account.open_episodes(str(days[-1].date()),len(days)-1)
        directory = OUT/'accounts'/str(capital)
        directory.mkdir(parents=True, exist_ok=True)
        frames = {n:pd.DataFrame(v) for n,v in dict(daily=account.daily,orders=account.orders,holdings=account.holdings,
            episodes=account.episodes,open_episodes=opened,events=account.events,buy_attempts=attempts).items()}
        for name,f in frames.items():
            f.to_parquet(directory/f'{name}.parquet',index=False)
        daily,orders,held,closed,op = (frames[n] for n in ('daily','orders','holdings','episodes','open_episodes'))
        for f in (daily,orders,held):
            f['date'] = pd.to_datetime(f.date)
        assert pd.DatetimeIndex(daily.date).equals(benchmark.index)
        checks[str(capital)] = dict(engine=check,independent=audit(daily,orders,held,capital),
            cooldown_checked=study.base.audit_cooldown(orders,closed,cal))
        np.testing.assert_allclose(daily.nav.iloc[-1],capital+closed.pnl.sum()+(op.pnl.sum() if len(op) else 0.),atol=1e-5,rtol=0)
        if capital==1000000:
            original = pd.read_parquet(baseline/'daily.parquet')
            assert pd.to_datetime(original.date).equals(daily.date)
            np.testing.assert_allclose(daily.select_dtypes('number'),original.select_dtypes('number'),atol=1e-6,rtol=0)
            checks[str(capital)]['sealed_baseline_reproduced'] = True
        attempted = frames['buy_attempts']
        filled = attempted[~attempted.blocked]
        row = dict(initial_cash=capital,**metrics(daily,benchmark),**trade_metrics(closed,op),
            ending_nav=float(daily.nav.iloc[-1]),net_profit=float(daily.nav.iloc[-1]-capital),
            fills=len(orders),fees_over_initial=float(daily.fees.sum()/capital),
            buy_attempts=len(attempted),blocked_buy_attempts=int(attempted.blocked.sum()),
            median_buy_rank=float(filled['rank'].median()),mean_buy_rank=float(filled['rank'].mean()),
            buys_beyond_top10_share=float(filled['rank'].gt(10).mean()),
            days_below_10=int(daily.holdings.lt(10).sum()),mean_buy_price=float(filled.raw_price.mean()))
        rows.append(row);curves[capital]=daily
        for year,g in daily.groupby(daily.date.dt.year):
            annual.append(dict(initial_cash=capital,year=int(year),**metrics(g,benchmark)))
        print(json.dumps(row),flush=True)
        del account
    summary=pd.DataFrame(rows);yearly=pd.DataFrame(annual)
    summary.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    yearly.to_csv(OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for capital,daily in curves.items():
        axes[0].plot(daily.date,daily.nav/capital,label=f'CNY {capital:,}')
        nav=daily.nav.to_numpy()/capital
        axes[1].plot(daily.date,100*(nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1),label=f'CNY {capital:,}')
    axes[0].set_ylabel('Net asset value (initial=1)');axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'nav_drawdown.png',dpi=160);plt.close(fig)
    display=summary[['initial_cash','ending_nav','total_return','cagr','max_drawdown','fees','average_holdings','cash_ratio','fills','win_rate','median_buy_rank','buys_beyond_top10_share']].copy()
    for c in ('total_return','cagr','max_drawdown','cash_ratio','win_rate','buys_beyond_top10_share'):display[c]=display[c].map(lambda v:f'{v:.2%}')
    table=yearly[['initial_cash','year','total_return','max_drawdown','fees']].copy()
    for c in ('total_return','max_drawdown'):table[c]=table[c].map(lambda v:f'{v:.2%}')
    text='<html lang="zh"><meta charset="utf-8"><title>1万元资金规模对照</title><style>body{font:16px/1.7 system-ui;max-width:1450px;margin:30px auto}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>80/1 Top10 直接10/5：1万元 vs 100万元</h1><p>2022-01-04至2025-12-31；同模型预测、同费用、仅改变初始现金。100万元重跑已逐日复现封存结果。未训练模型，未修改模拟盘。</p>'+display.round(2).to_html(index=False)+'<h2>逐年表现</h2>'+table.round(2).to_html(index=False)+'<img src="nav_drawdown.png"><p>费用沿用原富途费表研究假设：佣金0.03%最低3元、每单15元平台费及其他税费，非精确实扣复现。指标已扣费，不重复扣除。买入失败会向后搜索可负担股票，因此Top10是最多持仓数量，不保证持仓均为模型排名前10。</p><p>诊断中的失败次数是候选买入评估次数，不是提交给券商的拒单；同一股票可在多个日期重复统计。最大回撤按每日收盘净值计算。未结束周期只估值。</p><p>这是历史策略资金敏感性回放，不是V10逐笔执行仿真：沿用复权库存近似、统一涨跌幅过滤、理想开盘成交，不包含V10分红待核对停机或限价重挂。2022—2025已多次研究，非独立测试。未修改封存报告、模型或当前部署。</p><a href="comparison.csv">完整指标</a> · <a href="yearly.csv">逐年CSV</a></html>'
    (OUT/'report.html').write_text(text,encoding='utf-8')
    fees.verify(sources);fees.verify(protected)
    save(OUT/'verification.json',dict(status='passed',accounts=checks,source_and_live_hashes_unchanged=True))
    save(OUT/'completion.json',dict(status='complete',accounts=2,no_training=True,artifacts={str(f.relative_to(OUT)):digest(f) for f in OUT.rglob('*') if f.is_file() and f.suffix in ('.parquet','.csv','.html','.png')},protocol_sha256=digest(OUT/'protocol.json')))


if __name__=='__main__':
    run()
