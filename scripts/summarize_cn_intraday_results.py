"""Human-readable companion report for the completed, audited minute study."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_futu.research_cn_intraday import RUN, SOURCE, read, verify, save, digest


def main():
    verify()
    audit = read(RUN / 'independent_audit.json')
    if audit['status'] != 'passed':
        raise RuntimeError('Independent audit must pass first')
    summary = pd.read_csv(RUN / 'comparison.csv')
    yearly = pd.read_csv(RUN / 'yearly.csv')
    ic = pd.read_csv(RUN / 'sampled_rank_ic_yearly.csv')
    benchmark = pd.read_parquet(SOURCE / 'benchmark.parquet').benchmark_return
    names = {'daily15': '日频15因子对照', 'minute29': '日频15＋分钟14因子'}
    trades, reasons, contributors, importance = [], [], [], []
    for model in names:
        for bp in (0, 10):
            folder = RUN / 'accounts' / f'{model}_{bp}bp'
            d = pd.read_parquet(folder / 'daily.parquet')
            d['date'] = pd.to_datetime(d.date)
            c = pd.read_parquet(folder / 'closed.parquet')
            h = pd.read_parquet(folder / 'holdings.parquet')
            o = pd.read_parquet(folder / 'orders.parquet')
            cal = pd.DatetimeIndex(d.date)
            c['holding_sessions'] = cal.get_indexer(pd.to_datetime(c.exit_day)) - cal.get_indexer(pd.to_datetime(c.entry_day))
            wins = c.loc[c.pnl > 0]
            losses = c.loc[c.pnl < 0]
            trades.append(dict(model=model, friction_bp=bp, closed=len(c),
                mean_holding_sessions=float(c.holding_sessions.mean()),
                median_holding_sessions=float(c.holding_sessions.median()),
                win_rate=len(wins)/len(c), mean_win=float(wins.return_on_cost.mean()),
                mean_loss=float(losses.return_on_cost.mean()),
                mean_pnl=float(c.pnl.mean()), max_win=float(c.return_on_cost.max()),
                max_loss=float(c.return_on_cost.min()), open_positions=int(d.holdings.iloc[-1])))
            for reason, g in c.groupby('reason'):
                reasons.append(dict(model=model, friction_bp=bp, reason=reason,
                    closed=len(g), pnl=float(g.pnl.sum()), mean_return=float(g.return_on_cost.mean())))
            # Lifetime cash-flow attribution includes open-position mark-to-market.
            signed = np.where(o.side.eq('SELL'), o.value, -o.value) - o.fee
            pnl = pd.Series(signed.to_numpy(), index=o.code).groupby(level=0).sum()
            last = h.loc[pd.to_datetime(h.date).eq(d.date.iloc[-1])].set_index('code').value
            pnl = pnl.add(last, fill_value=0)
            np.testing.assert_allclose(pnl.sum(), d.nav.iloc[-1]-1e6, atol=1e-4, rtol=0)
            for code, value in pnl.sort_values(ascending=False).items():
                contributors.append(dict(model=model, friction_bp=bp, code=code, net_pnl=float(value)))
        import lightgbm as lgb
        for path in sorted((RUN / 'models' / model).glob('*/model.txt')):
            booster = lgb.Booster(model_file=str(path))
            gain = booster.feature_importance(importance_type='gain').astype(float)
            gain = gain / gain.sum() if gain.sum() else gain
            for feature, value in zip(booster.feature_name(), gain):
                importance.append(dict(model=model, quarter=path.parent.name, feature=feature, gain_share=float(value)))
    for name, rows in [('trade_statistics', trades), ('exit_reasons', reasons),
                       ('stock_contributions', contributors), ('feature_importance', importance)]:
        pd.DataFrame(rows).to_csv(RUN / f'{name}.csv', index=False, encoding='utf-8-sig')

    def table(frame, columns, percentages=()):
        f = frame[list(columns)].copy()
        if 'variant' in f: f['variant'] = f.variant.map(names)
        if 'model' in f: f['model'] = f.model.map(names)
        for key in percentages: f[key] = f[key].map(lambda v: f'{v:.2%}')
        for key in ('fees', 'mean_pnl'): 
            if key in f: f[key] = f[key].map(lambda v: f'{v:,.2f}')
        for key in ('sharpe', 'information_ratio', 'mean_holding_sessions', 'median_holding_sessions', 'mean'):
            if key in f: f[key] = f[key].map(lambda v: f'{v:.4f}')
        return f.rename(columns=columns).to_html(index=False, escape=True)

    main_table = table(summary, dict(variant='模型', friction_bp='每边额外摩擦(bp)', total_return='累计净收益',
        cagr='净年化', max_drawdown='最大回撤', annual_geometric_excess='年化几何超额', sharpe='夏普(无风险1%)',
        information_ratio='信息比率', fees='全期费用/元', fills='成交笔数'),
        ('total_return','cagr','max_drawdown','annual_geometric_excess'))
    year_table = table(yearly, dict(variant='模型', friction_bp='额外摩擦(bp)', year='年份', total_return='年度净收益',
        max_drawdown='年度最大回撤', benchmark_return='H00300收益', geometric_excess='几何超额'),
        ('total_return','max_drawdown','benchmark_return','geometric_excess'))
    trade_table = table(pd.DataFrame(trades), dict(model='模型', friction_bp='额外摩擦(bp)', closed='完整持仓周期',
        mean_holding_sessions='平均持有交易日', median_holding_sessions='持有交易日中位数', win_rate='胜率',
        mean_win='平均盈利幅度', mean_loss='平均亏损幅度', mean_pnl='平均每周期盈亏/元', open_positions='期末未平仓'),
        ('win_rate','mean_win','mean_loss'))
    ic_table = table(ic, dict(variant='模型', year='年份', mean='平均Rank IC', count='成熟横截面数'))
    base = summary.loc[summary.friction_bp.eq(0)].set_index('variant')
    delta = base.loc['minute29','cagr'] - base.loc['daily15','cagr']
    dd_delta = base.loc['minute29','max_drawdown'] - base.loc['daily15','max_drawdown']
    benchmark_total = float((1+benchmark).prod()-1)
    claims = (f'加入分钟特征后，基础费用下净年化相差{delta*100:+.2f}个百分点，最大回撤相差{dd_delta*100:+.2f}个百分点。'
              f'H00300同期累计收益{benchmark_total:.2%}。这只是固定规则的历史对照，不能据此宣布获得独立验证的日内Alpha。')
    t = pd.DataFrame(trades).loc[lambda x: x.friction_bp.eq(0)].set_index('model')
    diagnostics = (f'基础费用下，完整交易周期从{int(t.loc["daily15","closed"])}次增加到{int(t.loc["minute29","closed"])}次，'
        f'平均持有期从{t.loc["daily15","mean_holding_sessions"]:.2f}个交易日降到{t.loc["minute29","mean_holding_sessions"]:.2f}日。'
        f'胜率由{t.loc["daily15","win_rate"]:.2%}提高到{t.loc["minute29","win_rate"]:.2%}，'
        f'但盈利周期的平均收益由{t.loc["daily15","mean_win"]:.2%}降到{t.loc["minute29","mean_win"]:.2%}，'
        f'全期费用从{base.loc["daily15","fees"]/10000:.2f}万元增加到{base.loc["minute29","fees"]/10000:.2f}万元。'
        '交易更多、胜率更高没有自动转化为更高净收益。两组全期最大回撤都超过原30%历史筛选线；不能据此替换V10。')
    notes = '''<ul>
    <li>区间2022-01-04至2025-12-31；初始资金100万元。训练从2015年开始，季度从头重训，合计32个模型。2026年数据未进入本次训练、回测或标签。</li>
    <li>两组共用成熟的未来10交易日、相同时点开盘到开盘收益排名标签；日频特征只使用上一交易日值。沿用原S6各年事前树数和参数，没有为分钟模型重新搜索最优树数；此处的15因子对照也不是原部署S6的净值复算。</li>
    <li>训练每天固定抽样09:45、10:45、13:45、14:45四个时点；交易每5分钟评估。分钟特征由1分钟数据聚合成5分钟线，尚未检验不同采样方式。</li>
    <li>信号来自已结束的5分钟线，隔一根完整5分钟线后按后续线开盘成交。保留T+1、Top10、排名跌出80退出、新进第1名替换、止盈10%、止损5%；止盈无额外冷却，止损最早下一交易日再买。单次新买目标9.5%，保留仓位允许漂移，不补仓配平。</li>
    <li>基础费用沿用本项目富途模拟费率假设：佣金0.03%最低3元、平台费15元、经手费0.00487%、监管费0.002%、两项过户费各0.002%，经手、监管、过户费各最低0.01元；卖出另计0.1%。未计托管费，券商逐项舍入尚未核验。这是固定研究口径，不声称还原所有年份或现实券商最新费率。</li>
    <li>另做每边10bp摩擦账户，费用随成交金额计入现金并重新回放；费用已包含在净收益里，不能再扣一次。止损按含费盈亏触发，整手与现金约束也受费用影响，因此压力账户可能改变后续交易路径；差异不是在相同成交序列上简单多扣手续费。</li>
    <li>价格触发不保证成交在阈值：跳空、T+1、停牌及成交量限制可使亏损超过5%。零成交报价不能交易，每根线最多参与该线成交量5%；这仍不能还原排队及真实冲击成本。</li>
    <li>盘中相对收益参照510300ETF；总体绩效基准为H00300沪深300全收益指数。日终使用冻结日线的官方收盘价估值，年末未平仓单列。</li>
    <li>公开分钟数据上游来源披露不足；零成交占位价、一个矛盾量额记录已隔离。历史财务修订档案不完整，复权持仓模拟公司行为，未建立逐笔原始股数及分红现金账本。统一9.5%方向过滤也未完整还原各板块和ST涨跌停。</li>
    <li>2022—2025已反复参与此前研究，本轮不是新独立测试。不自动切换V10，不连接券商，不改变模拟盘和调度。</li>
    </ul>'''
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分钟因子回测结论</title><style>body{font:16px/1.65 system-ui;max-width:1450px;margin:32px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #dce2e8;text-align:right}th{background:#eef3f8}img{width:100%}li{margin:8px 0}.table{overflow-x:auto}</style><h1>A股分钟因子：已完成离线对照</h1><p>'+html.escape(claims)+'</p><h2>全期表现（已扣费用）</h2><div class="table">'+main_table+'</div><img src="curves.png" alt="净值和回撤"><h2>逐年表现</h2><div class="table">'+year_table+'</div><h2>完整持仓周期</h2>'+trade_table+'<p>成交笔数包含买入、卖出和部分成交；完整周期从买入至全部卖出，两者不能混作交易次数。持有期按交易日差计。</p><h2>成熟标签上的抽样Rank IC</h2>'+ic_table+'<p>同日四个横截面及相邻10日标签相互相关，不能把它们当成相互独立样本推断显著性。</p><h2>口径与局限</h2>'+notes+'<h2>核验</h2><p>独立核对现金、数量、费用、日净值、实际源分钟报价、成交量上限、T+1、冷却期，以及32个季度模型的成熟标签边界和保存重载一致性。详细记录：<a href="independent_audit.json">独立审计</a>、<a href="protocol.json">固定协议</a>、<a href="comparison.csv">指标</a>、<a href="yearly.csv">逐年</a>、<a href="stock_contributions.csv">股票贡献</a>、<a href="trade_statistics.csv">持有期与胜率</a>。</p></html>'
    page = page.replace('<h2>口径与局限</h2>', '<h2>结果如何理解</h2><p>'+html.escape(diagnostics)+'</p><h2>口径与局限</h2>')
    (RUN / 'summary.html').write_text(page, encoding='utf-8')
    files = [RUN/'summary.html', RUN/'independent_audit.json', RUN/'trade_statistics.csv', RUN/'exit_reasons.csv',
             RUN/'stock_contributions.csv', RUN/'feature_importance.csv', RUN/'sampled_rank_ic_yearly.csv']
    save(RUN/'delivery_manifest.json', dict(status='complete_and_independently_audited',
         pipeline_completion_sha256=digest(RUN/'completion.json'), artifacts={str(p):digest(p) for p in files}))
    print(claims, flush=True)


if __name__ == '__main__': main()
