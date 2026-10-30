"""Present the isolated stop comparison without mutating any sealed source."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_futu.research_s6_stop_loss import OUT, digest, read, save
from scripts.analyze_s6_realized_sales import reconstruct_sales, stock_date_sales


def main():
    comparison = read(OUT / "comparison.json")
    assert comparison["status"] == "complete"
    assert comparison["protocol_sha256"] == digest(OUT / "protocol.json")
    cases = comparison["cases"]
    details, phase_rows = [], []
    for offset in range(5):
        paired = {}
        for mode in ("baseline", "stop20"):
            directory = OUT / "accounts" / f"{mode}_{offset}"
            entry = read(directory / "complete.json")
            for filename, expected in entry["artifacts"].items():
                assert digest(filename) == expected
            paired[mode] = entry["periods"]["all"]
            if mode == "baseline":
                assert entry["baseline_parity_passed"]
                continue
            orders = pd.read_parquet(directory / "orders.parquet")
            fills, _ = reconstruct_sales(orders)
            sales = stock_date_sales(fills, pd.read_parquet(directory / "holdings.parquet"))
            stop_keys = orders.loc[orders.stop_loss & orders.notional.gt(0),
                                   ["datetime", "instrument"]].drop_duplicates()
            stop_sales = sales.merge(stop_keys, on=["datetime", "instrument"], validate="one_to_one")
            for row in stop_sales.itertuples():
                details.append(dict(offset=offset, date=str(row.datetime.date()),
                    instrument=row.instrument, net_return=row.net_return,
                    net_pnl=row.net_pnl, full_exit=bool(row.full_exit)))
        phase_rows.append(dict(offset=offset, baseline=paired["baseline"], stop20=paired["stop20"],
            cagr_delta=paired["stop20"]["cagr"]-paired["baseline"]["cagr"],
            drawdown_delta=paired["stop20"]["max_drawdown"]-paired["baseline"]["max_drawdown"]))
    stats = dict(cagr_improved_offsets=sum(r["cagr_delta"] > 0 for r in phase_rows),
        drawdown_reduced_offsets=sum(r["drawdown_delta"] < 0 for r in phase_rows),
        stop_sales=len(details), stop_sales_below_minus20=sum(r["net_return"] < -.2 for r in details),
        mean_stop_return=float(np.mean([r["net_return"] for r in details])),
        worst_stop_sale=min(details, key=lambda r: r["net_return"]),
        stop_sales_per_year={str(y): sum(r["date"].startswith(str(y)) for r in details)/5
                             for y in range(2022, 2027)},
        phase_cagr_std={mode:float(np.std([r[mode]["cagr"] for r in phase_rows], ddof=0))
                        for mode in ("baseline", "stop20")})
    save(OUT / "execution_diagnostics.json", dict(stats=stats, phases=phase_rows, stop_sales=details))

    def pct(value): return f"{value:.2%}"
    def table(headers, rows):
        return "<table><thead><tr>"+"".join(f"<th>{html.escape(str(v))}</th>" for v in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>"
    a, b = (cases[m]["periods"]["all"] for m in ("baseline", "stop20"))
    overall = table(["指标（5起点等权汇总）", "原版S6", "20%止损"], [
        ["平均复合年化", pct(a["cagr"]), pct(b["cagr"])],
        ["平均累计收益", pct(a["total_return"]), pct(b["total_return"])],
        ["平均最大回撤", pct(a["max_drawdown"]), pct(b["max_drawdown"])],
        ["最差起点最大回撤", pct(a["worst_phase_drawdown"]), pct(b["worst_phase_drawdown"])],
        ["平均全期费用（元/初始100万元）", f"{a['fees']:,.2f}", f"{b['fees']:,.2f}"],
        ["平均每日双边换手", pct(a["mean_daily_turnover"]), pct(b["mean_daily_turnover"])],
        ["平均持股数量", f"{a['mean_holdings']:.3f}", f"{b['mean_holdings']:.3f}"]])
    years = []
    for year in map(str, range(2022, 2027)):
        x, y = (cases[m]["periods"][year] for m in ("baseline", "stop20"))
        years.append([year if year != "2026" else "2026截至10月13日", pct(x["total_return"]),
            pct(y["total_return"]), f"{(y['total_return']-x['total_return'])*100:+.2f}个百分点",
            pct(x["max_drawdown"]), pct(y["max_drawdown"]), stats["stop_sales_per_year"][year]])
    annual = table(["年份", "原版收益", "止损收益", "变化", "原版年内回撤均值", "止损年内回撤均值", "平均止损卖出次数"], years)
    phases = table(["调仓偏移", "原版年化", "止损年化", "原版回撤", "止损回撤"], [
        [r["offset"], pct(r["baseline"]["cagr"]), pct(r["stop20"]["cagr"]),
         pct(r["baseline"]["max_drawdown"]), pct(r["stop20"]["max_drawdown"])] for r in phase_rows])
    text = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>S6 20%止损对照</title>
<style>body{{max-width:1150px;margin:36px auto;font:16px/1.7 system-ui,sans-serif;color:#163044;padding:0 24px}}h1,h2{{color:#153d60}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #dce4ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf4f8}}p{{max-width:1000px}}small{{color:#52687a}}</style>
<h1>S6：20%固定成本止损对照</h1>
<p>2022-01-04至2026-10-13；S6 QREG，季度重训，Top20、前6只合计40%股票仓位、总股票目标95%，每5日调仓，最多替换6只。预测与模型完全复用，没有重训。</p>
<p>止损是<strong>单股相对含买入费用的移动加权平均成本亏损20%</strong>，不是组合回撤，也不是从最高价回落20%。每日收盘判断，下个可交易日开盘卖出；停牌或跌停阻断则继续等待。卖后持现金至正常调仓，触发日不回买；止损优先于普通替换数量上限。正常调仓仍可能日后重新选入。</p>
<h2>全期对照</h2>{overall}<p>各起点先算复合年化，再取算术平均；没有把累计收益除以年份。年化因子252。最大回撤包括初始净值，按日终账户净值历史高点至后续低点计算。</p>
<h2>逐年对照</h2>{annual}<p>逐年列为区间累计收益，不作年化；2026为不完整年度。年内回撤各年从起始净值重新计算，不等于全期跨年最大回撤。</p>
<h2>全部起点</h2>{phases}<p>年化改善{stats['cagr_improved_offsets']}/5个起点；回撤下降{stats['drawdown_reduced_offsets']}/5个起点。</p>
<h2>止损成交核对</h2><p>五条独立账户合计{stats['stop_sales']}次止损成交（同一事件可能出现在多个起点，不是独立样本）；其中{stats['stop_sales_below_minus20']}次卖出净收益低于−20%。最差为{html.escape(str(stats['worst_stop_sale']))}。平均止损卖出净收益{pct(stats['mean_stop_return'])}。所以20%是触发线，不是保证成交损失上限。</p>
<h2>验证与局限</h2><p>五个原版账户均逐日还原封存净值、现金、费用及订单；十个账户已核对净值、交易费用、现金与持仓，来源数据哈希未改变。原模型、历史封存和模拟盘未修改。</p>
<p>仍是日线理想开盘成交，使用旧回测整手规则及统一±9.5%涨跌幅过滤，不代表已验证真实成交或已应用10月20日执行器取整修复。不包含盘中触发或20%精确成交假设。2022—2026已反复研究，本对照不能当作新的独立测试，更不能仅凭一组阈值判断未来止损一定有效。</p>
<small>协议SHA256：{comparison['protocol_sha256']}。详细账户与成交记录位于本目录accounts；execution_diagnostics.json记录逐笔止损和各起点指标。</small></html>"""
    (OUT / "report.html").write_text(text, encoding="utf-8")
    save(OUT / "report_manifest.json", {p.name:digest(p) for p in (
        OUT / "comparison.json", OUT / "execution_diagnostics.json", OUT / "report.html")})
    print(json.dumps(stats, ensure_ascii=False))
    print(str(OUT / "report.html"))


if __name__ == "__main__": main()
