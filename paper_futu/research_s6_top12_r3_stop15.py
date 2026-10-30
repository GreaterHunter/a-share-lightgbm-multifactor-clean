"""Three-day rebalance comparison using the sealed Top12 stop account engine."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from paper_futu import research_s6_top12_stop15 as engine

ROOT, START, END = engine.ROOT, engine.START, engine.END
OUT = ROOT / "reports/experiments/s6_top12_head6_60_r3_stop15_asof_20261013"
SOURCE_R5 = engine.output_dir(.6)
read, save, digest = engine.read, engine.save, engine.digest


def runtime_floor(original_floor):
    """User-authorized lower memory floors, confined to this experiment."""
    return {1.5: .8, 1.0: .5}.get(original_floor, original_floor)


def schedule_dates(calendar, start, end, offset, rebalance=3):
    """Fixed comparison interval; global trading-day cadence, never reset yearly."""
    if rebalance < 1 or not 0 <= offset < rebalance:
        raise ValueError("Offset must be a unique phase of a positive period")
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar must be unique and increasing")
    days = calendar[(calendar >= start) & (calendar <= end)]
    trades = days[offset::rebalance]
    indexes = calendar.get_indexer(trades)
    if len(trades) == 0 or (indexes <= 0).any():
        raise ValueError("No trades or missing previous trading-day signal")
    return days, trades, calendar[indexes-1]


def audit():
    previous = engine.verify_binding(SOURCE_R5)
    assert previous["rebalance"] == 5 and previous["head_weight"] == .6
    expected = dict(previous["sources"])
    for name in ("protocol.json", "comparison.json", "report_manifest.json", "report.html"):
        path = SOURCE_R5 / name
        expected[str(path)] = digest(path)
    for offset in range(5):
        path = SOURCE_R5 / f"accounts/top12_stop15_{offset}/complete.json"
        record = read(path)
        assert record["account_audit_passed"]
        assert record["protocol_sha256"] == digest(SOURCE_R5 / "protocol.json")
        expected[str(path)] = digest(path)
        expected.update(record["artifacts"])
    for path, expected_hash in expected.items():
        if digest(path) != expected_hash: raise ValueError(f"Source changed: {path}")
    protocol = {k:v for k,v in previous.items() if k not in ("implementation", "sources", "reference")}
    protocol.update(rebalance=3, offsets=[0,1,2], cases={"top12_stop15":.15},
        runtime=dict(max_workers=1,threads=4,startup_free_gib=.8,task_free_gib=.5,
            authorization="2026-10-25 user explicitly requested direct start after being informed of low memory"),
        reference="Top12/head6_60/15% stop, rebalance 5 days, all five phases",
        comparison="R3 all 3 phases versus R5 all 5 phases; same interval; no fictitious paired-phase significance",
        schedule_adapter="isolated process-local schedule function replacement; unchanged account engine and explicit output directory",
        implementation={**previous["implementation"],str(Path(__file__)):digest(__file__)}, sources=expected)
    path = OUT / "protocol.json"
    if path.exists() and read(path) != protocol:
        if any((OUT/"accounts").glob("*/complete.json")):
            raise ValueError("Protocol mismatch; preserve existing completed experiment")
        save(OUT/"pre_execution_protocols"/f"{digest(path)}.json",read(path))
    save(path, protocol)
    print("Input audit passed",flush=True)


def account(offset):
    # This patch is confined to this dedicated single-account subprocess.
    # It changes no module output path, source file, broker, or other process.
    from paper_futu import s6_explicit_account as adapter
    from src import runtime_budget
    original_guard=runtime_budget.require_memory
    def scoped_guard(required,purpose):
        return original_guard(runtime_floor(required),purpose)
    def bound_schedule(calendar, start, end, phase):
        days, trades, signals = schedule_dates(calendar,start,end,phase)
        save(OUT/f"schedules/offset_{offset}.json",dict(
            interval=[str(days[0].date()),str(days[-1].date())], rebalance=3,
            trades=[str(d.date()) for d in trades], signals=[str(d.date()) for d in signals]))
        return days, trades, signals
    with patch.object(adapter,"schedule_dates",bound_schedule), patch.object(runtime_budget,"require_memory",scoped_guard):
        engine.account(OUT,.6,"top12_stop15",offset)


def report():
    import numpy as np
    import pandas as pd
    from scripts.analyze_s6_realized_sales import reconstruct_sales,stock_date_sales
    protocol=engine.verify_binding(OUT)
    for path,value in protocol["sources"].items():
        if digest(path)!=value: raise ValueError(f"Source changed: {path}")
    cases,accounts={},[]
    for name,root,phases in (("rebalance5",SOURCE_R5,range(5)),("rebalance3",OUT,range(3))):
        entries=[]
        for offset in phases:
            directory=root/f"accounts/top12_stop15_{offset}"
            entry=read(directory/"complete.json")
            assert entry["account_audit_passed"] and entry["protocol_sha256"]==digest(root/"protocol.json")
            for path,value in entry["artifacts"].items(): assert digest(path)==value
            daily=pd.read_parquet(directory/"daily.parquet")
            assert str(daily.index[0].date())==START and str(daily.index[-1].date())==END
            if accounts:
                assert len(daily)==accounts[0]["days"]
            fills,_=reconstruct_sales(pd.read_parquet(directory/"orders.parquet"))
            sales=stock_date_sales(fills,pd.read_parquet(directory/"holdings.parquet"))
            winners=sales.loc[sales.net_class.eq("profit")]
            losers=sales.loc[sales.net_class.eq("loss")]
            entry.update(case=name,days=len(daily),trade_metrics=dict(win_rate=len(winners)/len(sales),
                mean_win=float(winners.net_return.mean()),mean_loss=float(losers.net_return.mean()),
                worst_sale=float(sales.net_return.min())))
            entries.append(entry)
        periods={}
        for period in entries[0]["periods"]:
            rows=[e["periods"][period] for e in entries]
            periods[period]={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            periods[period]["worst_phase_drawdown"]=max(r["max_drawdown"] for r in rows)
        cases[name]=dict(phases=len(entries),periods=periods,
            mean_stop_fills=float(np.mean([e["stop_sell_fills"] for e in entries])),
            phase_cagr_std=float(np.std([e["periods"]["all"]["cagr"] for e in entries])),
            worst_sale=min(e["trade_metrics"]["worst_sale"] for e in entries),
            trade_metrics={k:float(np.mean([e["trade_metrics"][k] for e in entries])) for k in entries[0]["trade_metrics"]})
        accounts.extend(entries)
    save(OUT/"comparison.json",dict(status="complete",source_hashes_unchanged=True,
        protocol_sha256=digest(OUT/"protocol.json"),cases=cases,accounts=accounts))
    def table(headers,rows):
        return "<table><tr>"+"".join(f"<th>{html.escape(str(v))}</th>" for v in headers)+"</tr>"+"".join(
            "<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in row)+"</tr>" for row in rows)+"</table>"
    columns=["5日调仓＋15%止损","3日调仓＋15%止损"]
    rows=[]
    for key,label,fmt in (("cagr","平均复合年化",".2%"),("total_return","平均累计收益",".2%"),
        ("max_drawdown","平均最大回撤",".2%"),("worst_phase_drawdown","最差起点回撤",".2%"),
        ("mean_daily_turnover","平均每日双边换手",".2%"),("fees","全期平均费用（元）",",.2f"),
        ("mean_holdings","平均持仓数",".2f")):
        rows.append([label,*[format(c["periods"]["all"][key],fmt) for c in cases.values()]])
    years=[[str(y) if y!=2026 else "2026截至10月13日",*[f"{c['periods'][str(y)]['total_return']:.2%}" for c in cases.values()]] for y in range(2022,2027)]
    phases=[[e["case"],e["offset"],f"{e['periods']['all']['cagr']:.2%}",f"{e['periods']['all']['max_drawdown']:.2%}"] for e in accounts]
    text=f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>S6 Top12：3日与5日调仓</title>
<style>body{{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#183449}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border-bottom:1px solid #dce5ed;text-align:right;padding:10px}}th{{background:#edf4f8}}th:first-child,td:first-child{{text-align:left}}</style>
<h1>S6 Top12：提高至3个交易日调仓</h1><p>2022-01-04至2026-10-13。固定Top12、前6只合计60%股票仓位、后6只40%、总股票目标95%、每次普通替换最多6只、15%单股成本止损。S6季度重训预测完全复用。只改变调仓周期，不改模拟盘。</p>
<p>3日周期覆盖全部3个相位（0/1/2）；5日对照复用全部5个相位（0/1/2/3/4）。区间一致，跨年不重置节奏。均值分别按各周期全部相位等权计算，不能把两组当成相同数量的配对独立样本，也不挑最佳起点。</p>
<h2>全期含成本比较</h2>{table(['指标',*columns],rows)}<h2>逐年区间收益（非年化）</h2>{table(['年份',*columns],years)}<h2>全部起点</h2>{table(['版本','起点偏移','复合年化','最大回撤'],phases)}
<p>年化因子252，先按每条账户计算复合年化后取算术平均；回撤从日终净值历史高点计算，包含初始净值。初始资金100万元，买入0.05%、卖出0.15%、最低5元，只交易目标仓位差。</p>
<h2>止损与成交限制</h2><p>每日收盘检查相对含买入费用的移动加权成本是否亏损15%，下一可交易日开盘卖出；止损不必等待3日调仓。停止卖出后，非调仓日持有现金，后续正常调仓可重新选入，不当日回买。跳空、停牌和跌停可能令实际亏损超过15%。</p>
<p>保留原日线理想开盘成交、旧整手规则和统一±9.5%方向过滤，不能视为真实成交保证。历史2022—2026反复参与研究，不是新的独立测试。3个新账户与5个原账户已核验现金、费用、净值、持仓上限、区间及源数据哈希；原封存目录、部署和模拟盘不变。</p>
<p>协议SHA256：{digest(OUT/'protocol.json')}。明细见comparison.json、schedules与accounts。</p></html>"""
    (OUT/"report.html").write_text(text,encoding="utf-8")
    save(OUT/"report_manifest.json",{p.name:digest(p) for p in (OUT/"comparison.json",OUT/"report.html")})
    print(json.dumps(cases,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["audit","account","report","all"],default="all")
    parser.add_argument("--offset",type=int,choices=range(3),default=0)
    args=parser.parse_args()
    if args.stage=="audit": audit()
    elif args.stage=="account": account(args.offset)
    elif args.stage=="report": report()
    else:
        audit()
        (OUT/"logs").mkdir(exist_ok=True)
        for offset in range(3):
            save(OUT/"status.json",dict(state="running",offset=offset,pid=os.getpid(),updated=time.time()))
            with (OUT/f"logs/offset_{offset}.log").open("a",encoding="utf-8") as log:
                result=subprocess.run([sys.executable,"-u","-m","paper_futu.research_s6_top12_r3_stop15",
                    "--stage","account","--offset",str(offset)],cwd=ROOT,
                    env={**os.environ,**engine.ENV},stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                save(OUT/"status.json",dict(state="failed",offset=offset,exit_code=result.returncode))
                raise SystemExit(result.returncode)
            print(f"Completed offset {offset}",flush=True)
        report()
        save(OUT/"status.json",dict(state="complete",updated=time.time()))


if __name__=="__main__": main()
