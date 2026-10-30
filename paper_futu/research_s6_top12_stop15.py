"""Isolated Top12 / 15% stop comparison; never changes live configuration."""
from __future__ import annotations

import argparse
from functools import partial
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from paper_futu import research_s6_stop_loss as original

ROOT, DATA, SOURCE = original.ROOT, original.DATA, original.SOURCE
START, END, ENV = original.START, original.END, original.ENV
read, save, digest = original.read, original.save, original.digest
CASES = {"top12": None, "top12_stop15": .15}


def output_dir(head_weight):
    if head_weight not in (.4, .5, .6):
        raise ValueError("Choose an explicitly specified head allocation")
    return ROOT / f"reports/experiments/s6_top12_head6_{round(head_weight*100)}_stop15_asof_20261013"


def strategy_parameters(head_weight, mode):
    return dict(risk_degree=.95, only_tradable=True, top_k=12, top_tier_size=6,
                top_tier_weight=head_weight, n_drop=6, threshold=CASES[mode])


def verify_binding(out):
    p = read(out / "protocol.json")
    for path, value in p["implementation"].items():
        if digest(path) != value:
            raise ValueError(f"Implementation changed: {path}")
    return p


def audit(out, head_weight):
    previous = read(original.OUT / "protocol.json")
    expected = dict(previous["sources"])
    expected[str(original.OUT / "protocol.json")] = digest(original.OUT / "protocol.json")
    expected[str(original.OUT / "comparison.json")] = digest(original.OUT / "comparison.json")
    for offset in range(5):
        path = original.OUT / f"accounts/baseline_{offset}/complete.json"
        record = read(path)
        assert record["baseline_parity_passed"] and record["account_audit_passed"]
        assert record["protocol_sha256"] == digest(original.OUT / "protocol.json")
        expected[str(path)] = digest(path)
        expected.update(record["artifacts"])
    config = ROOT / "configs/experiments/s6_head6_40_research_main.json"
    expected[str(config)] = digest(config)
    main = read(config)
    expected[main["model_path"]] = main["model_sha256"]
    for path, value in expected.items():
        if digest(path) != value:
            raise ValueError(f"Frozen input changed: {path}")
    if previous["implementation_sha256"] != digest(original.__file__):
        raise ValueError("Previously audited stop engine changed")
    protocol = dict(model="S6_QREG", start=START, end=END, top_k=12, head_size=6,
        head_weight=head_weight, risk_degree=.95, ordinary_n_drop=6, rebalance=5,
        offsets=list(range(5)), cases=CASES, initial_cash=1e6,
        reference="existing audited Top20/head6_40 QREG without stop",
        stop_rule="previous completed close / moving average purchase cost including buy fees <= -15%; next tradable open exit",
        stop_overrides_replacement_limit=True, no_same_day_stop_reentry=True,
        reentry="eligible again at a subsequent scheduled rebalance; no additional cooldown",
        cash="stop-only day keeps all other amounts unchanged; ordinary rebalances use original sizing",
        costs=dict(buy=.0005, sell=.0015, minimum=5),
        limitations=["ideal daily open fills", "original sealed lot sizing", "uniform +/-9.5% direction limit", "historically reused data, not independent test"],
        runtime=dict(max_workers=1, threads=4, startup_free_gib=1.5, task_free_gib=1.),
        implementation={str(Path(__file__)):digest(__file__), str(Path(original.__file__)):digest(original.__file__)},
        sources=expected)
    path = out / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("Protocol mismatch: do not overwrite existing experiment")
    save(path, protocol)
    print("Input audit passed", flush=True)


def account(out, head_weight, mode, offset):
    from src.runtime_budget import require_memory
    require_memory(1.5, "Top12 single-process account startup")
    for key, value in ENV.items(): os.environ[key] = value
    import numpy as np
    import pandas as pd
    from paper_futu.s6_explicit_account import init_qlib, schedule_dates
    from src.compare_models import enrich_daily_report
    from qlib.data import D
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.decision import Order
    from qlib.contrib.evaluate import backtest_daily

    protocol = verify_binding(out)
    assert protocol["head_weight"] == head_weight
    directory = out / f"accounts/{mode}_{offset}"
    complete = directory / "complete.json"
    if complete.exists():
        entry = read(complete)
        assert entry["protocol_sha256"] == digest(out / "protocol.json")
        for path, value in entry["artifacts"].items(): assert digest(path) == value
        return
    init_qlib(SimpleNamespace(DATA=DATA, OUT=out, NAME=out.name))
    Market, Strategy = original.classes()
    stream = read(SOURCE / "streams.json")["streams"]["history_QREG"]
    assert digest(stream["path"]) == stream["sha256"]
    prediction = pd.read_parquet(stream["path"], columns=["prediction"])
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    days, scheduled, signal_dates = schedule_dates(calendar, START, END, offset)
    prior_day = calendar[calendar < days[0]][-1]
    signal = prediction.loc[prediction.index.get_level_values("datetime").isin(signal_dates)]
    assert signal.index.get_level_values("datetime").nunique() == len(signal_dates)
    market = Market(freq="day", start_time=prior_day, end_time=END,
        codes=sorted(prediction.index.get_level_values("instrument").unique()), deal_price="open",
        open_cost=.0005, close_cost=.0015, min_cost=5., trade_unit=100,
        limit_threshold=("($open / Ref($close, 1) - 1) >= 0.095", "($open / Ref($close, 1) - 1) <= -0.095"))
    require_memory(1., "Top12 loaded account safety floor")
    market.reset_audit(0., 12)
    strategy = Strategy(signal=signal, **strategy_parameters(head_weight, mode))
    module = importlib.import_module("qlib.backtest.backtest")
    with patch.object(module, "tqdm", partial(module.tqdm, disable=True)):
        raw, positions = backtest_daily(start_time=prior_day, end_time=END, strategy=strategy,
            executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True,
                verbose=False, trade_type=SimulatorExecutor.TT_SERIAL), account=1e6,
            benchmark="SH000300", exchange_kwargs={"exchange":market})
    report = enrich_daily_report(raw.reindex(days), f"{mode}_{offset}")
    orders = pd.DataFrame(market.journal)
    trades = orders.loc[orders.notional.gt(0)]
    assert trades.loc[~trades.datetime.isin(scheduled), "stop_loss"].all()
    assert trades.loc[trades.stop_loss, "direction"].eq(Order.SELL).all()
    if CASES[mode] is None: assert not trades.stop_loss.any()
    totals = trades.groupby("datetime")[["notional", "fee"]].sum().reindex(days, fill_value=0.)
    prior = report.account.shift(1, fill_value=1e6)
    report["fees_cash"], report["traded_notional"] = totals.fee, totals.notional
    report["turnover"] = totals.notional/prior
    holding, contribution, previous = [], [], {}
    for day in days:
        position = positions[day]
        market.check_holding_limit(position, day, "top12_close")
        current = {}
        for stock in position.get_stock_list():
            qty, price = position.get_stock_amount(stock), position.get_stock_price(stock)
            current[stock] = qty*price
            holding.append(dict(datetime=day, instrument=stock, adjusted_quantity=qty,
                adjusted_close=price, market_value=qty*price, cash=position.get_cash()))
        today = trades.loc[trades.datetime.eq(day)]
        flow = today.assign(flow=np.where(today.direction.eq(Order.BUY), today.notional, -today.notional)).groupby("instrument").flow.sum()
        for stock in set(previous)|set(current)|set(flow.index):
            contribution.append(dict(datetime=day, instrument=stock,
                pnl_before_fees=current.get(stock,0)-previous.get(stock,0)-flow.get(stock,0)))
        previous = current
    holding, pnl = pd.DataFrame(holding), pd.DataFrame(contribution)
    report["holdings_count"] = holding.groupby("datetime").size().reindex(days, fill_value=0)
    values = holding.groupby("datetime").market_value.sum().reindex(days, fill_value=0)
    gross = pnl.groupby("datetime").pnl_before_fees.sum().reindex(days, fill_value=0)
    for actual, expected in ((values+report.cash, report.account),
        (prior+gross-totals.fee, report.account), ((1+report.net_return).cumprod()*1e6, report.account),
        (report.cost*prior, totals.fee)):
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-10)
    assert report.cash.min() >= -1e-5 and report.holdings_count.max() <= 12
    if offset: np.testing.assert_array_equal(report.account.iloc[:offset], np.full(offset, 1e6))
    baseline = pd.read_parquet(original.OUT / f"accounts/baseline_{offset}/daily.parquet")
    pd.testing.assert_index_equal(report.index, baseline.index)
    report["bench"], report["benchmark_nav"] = baseline.bench, baseline.benchmark_nav
    report["active_return"] = report.net_return-report.bench
    report["relative_excess_nav"] = report.portfolio_nav/report.benchmark_nav
    periods = {"all":original.metrics(report), **{str(y):original.metrics(g) for y,g in report.groupby(report.index.year)}}
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for filename, frame in (("daily", report), ("orders", orders), ("holdings", holding), ("contributions", pnl)):
        path = directory / f"{filename}.parquet"
        frame.to_parquet(path, index=filename=="daily")
        files.append(path)
    for filename, records in (("stop_events", strategy.stop_events), ("generation_diagnostics", market.generation_journal)):
        path = directory / f"{filename}.json"
        save(path, records)
        files.append(path)
    stop_fills = trades.loc[trades.stop_loss]
    save(complete, dict(mode=mode, offset=offset, protocol_sha256=digest(out/"protocol.json"),
        periods=periods, triggers=len(strategy.stop_events), stop_sell_fills=len(stop_fills),
        stop_notional=float(stop_fills.notional.sum()),
        unresolved_stops=sorted(set(strategy.pending)&set(positions[days[-1]].get_stock_list())),
        account_audit_passed=True, artifacts={str(f):digest(f) for f in files}))
    print(f"Completed {mode}_{offset}: {periods['all']}", flush=True)


def report_results(out):
    import html
    import numpy as np
    import pandas as pd
    from scripts.analyze_s6_realized_sales import reconstruct_sales, stock_date_sales
    protocol = verify_binding(out)
    for path, value in protocol["sources"].items():
        if digest(path) != value: raise ValueError(f"Source changed: {path}")
    cases, accounts = {}, []
    for mode in ("top20_reference", *CASES):
        entries = []
        for offset in range(5):
            directory = original.OUT/f"accounts/baseline_{offset}" if mode == "top20_reference" else out/f"accounts/{mode}_{offset}"
            entry = read(directory / "complete.json")
            assert entry["account_audit_passed"]
            if mode != "top20_reference": assert entry["protocol_sha256"] == digest(out / "protocol.json")
            for path, value in entry["artifacts"].items(): assert digest(path) == value
            orders = pd.read_parquet(directory / "orders.parquet")
            fills, _ = reconstruct_sales(orders)
            sales = stock_date_sales(fills, pd.read_parquet(directory / "holdings.parquet"))
            wins, losses = sales.loc[sales.net_class.eq("profit")], sales.loc[sales.net_class.eq("loss")]
            entry["case"] = mode
            entry["trade_metrics"] = dict(win_rate=len(wins)/len(sales), mean_win=float(wins.net_return.mean()),
                mean_loss=float(losses.net_return.mean()), worst_sale=float(sales.net_return.min()))
            entries.append(entry)
        summary = {}
        for period in entries[0]["periods"]:
            rows = [e["periods"][period] for e in entries]
            summary[period] = {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            summary[period]["worst_phase_drawdown"] = max(r["max_drawdown"] for r in rows)
        cases[mode] = dict(periods=summary, mean_stop_fills=float(np.mean([e["stop_sell_fills"] for e in entries])),
            worst_sale=min(e["trade_metrics"]["worst_sale"] for e in entries),
            phase_cagr_std=float(np.std([e["periods"]["all"]["cagr"] for e in entries])),
            trade_metrics={k:float(np.mean([e["trade_metrics"][k] for e in entries])) for k in entries[0]["trade_metrics"]})
        accounts.extend(entries)
    paired = []
    for offset in range(5):
        a = next(e for e in accounts if e["case"] == "top12" and e["offset"] == offset)
        b = next(e for e in accounts if e["case"] == "top12_stop15" and e["offset"] == offset)
        paired.append(dict(offset=offset, cagr_delta=b["periods"]["all"]["cagr"]-a["periods"]["all"]["cagr"],
            drawdown_delta=b["periods"]["all"]["max_drawdown"]-a["periods"]["all"]["max_drawdown"]))
    save(out / "comparison.json", dict(status="complete", source_hashes_unchanged=True,
        protocol_sha256=digest(out/"protocol.json"), cases=cases, accounts=accounts, paired_differences=paired))
    labels = ["原版Top20，无止损", "Top12，无止损", "Top12，15%止损"]
    def table(headers, rows):
        return "<table><tr>"+"".join(f"<th>{html.escape(str(x))}</th>" for x in headers)+"</tr>"+"".join(
            "<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)+"</table>"
    overall = []
    for key, label, fmt in (("cagr","平均复合年化",".2%"), ("total_return","平均累计收益",".2%"),
        ("max_drawdown","平均最大回撤",".2%"), ("worst_phase_drawdown","最差起点回撤",".2%"),
        ("fees","全期平均费用（元）",",.2f"), ("mean_holdings","平均实际持股数",".2f")):
        overall.append([label, *[format(c["periods"]["all"][key],fmt) for c in cases.values()]])
    years = [[str(y) if y!=2026 else "2026截至10月13日", *[f"{c['periods'][str(y)]['total_return']:.2%}" for c in cases.values()]] for y in range(2022,2027)]
    weight = protocol["head_weight"]
    text = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>S6 Top12与15%止损</title>
<style>body{{max-width:1150px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#173448}}table{{width:100%;border-collapse:collapse;margin:20px 0}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #dce5ed}}th{{background:#edf4f8}}th:first-child,td:first-child{{text-align:left}}</style>
<h1>S6：Top12与15%止损</h1><p>2022-01-04至2026-10-13；相同的5个调仓起点，含成本。复用同一S6 QREG季度重训预测，无新增训练。</p>
<p>Top12权重：前6只合计{weight:.0%}股票仓位，后6只合计{1-weight:.0%}；总股票目标95%，每5日调仓，普通替换上限仍为6只（占12只的50%，不等于原Top20的30%）。原Top20参照维持前6只合计40%。因此Top20与Top12差异不能单独归因于持股数量；Top12两组之间才是止损的配对对照。</p>
<h2>全期</h2>{table(['指标',*labels],overall)}<h2>逐年收益（非年化）</h2>{table(['年份',*labels],years)}
<p>年化是每条账户按252交易日计算复合年化，再对5个起点取算术平均。最大回撤从日终净值历史高点计算，包含初始净值，不是单股止损线。</p>
<h2>止损口径</h2><p>单股相对含买入费用的移动加权平均成本亏损至少15%，收盘确认后在下个可交易日开盘卖出；不是盘中精确按−15%成交，也不保证最大损失不超过15%。停牌/跌停阻断时持续等待。止损日不回买，之后正常调仓可重新入选；非调仓日止损所得持有现金，不改其他股票数量。止损优先于普通6只替换上限。</p>
<h2>审计与限制</h2><p>10条新账户及5条已核验原版账户。净值、现金、费用、持股上限、区间与数据哈希已核对；模拟盘、部署及历史封存未修改。沿用原回测的整手规则、统一±9.5%涨跌幅过滤及日线理想开盘成交，不代表真实交易保证。未混入近期执行器整手修复。2022—2026已经反复研究，不是独立测试。</p>
<p>协议SHA256：{digest(out/'protocol.json')}。逐起点、逐年指标及订单记录见comparison.json和accounts。</p></html>"""
    (out/"report.html").write_text(text,encoding="utf-8")
    save(out/"report_manifest.json",{f.name:digest(f) for f in (out/"comparison.json",out/"report.html")})
    print(json.dumps(cases,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-weight", type=float, choices=[.4,.5,.6], required=True)
    parser.add_argument("--stage", choices=["audit","account","report","all"], default="all")
    parser.add_argument("--mode", choices=list(CASES), default="top12_stop15")
    parser.add_argument("--offset",type=int,choices=range(5),default=0)
    args = parser.parse_args()
    out = output_dir(args.head_weight)
    if args.stage == "audit": audit(out,args.head_weight)
    elif args.stage == "account": account(out,args.head_weight,args.mode,args.offset)
    elif args.stage == "report": report_results(out)
    else:
        audit(out,args.head_weight)
        for offset in range(5):
            for mode in CASES:
                save(out/"status.json",dict(state="running",mode=mode,offset=offset,pid=os.getpid(),updated=time.time()))
                logdir=out/"logs"
                logdir.mkdir(exist_ok=True)
                with (logdir/f"{mode}_{offset}.log").open("a",encoding="utf-8") as log:
                    result=subprocess.run([sys.executable,"-u","-m","paper_futu.research_s6_top12_stop15",
                        "--head-weight",str(args.head_weight),"--stage","account","--mode",mode,"--offset",str(offset)],
                        cwd=ROOT,env={**os.environ,**ENV,"PYTHONWARNINGS":"once"},stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    save(out/"status.json",dict(state="failed",mode=mode,offset=offset,exit_code=result.returncode))
                    raise SystemExit(result.returncode)
                print(f"Completed {mode}_{offset}",flush=True)
        report_results(out)
        save(out/"status.json",dict(state="complete",updated=time.time()))


if __name__ == "__main__": main()
