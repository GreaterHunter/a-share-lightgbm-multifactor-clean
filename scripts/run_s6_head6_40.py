"""Five isolated S6 head-six/40% replays; no fits or deployment changes."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from functools import partial
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/s6_head6_40_asof_20261013"
REFERENCE = ROOT / "reports/experiments/s6_head5_40_asof_20261013"
_EXCHANGE = None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    from paper_futu.final_ablation_core import save_json
    save_json(path, value)


def init_qlib():
    from paper_futu.overfit_four_core import DATA
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=str(DATA / "provider"), region=REG_CN, kernels=1,
        exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                     "kwargs": {"uri": (OUT / "mlruns").as_uri(), "default_exp_name": OUT.name}})


def replay(job, key):
    """Preserve the frozen account logic, with new outputs and no V0 reuse path."""
    global _EXCHANGE
    import numpy as np
    import pandas as pd
    from qlib.data import D
    from qlib.backtest.decision import Order
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.evaluate import backtest_daily
    from paper_futu import final_ablation_core as frozen
    from paper_futu.final_ablation_account import AuditedExchange, AuditedStrategy, summarize_accounts
    from paper_futu.overfit_four_core import DATA, verify_account
    from src.compare_models import enrich_daily_report
    from src.runtime_budget import require_memory

    require_memory(2., "S6 head-six account")
    prediction = pd.read_parquet(job["prediction_path"])
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    trade_dates = calendar[(calendar >= frozen.START) & (calendar <= frozen.AS_OF)]
    schedule = trade_dates[job["offset"]::job["rebalance"]]
    signal_dates = calendar[calendar.get_indexer(schedule)-1]
    signal = prediction.loc[prediction.index.get_level_values("datetime").isin(signal_dates), ["prediction"]]
    if len(signal_dates) != signal.index.get_level_values("datetime").nunique():
        raise ValueError("Missing scheduled signals")
    if _EXCHANGE is None:
        schema = pd.read_parquet(DATA / "raw_features.parquet", columns=[])
        codes = sorted(schema.index.get_level_values("instrument").unique())
        _EXCHANGE = AuditedExchange(freq="day", start_time=calendar[calendar < frozen.START][-1],
            end_time=frozen.AS_OF, codes=codes, deal_price="open", open_cost=.0005, close_cost=.0015,
            min_cost=5, trade_unit=100, limit_threshold=(
                "($open / Ref($close, 1) - 1) >= 0.095", "($open / Ref($close, 1) - 1) <= -0.095"))
    market = _EXCHANGE
    market.reset_audit(job["friction"], job["portfolio"]["top_k"])
    strategy = AuditedStrategy(signal=signal, risk_degree=.95, only_tradable=True, **job["portfolio"])
    module = importlib.import_module("qlib.backtest.backtest")
    with patch.object(module, "tqdm", partial(module.tqdm, disable=True)):
        raw, positions = backtest_daily(start_time=calendar[calendar < frozen.START][-1], end_time=frozen.AS_OF,
            strategy=strategy, executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True,
                verbose=False, trade_type=SimulatorExecutor.TT_SERIAL), account=1_000_000,
            benchmark="SH000300", exchange_kwargs={"exchange": market})
    report = enrich_daily_report(raw.reindex(trade_dates), key)
    if report[["net_return", "cost", "account"]].isna().any().any():
        raise ValueError("Missing common comparison interval")
    opening = D.features(["SH000300"], ["$open", "$close"], start_time=trade_dates[0], end_time=trade_dates[0]).iloc[0]
    report.loc[trade_dates[0], "bench"] = opening.iloc[1]/opening.iloc[0]-1
    report.loc[frozen.AS_OF, "bench"] = read(DATA / "benchmark_audit.json")["last_daily_return"]
    if report.bench.isna().any():
        raise ValueError("Missing benchmark")
    report["active_return"] = report.net_return-report.bench
    report["benchmark_nav"] = (1+report.bench).cumprod()
    report["relative_excess_nav"] = report.portfolio_nav/report.benchmark_nav
    prior = report.account.shift(1, fill_value=1_000_000.)
    orders = pd.DataFrame(market.journal)
    trades = orders.loc[orders.notional.gt(0)]
    totals = trades.groupby("datetime")[["notional", "fee"]].sum().reindex(trade_dates, fill_value=0)
    report["fees_cash"], report["traded_notional"] = totals.fee, totals.notional
    report["turnover"] = totals.notional/prior
    np.testing.assert_allclose(report.cost*prior, totals.fee, atol=1e-5, rtol=1e-10)
    holdings, contributions, previous_values = [], [], {}
    for day in trade_dates:
        position = positions[day]
        market.check_holding_limit(position, day, "daily_close")
        current_values = {}
        for stock in position.get_stock_list():
            quantity, price = position.get_stock_amount(stock), position.get_stock_price(stock)
            value = quantity*price
            current_values[stock] = value
            holdings.append(dict(datetime=day, instrument=stock, adjusted_quantity=quantity,
                                 adjusted_close=price, market_value=value, cash=position.get_cash()))
        today = trades.loc[trades.datetime.eq(day)]
        flows = today.assign(flow=np.where(today.direction.eq(Order.BUY), today.notional, -today.notional)).groupby("instrument").flow.sum()
        for stock in set(previous_values) | set(current_values) | set(flows.index):
            contributions.append(dict(datetime=day, instrument=stock,
                pnl_before_fees=current_values.get(stock, 0)-previous_values.get(stock, 0)-flows.get(stock, 0)))
        previous_values = current_values
        if position.get_cash() < -1e-5:
            raise ValueError("Negative account cash")
    holding_frame = pd.DataFrame(holdings)
    pnl_frame = pd.DataFrame(contributions)
    report["holdings_count"] = holding_frame.groupby("datetime").size().reindex(trade_dates, fill_value=0)
    gross_pnl = pnl_frame.groupby("datetime").pnl_before_fees.sum().reindex(trade_dates, fill_value=0)
    np.testing.assert_allclose(prior+gross_pnl-totals.fee, report.account, atol=1e-4, rtol=1e-10)
    np.testing.assert_allclose((1+report.net_return).cumprod()*1e6, report.account, atol=1e-4, rtol=1e-10)
    rows = summarize_accounts(report, key)
    year_rows = [r for r in rows if r["period"].isdigit()]
    all_row = next(r for r in rows if r["period"] == "all")
    if not np.isclose(np.prod([1+r["total_return"] for r in year_rows])-1, all_row["total_return"], atol=1e-10):
        raise ValueError("Annual returns do not compound")
    directory = OUT / "accounts" / key
    directory.mkdir(parents=True, exist_ok=True)
    report.to_parquet(directory/"daily.parquet")
    orders.to_parquet(directory/"orders.parquet", index=False)
    pd.DataFrame(market.generation_journal, columns=["datetime", "instrument", "direction", "reason"]).to_parquet(
        directory/"generation_diagnostics.parquet", index=False)
    holding_frame.to_parquet(directory/"holdings.parquet", index=False)
    pnl_frame.to_parquet(directory/"contributions.parquet", index=False)
    for row in rows:
        row.update(case_id=key, **job)
    pd.DataFrame(rows).to_csv(directory/"metrics.csv", index=False, encoding="utf-8-sig")
    audit = verify_account(directory)
    return dict(directory=str(directory), rows=rows, audit=audit)


def worker(offsets):
    protocol = read(OUT / "protocol.json")
    assert digest(__file__) == protocol["adapter_sha256"]
    assert digest(protocol["reference_config"]["prediction_path"]) == protocol["prediction_sha256"]
    init_qlib()
    for offset in offsets:
        job = {**protocol["reference_config"], "portfolio": dict(top_k=20, top_tier_size=6,
               top_tier_weight=.4, n_drop=6), "offset": offset}
        from paper_futu import final_ablation_core as frozen
        key = frozen.identity(dict(config=job, adapter=protocol["adapter_sha256"],
                                   snapshot=protocol["snapshot_id"]))[:24]
        path = OUT / "tasks" / f"offset_{offset}.json"
        if path.exists():
            prior = read(path)
            if prior["status"] == "complete":
                assert prior["job"] == job and prior["adapter_sha256"] == protocol["adapter_sha256"]
                for artifact, expected in prior["artifacts"].items():
                    assert digest(artifact) == expected
                continue
            raise RuntimeError(f"Previous incomplete task needs review: {path}")
        started = time.time()
        record = dict(status="running", offset=offset, pid=os.getpid(), job=job,
                      adapter_sha256=protocol["adapter_sha256"], started=started)
        save(path, record)
        try:
            result = replay(job, key)
            record.update(status="complete", result=result, elapsed_seconds=time.time()-started,
                          artifacts={str(p): digest(p) for p in Path(result["directory"]).iterdir() if p.is_file()})
            save(path, record)
            print(f"COMPLETE offset={offset} seconds={record['elapsed_seconds']:.1f}", flush=True)
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            save(path, record)
            raise


def audit_and_run():
    import psutil
    from paper_futu import overfit_four_core as core
    from src.runtime_budget import require_memory
    running = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid != os.getpid() and "scripts.run_s6_head6_40" in (process.info["cmdline"] or []):
            running.append(process.pid)
    if running:
        raise RuntimeError(f"Existing matching process: {running}")
    store = core.Store()
    core.verify_protected(store.protocol)
    reference = read(REFERENCE / "verified_accounts.json")
    assert reference["status"] == "passed" and reference["snapshot_id"] == store.protocol["snapshot_id"]
    ref_manifest = read(REFERENCE / "report_manifest.json")
    assert ref_manifest["source_audit_sha256"] == digest(REFERENCE / "verified_accounts.json")
    for account in reference["accounts"]:
        for path, expected in account["artifacts"].items():
            assert digest(path) == expected
        core.verify_account(Path(account["directory"]))
    family = store.protocol["families"]["raw10"]
    assert reference["prediction_sha256"] == family["prediction_sha256"]
    assert digest(family["prediction_path"]) == reference["prediction_sha256"]
    # Same frozen inputs/configuration; only head size and weight change.
    assert reference["config"]["portfolio"] == dict(top_k=20, top_tier_size=5, top_tier_weight=.25, n_drop=6)
    protocol = dict(reference_config=reference["config"], snapshot_id=reference["snapshot_id"],
        prediction_sha256=reference["prediction_sha256"], adapter_sha256=digest(__file__),
        source_science_sha256=core.frozen.code_hash(), source_replay_sha256=digest(ROOT / "paper_futu/overfit_four_account.py"),
        reference_audit_sha256=digest(REFERENCE / "verified_accounts.json"),
        offsets=list(range(5)), new_fits=0, new_accounts=5, min_start_gib=3., min_task_gib=2.)
    if (OUT / "protocol.json").exists():
        assert read(OUT / "protocol.json") == protocol, "Existing experiment binding changed"
    else:
        save(OUT / "protocol.json", protocol)
    active, logs = [], []
    try:
        for lane, offsets in enumerate(["0,2,4", "1,3"]):
            require_memory(3., f"Launch S6 lane {lane}")
            stdout = (OUT / f"lane{lane}.stdout.log").open("a", encoding="utf-8")
            stderr = (OUT / f"lane{lane}.stderr.log").open("a", encoding="utf-8")
            logs.extend([stdout, stderr])
            env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                   "NUMEXPR_NUM_THREADS": "4", "PYTHONUTF8": "1"}
            process = subprocess.Popen([sys.executable, "-u", "-m", "scripts.run_s6_head6_40",
                "--stage", "worker", "--offsets", offsets], cwd=ROOT, stdout=stdout, stderr=stderr,
                env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            active.append(process)
            save(OUT / "controller.json", dict(status="running", pid=os.getpid(),
                 workers=[dict(pid=p.pid, create_time=psutil.Process(p.pid).create_time()) for p in active]))
            time.sleep(1)
        while any(p.poll() is None for p in active):
            time.sleep(1)
        if any(p.returncode for p in active):
            raise RuntimeError(f"Worker failed; inspect lane logs: {[p.returncode for p in active]}")
    finally:
        for p in active:
            p.wait()
        for log in logs:
            log.close()
    import numpy as np
    import pandas as pd
    base_daily = pd.read_parquet(Path(reference["accounts"][0]["directory"]) / "daily.parquet")
    rows, accounts = list(reference["rows"]), []
    for offset in range(5):
        task = read(OUT / "tasks" / f"offset_{offset}.json")
        assert task["status"] == "complete"
        assert task["adapter_sha256"] == protocol["adapter_sha256"]
        for path, expected in task["artifacts"].items():
            assert digest(path) == expected
        directory = Path(task["result"]["directory"])
        core.verify_account(directory)
        daily = pd.read_parquet(directory / "daily.parquet")
        assert daily.index.equals(base_daily.index)
        np.testing.assert_allclose(daily.bench, base_daily.bench, atol=1e-15, rtol=0)
        for row in task["result"]["rows"]:
            if str(row["period"]) in ["all", "2022", "2023", "2024", "2025", "2026"]:
                rows.append(dict(setting="S6_head6_40", offset=offset, period=str(row["period"]),
                                 **{f: row[f] for f in ["total_return", "annualized_return", "max_drawdown",
                                    "total_fees_cny", "average_daily_turnover", "average_holdings", "total_excess_return"]}))
        accounts.append(dict(offset=offset, directory=str(directory), artifacts=task["artifacts"]))
    core.verify_protected(store.protocol)
    assert len(rows) == 90 and len(accounts) == 5
    save(OUT / "verified_accounts.json", dict(status="passed", rows=rows, new_accounts=accounts,
        reused_accounts=10, new_replays=5, new_fits=0, common_interval=True, protected_unchanged=True,
        start=reference["start"], end=reference["end"], trading_days=reference["trading_days"],
        protocol_sha256=digest(OUT / "protocol.json")))
    save(OUT / "controller.json", dict(status="complete", pid=os.getpid(), completed_accounts=5))
    print(json.dumps(dict(status="passed", new_accounts=5, new_fits=0)), flush=True)


def report():
    # Reporting also runs under the bundled analysis runtime, without Qlib/YAML.
    # This local writer does not alter the replay or its frozen task bindings.
    def save(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    import numpy as np
    import pandas as pd
    audit = read(OUT / "verified_accounts.json")
    assert audit["status"] == "passed" and audit["protected_unchanged"]
    source = pd.DataFrame(audit["rows"])
    overall, yearly = [], []
    settings = ["S6_equal", "S6_head5_40", "S6_head6_40"]
    labels = dict(zip(settings, ["S6等权", "前5只合计40%", "前6只合计40%"]))
    for setting in settings:
        all_rows = source.loc[source.setting.eq(setting) & source.period.eq("all")].sort_values("offset")
        assert all_rows.offset.tolist() == list(range(5))
        overall.append(dict(setting=setting, mean_cagr=float(all_rows.annualized_return.mean()),
            phase_std_cagr=float(np.std(all_rows.annualized_return, ddof=0)),
            worst_drawdown=float(-all_rows.max_drawdown.min()), mean_fees_cny=float(all_rows.total_fees_cny.mean()),
            mean_daily_turnover=float(all_rows.average_daily_turnover.mean()),
            average_holdings=float(all_rows.average_holdings.mean())))
        for year in ["2022", "2023", "2024", "2025", "2026"]:
            frame = source.loc[source.setting.eq(setting) & source.period.eq(year)]
            assert len(frame) == 5
            yearly.append(dict(setting=setting, year=year, mean_return=float(frame.total_return.mean()),
                               mean_geometric_excess=float(frame.total_excess_return.mean())))
    result = dict(overall=overall, yearly=yearly, phases=source.loc[source.period.eq("all")].to_dict("records"),
                  source_audit_sha256=digest(OUT / "verified_accounts.json"))
    save(OUT / "comparison.json", result)
    table = pd.DataFrame(overall).replace({"setting": labels})
    for name in ["mean_cagr", "phase_std_cagr", "worst_drawdown", "mean_daily_turnover"]:
        table[name] = table[name].map(lambda v: f"{v:.2%}")
    table.mean_fees_cny = table.mean_fees_cny.map(lambda v: f"{v/10000:.2f}万")
    table = table.drop(columns="average_holdings").rename(columns=dict(setting="版本", mean_cagr="平均年化",
        phase_std_cagr="起点标准差", worst_drawdown="最差回撤", mean_fees_cny="全期平均费用", mean_daily_turnover="平均日双边换手"))
    years = pd.DataFrame(yearly).pivot(index="year", columns="setting", values="mean_return")[settings]
    years = years.map(lambda v: f"{v:.2%}").rename(columns=labels).rename_axis("年份")
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>S6前六只40%对照</title>
<style>body{font:16px/1.7 system-ui,'Microsoft YaHei';max-width:1100px;margin:35px auto;padding:0 18px;color:#183047}table{border-collapse:collapse;width:100%}th,td{text-align:right;padding:9px;border-bottom:1px solid #ddd}th{background:#24475e;color:white}h1{font-size:26px}</style>
<h1>S6：前六只合计40%与等权、前五只40%对照</h1>
<p>2022-01-04至2026-10-13；同一原版S6预测、Top20、每5个交易日调仓、最多替换6只；五起点0/1/2/3/4连续跨年。仅改变权重，不重训、不切换模拟盘。</p>
<p>股票目标仓位95%；新组前6只每只目标占账户6.3333%，其余14只每只4.0714%，另留5%现金。按保留股票的目标仓位差交易。买入0.05%、卖出0.15%、最低5元/笔，初始100万元。</p>
<h2>全期比较</h2><p>各起点先计算252交易日口径的复利年化，再取算术平均。标准差为五个起点年化的总体标准差；最差回撤为五个日收盘账户净值最大回撤中最差值。费用为全期实际费用均值。</p>'''
    body += table.to_html(index=False, border=0) + "<h2>逐年净收益（五起点均值）</h2><p>2026为截至10月13日累计收益，非年化。</p>" + years.to_html(border=0)
    body += '<p>只有前6只40%新增5个账户，其余10个账户经哈希和账务复核后复用。预测、快照、源代码绑定和账户审计见<a href="verified_accounts.json">验收记录</a>、<a href="protocol.json">配置</a>及<a href="comparison.json">数值结果</a>。</p><p>该历史区间已经反复用于研究，不能当成全新独立测试，也不因此自动选择新冠军。保留日线理想开盘成交、统一9.5%涨跌幅过滤和公司行动近似等原有局限。</p></html>'
    (OUT / "comparison.html").write_text(body, encoding="utf-8")
    save(OUT / "report_manifest.json", dict(status="complete",
        replay_adapter_sha256=read(OUT / "protocol.json")["adapter_sha256"], report_builder_sha256=digest(__file__),
        source_audit_sha256=digest(OUT / "verified_accounts.json"), comparison_sha256=digest(OUT / "comparison.json"),
        html_sha256=digest(OUT / "comparison.html"), new_accounts=5, reused_accounts=10, new_fits=0))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["run", "worker", "report"])
    parser.add_argument("--offsets", default="")
    args = parser.parse_args()
    if args.stage == "worker":
        worker([int(x) for x in args.offsets.split(",")])
    elif args.stage == "run":
        audit_and_run()
    else:
        report()
