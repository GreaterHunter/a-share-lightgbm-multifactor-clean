"""Paired frozen-year and full-history cadence results, accounting and attribution."""
from __future__ import annotations

from html import escape
import math
from pathlib import Path
import time

from paper_futu import research_s6_2026 as study

LABELS = {"V0":"原版年度主模型", "V2":"近5年窗口", "V3":"近10年窗口", "V4":"近期样本加权",
          "V7":"L1、L2联合增强", "V8":"降低树复杂度", "M6":"半年重训", "M3":"季度重训"}


def safe(value):
    if isinstance(value, dict):
        return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [safe(v) for v in value]
    if hasattr(value,"item"):
        return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def statistics(frame):
    """Subperiod rates compound from their own opening capital; include initial NAV in MDD."""
    import numpy as np
    wealth = np.r_[1.,(1+frame.net_return).cumprod().to_numpy()]
    benchmark = float((1+frame.bench).prod())
    return dict(trading_days=len(frame), total_return=float(wealth[-1]-1),
        annualized_return=float(wealth[-1]**(252/len(frame))-1),
        benchmark_return=benchmark-1, geometric_excess=float(wealth[-1]/benchmark-1),
        max_drawdown=float(-(wealth/np.maximum.accumulate(wealth)-1).min()),
        fees_cny=float(frame.fees_cash.sum()), daily_turnover=float(frame.turnover.mean()),
        average_holdings=float(frame.holdings_count.mean()), minimum_holdings=int(frame.holdings_count.min()),
        maximum_holdings=int(frame.holdings_count.max()), bank_weight=float(frame.bank_weight.mean()),
        largest_stock_weight=float(frame.largest_stock_weight.max()),
        start=str(frame.index[0].date()), end=str(frame.index[-1].date()))


def periods(frame):
    yield "all","all",frame
    for kind,values in (("year",frame.index.year.astype(str)),
                        ("quarter",frame.index.to_period("Q").astype(str)),
                        ("month",frame.index.to_period("M").astype(str))):
        for name,part in frame.groupby(values):
            yield kind,str(name),part
    # Fixed paired windows isolate differences after the first model update.
    for name,boundary in (("after_first_quarter_update","2022-04-01"),
                          ("after_first_half_update","2022-07-01")):
        if frame.index[0].year == 2022:
            yield "update_window",name,frame.loc[frame.index >= boundary]
    for name,boundary in (("2026_after_quarter_update","2026-04-01"),
                          ("2026_after_half_update","2026-07-01")):
        yield "update_window",name,frame.loc[frame.index >= boundary]


def build_report():
    import numpy as np
    import pandas as pd
    from paper_futu import overfit_four_core as core
    p = study.check_sources()
    rows,stocks,dailies,account_artifacts = [],[],{},{}
    results = []
    for job in study.account_jobs():
        name = f"{job['mode']}_{job['case']}_{job['offset']}"
        result = study.cached("account",name)
        if result is None:
            raise ValueError(f"Account incomplete: {name}")
        results.append(dict(**job,directory=result["directory"],source_reused=False))
    results += [dict(mode="history",case="V0",offset=a["offset"],directory=a["directory"],source_reused=True)
                for a in p["historical_baseline_accounts"]]
    if len(results) != 55:
        raise ValueError("Expected 40 fresh accounts and 15 historical accounts")
    for result in results:
        directory = Path(result["directory"])
        core.verify_account(directory)
        daily = pd.read_parquet(directory / "daily.parquet")
        holding = pd.read_parquet(directory / "holdings.parquet")
        orders = pd.read_parquet(directory / "orders.parquet")
        contribution = pd.read_parquet(directory / "contributions.parquet")
        expected_start = study.START if result["mode"] == "fresh" else study.HISTORY_START
        expected_days = study.calendar()
        expected_days = expected_days[(expected_days >= expected_start) & (expected_days <= study.END)]
        if not daily.index.equals(expected_days):
            raise ValueError("Account interval differs")
        weights = holding.assign(weight=holding.market_value/holding.datetime.map(daily.account))
        daily["bank_weight"] = weights.loc[weights.instrument.isin(p["banks"])].groupby("datetime").weight.sum().reindex(daily.index,fill_value=0)
        daily["largest_stock_weight"] = weights.groupby("datetime").weight.max().reindex(daily.index,fill_value=0)
        fills = orders.loc[orders.notional.gt(0)]
        expected_fees = np.maximum(5.,fills.notional*np.where(fills.direction.eq(1),.0005,.0015))
        np.testing.assert_allclose(fills.fee,expected_fees,atol=1e-7,rtol=1e-10)
        account_value = holding.groupby("datetime").market_value.sum().reindex(daily.index,fill_value=0)
        np.testing.assert_allclose(account_value+daily.cash,daily.account,atol=1e-4,rtol=1e-10)
        stock_fees = fills.groupby(["datetime","instrument"]).fee.sum()
        stock_pnl = contribution.set_index(["datetime","instrument"]).pnl_before_fees
        net = stock_pnl.sub(stock_fees,fill_value=0).rename("net_pnl").reset_index()
        identity = {k:result[k] for k in ("mode","case","offset")}
        dailies[(result["mode"],result["case"],result["offset"])] = daily
        prior = daily.account.shift(1,fill_value=1e6)
        for kind,name,part in periods(daily):
            rows.append(dict(**identity,period_kind=kind,period=name,**statistics(part)))
            if kind in ("all","year"):
                value = net.loc[net.datetime.isin(part.index)].groupby("instrument").net_pnl.sum()
                initial = float(prior.loc[part.index[0]])
                np.testing.assert_allclose(value.sum(),part.account.iloc[-1]-initial,atol=1e-4,rtol=1e-10)
                for instrument,pnl in value.items():
                    stocks.append(dict(**identity,period_kind=kind,period=name,instrument=instrument,
                        net_pnl_cny=float(pnl),return_contribution=float(pnl/initial),bank=instrument in p["banks"]))
        account_artifacts[f"{result['mode']}_{result['case']}_{result['offset']}"] = {
            str(f):study.digest(f) for f in directory.iterdir() if f.is_file()}
    # Before the first update, the annual and rolling models/accounts must be identical.
    for mode,boundary_year in (("history",2022),("fresh",2026)):
        for offset in range(5):
            base = dailies[(mode,"V0",offset)]
            for case,month in (("M6",7),("M3",4)):
                mask = base.index < pd.Timestamp(boundary_year,month,1)
                other = dailies[(mode,case,offset)]
                np.testing.assert_allclose(base.loc[mask,["account","fees_cash","cash"]],
                    other.loc[mask,["account","fees_cash","cash"]],atol=1e-5,rtol=1e-10)
    frame = pd.DataFrame(rows)
    frame.to_csv(study.OUT / "account_period_metrics.csv",index=False,encoding="utf-8-sig")
    grouped = []
    for key,part in frame.groupby(["mode","case","period_kind","period"],sort=False):
        if len(part) != 5 or sorted(part.offset) != list(range(5)):
            raise ValueError("Every comparison needs five offsets")
        row = dict(zip(("mode","case","period_kind","period"),key))
        row.update(mean_return=float(part.total_return.mean()),median_return=float(part.total_return.median()),
            phase_return_std=float(part.total_return.std(ddof=0)),min_return=float(part.total_return.min()),
            max_return=float(part.total_return.max()),mean_cagr=float(part.annualized_return.mean()),
            phase_cagr_std=float(part.annualized_return.std(ddof=0)),worst_drawdown=float(part.max_drawdown.max()),
            mean_excess=float(part.geometric_excess.mean()),benchmark_return=float(part.benchmark_return.iloc[0]),
            mean_fees_cny=float(part.fees_cny.mean()),mean_daily_turnover=float(part.daily_turnover.mean()),
            mean_holdings=float(part.average_holdings.mean()),bank_weight=float(part.bank_weight.mean()),
            max_stock_weight=float(part.largest_stock_weight.max()),trading_days=int(part.trading_days.iloc[0]))
        np.testing.assert_allclose(part.benchmark_return,part.benchmark_return.iloc[0],atol=1e-15)
        grouped.append(row)
    summary = pd.DataFrame(grouped)
    summary.to_csv(study.OUT / "summary.csv",index=False,encoding="utf-8-sig")
    stock = pd.DataFrame(stocks)
    # Zero-fill absent stocks across phases before averaging their contributions.
    stock = stock.groupby(["mode","case","period_kind","period","instrument","bank"],as_index=False)[["net_pnl_cny","return_contribution"]].sum()
    stock[["net_pnl_cny","return_contribution"]] /= 5
    stock.to_csv(study.OUT / "stock_contributions.csv",index=False,encoding="utf-8-sig")
    paired = []
    for mode in ("fresh","history"):
        targets = study.CASES[1:] if mode == "fresh" else ("M6","M3")
        for case in targets:
            a = frame.loc[frame["mode"].eq(mode) & frame.case.eq("V0")]
            b = frame.loc[frame["mode"].eq(mode) & frame.case.eq(case)]
            merge = b.merge(a,on=["offset","period_kind","period"],suffixes=("_new","_base"))
            for _,row in merge.iterrows():
                paired.append(dict(mode=mode,case=case,offset=int(row.offset),period_kind=row.period_kind,period=row.period,
                    return_difference=float(row.total_return_new-row.total_return_base),
                    cagr_difference=float(row.annualized_return_new-row.annualized_return_base),
                    drawdown_difference=float(row.max_drawdown_new-row.max_drawdown_base),
                    fees_difference=float(row.fees_cny_new-row.fees_cny_base)))
    pd.DataFrame(paired).to_csv(study.OUT / "paired_differences.csv",index=False,encoding="utf-8-sig")
    signal_summary = signal_diagnostics()
    reference = study.read(study.ROOT / "reports/experiments/s6_head6_40_asof_20261013/comparison.json")
    carry_2026 = next(r for r in reference["yearly"] if r["setting"] == "S6_head6_40" and r["year"] == "2026")
    result = safe(dict(summary=grouped,signal_summary=signal_summary,paired_differences=paired,
        old_continuous_2026_reference=carry_2026,accounts=55,new_accounts=50,reused_accounts=5,
        new_fits=15,selected_main="V0",snapshot_id=p["snapshot_id"],
        all_initial_accounts_equal_before_first_update=True,limitations=p["limitations"]))
    study.save(study.OUT / "comparison.json",result)
    write_html(summary,stock,p)
    core.verify_protected(core.Store().protocol,full_snapshot=True)
    artifacts = {str(path):study.digest(path) for path in study.OUT.rglob("*") if path.is_file()
                 and path.suffix not in (".log",) and path.name not in
                 ("controller.lock","report_manifest.json","completion.json","status.json")}
    study.save(study.OUT / "report_manifest.json",dict(status="complete",implementation_sha256=study.implementation(),
        snapshot_id=p["snapshot_id"],sources=p["sources"],account_artifacts=account_artifacts,artifacts=artifacts,
        protected_unchanged=True,models_reloaded=True,annual_tree_choices={y:b["trees"] for y,b in p["annual_bindings"].items()},
        counts=dict(frozen_2026_variants=6,fresh_accounts=40,historical_accounts=15,new_fits=15)))
    study.save(study.OUT / "completion.json",dict(status="complete",finished=time.time(),accounts=55,
        new_accounts=50,reused_accounts=5,new_fits=15,report_sha256=study.digest(study.OUT / "comparison.json"),
        manifest_sha256=study.digest(study.OUT / "report_manifest.json"),protected_unchanged=True))
    study.save(study.OUT / "status.json",dict(status="complete",phase="report",updated=time.time(),
        accounts=55,new_accounts=50,reused_accounts=5,new_fits=15,main="V0"))
    print(summary.loc[summary.period_kind.eq("all"),["mode","case","mean_return","mean_cagr","worst_drawdown","mean_fees_cny"]].to_string(index=False),flush=True)


def signal_diagnostics():
    import pandas as pd
    from paper_futu import final_ablation_core as frozen
    labels = pd.read_parquet(study.DATA / "labels_h10.parquet")
    rows = []
    for name,stream in study.read(study.OUT / "streams.json")["streams"].items():
        pred = pd.read_parquet(stream["path"])
        joined = pred.join(labels[["label","label_end_date"]])
        joined = joined.loc[joined.label_end_date.le(pd.Timestamp(study.END)) & joined.label.notna()]
        ic = frozen.daily_ic(joined.prediction,joined.label)
        dates = pred.execution_date.groupby(level="datetime").first()
        ic["execution_date"] = dates.reindex(ic.index)
        if ic.execution_date.isna().any():
            raise ValueError("Missing IC execution date")
        ic.to_parquet(study.OUT / "predictions" / f"{name}_daily_ic.parquet")
        for kind,groups in (("all",[("all",ic)]),("year",ic.groupby(ic.execution_date.dt.year)),
                             ("quarter",ic.groupby(ic.execution_date.dt.to_period("Q").astype(str)))):
            for period,part in groups:
                rows.append(dict(stream=name,period_kind=kind,period=str(period),valid_days=int(part.rank_ic.notna().sum()),
                    mean_ic=float(part.ic.mean()),mean_rank_ic=float(part.rank_ic.mean()),
                    positive_rank_ic_ratio=float(part.rank_ic.dropna().gt(0).mean())))
    pd.DataFrame(rows).to_csv(study.OUT / "signal_ic_summary.csv",index=False,encoding="utf-8-sig")
    return safe(rows)


def write_html(summary,stocks,p):
    def table(frame,columns,labels):
        part = frame[columns].copy().rename(columns=labels)
        for c in part:
            if c in ("版本","年份","月份","股票代码"):
                if c == "版本": part[c] = part[c].map(LABELS)
                continue
            if "费用" in c: part[c] = part[c].map(lambda v:f"{v:,.0f}")
            else: part[c] = part[c].map(lambda v:f"{v:.2%}")
        return part.to_html(index=False,border=0,escape=True)
    common = ["case","mean_return","mean_cagr","worst_drawdown","phase_return_std","mean_excess","mean_fees_cny","bank_weight"]
    labels = dict(case="版本",mean_return="累计净收益",mean_cagr="平均年化",worst_drawdown="最差回撤",
        phase_return_std="收益起点标准差",mean_excess="几何超额",mean_fees_cny="平均费用（元）",bank_weight="银行平均仓位")
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>S6主策略与重训频率研究</title>
<style>body{font:16px/1.7 system-ui,'Microsoft YaHei';color:#20374a;max-width:1350px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:24px}th,td{padding:9px;text-align:right;border-bottom:1px solid #d8e0e7}th{background:#24465d;color:white}h1{font-size:27px}h2{font-size:21px}.scroll{overflow-x:auto}a{color:#176489}</style>
<h1>S6：前6只合计40%的训练消融与全历史重训对照</h1>
<p>主策略固定Top20，前6只占股票仓位40%，其余14只60%；股票目标仓位95%，每5个交易日调仓，最多更换6只。预测未来10日开盘→开盘收益排名。</p>
<p>费用为买入0.05%、卖出0.15%、每笔最低5元。表中为五起点均值；最差回撤为五个日收盘账户净值最大回撤中的最大值，包含初始净值。起点标准差不等于日收益波动率。累计净收益与年化分列。</p>
<h2>2026年空仓起步：6组训练规则</h2><p>2026-01-05至2026-10-13，各起点100万元现金开始；V0/V2/V3/V4/V7/V8训练及验证均不使用2026年标签。</p><div class="scroll">'''
    fresh = summary.loc[summary["mode"].eq("fresh") & summary.period_kind.eq("all")].set_index("case")
    body += table(fresh.loc[list(study.VARIANTS)].reset_index(),common,labels)+"</div>"
    body += "<h2>2022—2026全程：年度、半年、季度重训</h2><p>连续账户从2022-01-04的100万元现金开始，五个调仓起点跨年不重置。半年每年年初和7月更新；季度每年1、4、7、10月更新。每次只使用截至上一季/半年最后交易日已经成熟的标签。</p>"
    hist = summary.loc[summary["mode"].eq("history") & summary.period_kind.eq("all")].set_index("case")
    body += '<div class="scroll">'+table(hist.loc[["V0","M6","M3"]].reset_index(),common,labels)+"</div>"
    years = summary.loc[summary["mode"].eq("history") & summary.period_kind.eq("year")]
    body += "<h2>连续账户逐年净收益</h2>"+years.pivot(index="period",columns="case",values="mean_return")[["V0","M6","M3"]].rename(columns=LABELS).map(lambda x:f"{x:.2%}").to_html(border=0)
    body += "<h2>2026年空仓起步：重训频率对照</h2><p>与上述连续账户使用相同预测，但从2026年重新空仓建仓；两个表的2026收益不能混用。</p>"
    body += table(fresh.loc[["V0","M6","M3"]].reset_index(),common,labels)
    body += "<h2>首次更新后的配对区间</h2><p>历史连续账户分别比较自2022-04-01、2022-07-01起的相同期段；其中半年版本与年度模型在2022年7月前应完全一致。</p>"
    for period in ("after_first_quarter_update","after_first_half_update"):
        part = summary.loc[summary["mode"].eq("history") & summary.period.eq(period)].set_index("case")
        body += "<p>"+escape(period)+"</p>"+table(part.loc[["V0","M6","M3"]].reset_index(),common,labels)
    body += "<h2>主模型2026年主要盈利贡献</h2><p>金额为五起点均值；收益贡献为股票净损益除以相应账户期初净值，不是该股票自身涨幅。</p>"
    leaders = stocks.loc[stocks["mode"].eq("fresh") & stocks.case.eq("V0") & stocks.period_kind.eq("all")].nlargest(12,"net_pnl_cny")
    body += table(leaders,["instrument","net_pnl_cny","return_contribution"],dict(instrument="股票代码",net_pnl_cny="净盈利（费用后，元）",return_contribution="收益贡献"))
    body += "<h2>边界、复现与局限</h2><p>各年树数仅使用该年前三个完整年度验证折选择，年内不重新选树。年度树数为："+escape(str({y:b["trees"] for y,b in p["annual_bindings"].items()}))+"。六组年度冻结对照与使用年内成熟数据的滚动对照分别解释。</p><ul>"
    body += "".join("<li>"+escape(s)+"</li>" for s in p["limitations"])+"</ul>"
    body += '<p><a href="comparison.json">完整数值</a> · <a href="summary.csv">汇总表</a> · <a href="paired_differences.csv">同起点配对差异</a> · <a href="signal_ic_summary.csv">IC</a> · <a href="stock_contributions.csv">股票贡献</a> · <a href="protocol.json">配置与训练绑定</a> · <a href="report_manifest.json">哈希与验收</a></p></html>'
    (study.OUT / "comparison.html").write_text(body,encoding="utf-8")
