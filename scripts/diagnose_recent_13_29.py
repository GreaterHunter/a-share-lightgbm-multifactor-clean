"""Read existing rank-13/29 accounts; no fitting, replay or deployment changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/experiments"
RECENT = BASE / "recent_2024_2026_ranking_asof_20261013"
OLD = BASE / "final_ablation_asof_20261013"
STUDY = BASE / "overfit_four_asof_20261013"
OUT = RECENT / "rank13_29_diagnosis"
CASES = {"rank13": "d66cb5d478400f2393bf", "rank29": "781bf58878e7c928e45d",
         "fixed13": "667ad48c19a88b281828", "fixed29": "16fa1240d66271ad5d29"}
STYLE = ["book_to_price", "earnings_to_price", "volatility_60d", "mom_20d", "mom_60d"]
PERIODS = {"2022_2023": ("2022-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-10-13"),
           "all": ("2022-01-01", "2026-10-13"), **{str(y): (f"{y}-01-01", f"{y}-12-31") for y in range(2022, 2027)}}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def extract():
    # The project runtime supplies Parquet support missing from the bundled runtime.
    import numpy as np
    import pandas as pd
    from paper_futu import overfit_four_core as core
    protocol = core.Store().protocol
    core.verify_protected(protocol)
    collected = read(RECENT / "collected.json")
    ranks = read(RECENT / "comparison.json")
    assert [ranks["top30_without_single_concentration"][i-1]["candidate_key"] for i in [13, 29]] == [CASES["rank13"], CASES["rank29"]]
    candidates = {c["candidate_key"]: c for c in collected["candidates"]}
    banks = protocol["bank_names"]
    names_file = ROOT / "data/external/fundamental_disclosure_dates.parquet"
    names_frame = pd.read_parquet(names_file, columns=["instrument", "stock_name", "report_date"])
    names = names_frame.dropna(subset=["stock_name"]).sort_values("report_date").drop_duplicates("instrument", keep="last").set_index("instrument").stock_name.to_dict()
    names.update(banks)
    style = pd.read_parquet(core.DATA / "rank_features.parquet", columns=STYLE,
                            filters=[("datetime", ">=", pd.Timestamp("2022-01-01"))])+.5
    rows, stocks, source_accounts, snapshots = [], [], [], []
    for label, key in CASES.items():
        candidate = candidates[key]
        for account in candidate["accounts"]:
            task_path = Path(account["task_path"])
            assert digest(task_path) == account["task_sha256"]
            task = read(task_path)
            directory = Path(account["directory"])
            for filename in ["daily.parquet", "metrics.csv", "holdings.parquet", "orders.parquet", "contributions.parquet"]:
                path = directory / filename
                assert digest(path) == task["artifacts"][str(path)], str(path)
            core.verify_account(directory)
            daily = pd.read_parquet(directory / "daily.parquet")
            holdings = pd.read_parquet(directory / "holdings.parquet")
            orders = pd.read_parquet(directory / "orders.parquet")
            pnl = pd.read_parquet(directory / "contributions.parquet")
            fills = orders.loc[orders.notional.gt(0)]
            fees = fills.groupby(["datetime", "instrument"]).fee.sum().rename("fee")
            profit = pnl.set_index(["datetime", "instrument"]).join(fees).fillna({"fee": 0})
            profit["net_pnl"] = profit.pnl_before_fees-profit.fee
            assert not fees.index.difference(profit.index).size
            np.testing.assert_allclose(profit.groupby(level="datetime").net_pnl.sum().reindex(daily.index, fill_value=0),
                daily.account.diff().fillna(daily.account.iloc[0]-1e6), rtol=1e-9, atol=1e-4)
            h = holdings.merge(daily[["account"]], left_on="datetime", right_index=True)
            h["weight"] = h.market_value/h.account
            hh = h.set_index(["instrument", "datetime"]).join(style)
            bank_weight = h.loc[h.instrument.isin(banks)].groupby("datetime").weight.sum().reindex(daily.index, fill_value=0.)
            exposure = pd.DataFrame(dict(bank_weight=bank_weight,
                largest_weight=h.groupby("datetime").weight.max().reindex(daily.index)))
            for field in STYLE:
                valid = hh.loc[hh[field].notna()]
                exposure[field] = (valid.market_value*valid[field]).groupby(level="datetime").sum()/valid.market_value.groupby(level="datetime").sum()
            for period, (start, end) in PERIODS.items():
                days = daily.loc[start:end]
                r = days.net_return.to_numpy(); nav = np.r_[1., np.cumprod(1+r)]
                prior = daily.account.shift(1, fill_value=1e6).loc[days.index[0]]
                net = profit.loc[profit.index.get_level_values("datetime").isin(days.index)].groupby(level="instrument").net_pnl.sum()
                assert np.isclose(net.sum()/prior, nav[-1]-1, atol=1e-9)
                log_returns = np.log1p(days.net_return)
                log2024 = float(log_returns.loc[log_returns.index.year == 2024].sum())
                row = dict(case=label, candidate_key=key, offset=int(account["offset"]), period=period,
                    trading_days=len(days), total_return=float(nav[-1]-1), cagr=float(nav[-1]**(252/len(days))-1),
                    max_drawdown=float(-np.min(nav/np.maximum.accumulate(nav)-1)),
                    benchmark_return=float((1+days.bench).prod()-1),
                    geometric_excess=float(nav[-1]/(1+days.bench).prod()-1),
                    fees_cny=float(days.fees_cash.sum()), average_daily_turnover=float(days.turnover.mean()),
                    bank_weight=float(exposure.loc[days.index].bank_weight.mean()),
                    largest_weight_mean=float(exposure.loc[days.index].largest_weight.mean()),
                    largest_weight_max=float(exposure.loc[days.index].largest_weight.max()),
                    bank_net_return_contribution=float(net.reindex(list(banks), fill_value=0).sum()/prior),
                    log_growth_share_2024=float(log2024/log_returns.sum()) if abs(log_returns.sum()) > 1e-10 else None,
                    **{field: float(exposure.loc[days.index, field].mean()) for field in STYLE})
                rows.append(row)
                if label.startswith("rank") and period in ["all", "2024_2026", "2022_2023", "2024", "2025", "2026"]:
                    part = h.loc[h.datetime.isin(days.index)]
                    weights = part.groupby("instrument").weight.sum()/len(days)
                    held_days = part.groupby("instrument").datetime.nunique()
                    for stock in weights.index.union(net.index):
                        stocks.append(dict(case=label, offset=int(account["offset"]), period=period, instrument=stock,
                            name=names.get(stock, stock), bank=stock in banks,
                            mean_nav_weight=float(weights.get(stock, 0)), held_day_fraction=float(held_days.get(stock, 0)/len(days)),
                            net_return_contribution=float(net.get(stock, 0)/prior), net_profit_cny=float(net.get(stock, 0))))
            for period, excluded in [("excluding2024", [2024]), ("excluding2022_2024", [2022, 2024])]:
                days = daily.loc[~daily.index.year.isin(excluded)]
                rows.append(dict(case=label, candidate_key=key, offset=int(account["offset"]), period=period,
                    cagr=float((1+days.net_return).prod()**(252/len(days))-1),
                    total_return=float((1+days.net_return).prod()-1), trading_days=len(days),
                    discontinuous_diagnostic=True))
            if label.startswith("rank"):
                last = h.loc[h.datetime.eq(daily.index[-1])].sort_values("weight", ascending=False)
                snapshots.extend(dict(case=label, offset=int(account["offset"]), instrument=r.instrument,
                    name=names.get(r.instrument, r.instrument), weight=float(r.weight)) for r in last.itertuples())
            source_accounts.append(dict(case=label, offset=int(account["offset"]), task_path=str(task_path),
                task_sha256=digest(task_path), directory=str(directory), artifacts=task["artifacts"]))
        print(f"EXTRACT {label}: 5 existing accounts reconciled", flush=True)
    signal_path = BASE / "all_full15_offsets_asof_20261013/cross_model_diagnostic/signal_yearly.csv"
    signal = pd.read_csv(signal_path)
    signal = signal.loc[signal.stream.isin(["lgb_15_raw_rank_h10_tuned_m12_s42", "lgb_15_raw_rank_h10_fixed_m12_s42"])]
    selections = {}
    for policy in ["fixed", "tuned"]:
        meta = read(OLD / f"predictions/lgb_15_raw_rank_h10_{policy}_m12_s42.json")
        assert digest(meta["prediction_path"]) == meta["prediction_sha256"]
        selections[policy] = [dict(year=p["selection_year"], trees=p["trees"], parameter_id=p["config"]["id"],
            parameters=p["config"]["parameters"], cutoff=p["cutoff"], train_signal_end=p["train_signal_end"],
            label_end_max=p["label_end_max"], model_sha256=p["model_sha256"]) for p in meta["periods"]]
        assert all(p["label_end_max"] <= p["cutoff"] for p in meta["periods"])
    core.verify_protected(protocol)
    save(OUT / "extracted.json", dict(status="passed", cases=CASES, rows=rows, stocks=stocks,
        latest_holdings=snapshots, signals=json.loads(signal.to_json(orient="records")), selections=selections,
        accounts=source_accounts, stock_name_source=str(names_file), stock_name_sha256=digest(names_file),
        source_collected_sha256=digest(RECENT / "collected.json"), signal_source_sha256=digest(signal_path),
        new_fits=0, new_replays=0, protected_unchanged=True))
    print("EXTRACT passed: 20 accounts, no new fits/replays", flush=True)


def report():
    import numpy as np
    import pandas as pd
    raw = read(OUT / "extracted.json")
    assert raw["status"] == "passed"
    data = pd.DataFrame(raw["rows"])
    summaries = []
    for (case, period), group in data.groupby(["case", "period"], sort=False):
        row = dict(case=case, period=period, phase_count=len(group), mean_cagr=float(group.cagr.mean()),
            phase_std_cagr=float(group.cagr.std(ddof=0)), min_cagr=float(group.cagr.min()), max_cagr=float(group.cagr.max()),
            mean_total_return=float(group.total_return.mean()))
        for field in ["max_drawdown", "benchmark_return", "geometric_excess", "fees_cny", "average_daily_turnover", "bank_weight",
                      "largest_weight_mean", "bank_net_return_contribution", "log_growth_share_2024", *STYLE]:
            if group[field].notna().any():
                row[field] = float(group[field].mean())
        if group.max_drawdown.notna().any():
            row["worst_drawdown"] = float(group.max_drawdown.max())
        if group.largest_weight_max.notna().any():
            row["largest_stock_weight_observed"] = float(group.largest_weight_max.max())
        assert len(group) == 5
        summaries.append(row)
    sf = pd.DataFrame(raw["stocks"])
    stock_rows = []
    for (case, period, code), group in sf.groupby(["case", "period", "instrument"]):
        # An absent stock in a phase contributes zero, not a missing denominator.
        stock_rows.append(dict(case=case, period=period, instrument=code, name=group.name.iloc[0], bank=bool(group.bank.iloc[0]),
            **{field: float(group[field].sum()/5) for field in ["mean_nav_weight", "held_day_fraction", "net_return_contribution", "net_profit_cny"]}))
    stocks = pd.DataFrame(stock_rows)
    details = []
    for case in ["rank13", "rank29"]:
        for period in ["all", "2022_2023", "2024_2026", "2024", "2025", "2026"]:
            part = stocks.loc[stocks.case.eq(case) & stocks.period.eq(period)]
            winners = part.sort_values("net_return_contribution", ascending=False)
            positive = winners.loc[winners.net_return_contribution.gt(0)]
            total = float(part.net_return_contribution.sum())
            details.append(dict(case=case, period=period,
                top_holdings=part.sort_values("mean_nav_weight", ascending=False).head(12).to_dict("records"),
                top_profit_contributors=winners.head(12).to_dict("records"),
                worst_profit_contributors=winners.tail(5).to_dict("records"),
                top1_share_of_positive_profit=float(positive.net_return_contribution.head(1).sum()/positive.net_return_contribution.sum()),
                top5_share_of_positive_profit=float(positive.net_return_contribution.head(5).sum()/positive.net_return_contribution.sum()),
                total_net_return_contribution=total,
                bank_share_of_net_profit=float(part.loc[part.bank].net_return_contribution.sum()/total) if abs(total)>1e-10 else None))
    # Paired annual outcomes: fixed vs tuned, same portfolio and each rebalance phase.
    paired = []
    for tuned, fixed in [("rank13", "fixed13"), ("rank29", "fixed29")]:
        for period in PERIODS:
            t=data.loc[data.case.eq(tuned)&data.period.eq(period)].set_index("offset")
            f=data.loc[data.case.eq(fixed)&data.period.eq(period)].set_index("offset")
            d=t.cagr-f.cagr
            paired.append(dict(case=tuned, fixed_control=fixed, period=period,
                mean_cagr_difference=float(d.mean()), tuned_winning_offsets=int(d.gt(0).sum())))
    output=dict(status="complete", summaries=summaries, stock_diagnostics=details, paired_controls=paired,
        signals=raw["signals"], selections=raw["selections"], latest_holdings=raw["latest_holdings"],
        extracted_sha256=digest(OUT / "extracted.json"), new_fits=0, new_replays=0)
    save(OUT / "diagnosis.json", output)
    def table(frame, pct=()):
        frame=frame.copy()
        for col in pct:
            frame[col]=frame[col].map(lambda v:f"{v:.2%}")
        return frame.to_html(index=False, border=0)
    summary=pd.DataFrame(summaries)
    years=summary.loc[summary.period.isin([str(y) for y in range(2022,2027)]) & summary.case.isin(["rank13","rank29"])]
    view=years.pivot(index="period",columns="case",values="mean_total_return").reset_index()
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>近期第13和29组诊断</title>
<style>body{font:15px/1.7 system-ui,'Microsoft YaHei';max-width:1250px;margin:30px auto;padding:0 18px;color:#183047}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;border-bottom:1px solid #ddd;padding:7px}th{background:#24475e;color:white}h1{font-size:26px}h2{font-size:20px;margin-top:28px}</style>
<h1>近期榜第13、29组：其他时期、持仓与过拟合风险</h1><p>两组共用原始15因子、H10开盘到开盘标签、年度调参及年度从头重训预测。第13组Top5等权最多换2只；第29组Top10前2只占股票仓位40%、最多换10只。95%股票目标仓位。所有统计基于现有5起点账户，未新增训练或回测，不改模拟盘。</p>
<h2>逐年成本后收益（5起点平均）</h2>'''
    body+=table(view,pct=["rank13","rank29"])
    sub=summary.loc[summary.period.isin(["2022_2023","2024_2026","all"])][["case","period","mean_cagr","phase_std_cagr","worst_drawdown","mean_total_return"]]
    body+="<h2>分时期及相同持仓固定参数对照</h2>"+table(sub,pct=["mean_cagr","phase_std_cagr","worst_drawdown","mean_total_return"])
    body+="<p>fixed13/fixed29是与第13/29组持仓规则完全相同的固定参数版本。年化按252交易日；2026为截至2026-10-13。起点标准差不是日波动率，最差回撤为5起点各自日收盘净值最大回撤的最差值。分段是截取既有账户，不重新以现金开仓。剔除年份的结果只作不连续依赖诊断。</p>"
    for case in ["rank13","rank29"]:
        body+=f"<h2>{case}近期主要持仓及利润贡献</h2><p>持仓是2024—2026所有交易日、所有起点的平均账户净值权重，未持有计零。不是今日持仓。利润贡献扣除分配至该股票的实际交易费用，再除各起点期初净值。名称仅辅助识别。</p>"
        d=next(d for d in details if d["case"]==case and d["period"]=="2024_2026")
        body+=table(pd.DataFrame(d["top_holdings"])[["instrument","name","mean_nav_weight","held_day_fraction","net_return_contribution"]],pct=["mean_nav_weight","held_day_fraction","net_return_contribution"])
        body+="<h3>近期盈利贡献最高的股票</h3>"+table(pd.DataFrame(d["top_profit_contributors"])[["instrument","name","net_return_contribution"]],pct=["net_return_contribution"])
    body+="<h2>银行及风格暴露（历史描述）</h2>"+table(years[["case","period","bank_weight",*STYLE]],pct=["bank_weight",*STYLE])
    body+="<p>B/P、E/P、波动率及动量是当日股票池横截面分位的持仓市值加权平均。只使用此前核验的银行标记，不等于完整历史行业分类。高B/P、E/P表示低估值倾向。银行盈利贡献不等于经完整因子模型剥离后的Alpha。</p>"
    signal=pd.DataFrame(raw["signals"])[["policy","year","rank_ic","bank_only_rank_ic","nonbank_only_rank_ic","bank_group_demeaned_rank_ic"]]
    body+="<h2>共同预测模型的年度Rank IC</h2>"+table(signal,pct=["rank_ic","bank_only_rank_ic","nonbank_only_rank_ic","bank_group_demeaned_rank_ic"])
    body+='<p>2022—2026已反复参与策略选择。近期高收益、控制银行后相关性为正，都不能证明不存在过拟合。两组起点敏感度较高，需要区分参数敏感、风格依赖与历史选择偏差；无正式过拟合概率，不自动选冠军或切换部署。<a href="diagnosis.json">完整数值</a>；<a href="extracted.json">账户与哈希依据</a>。</p></html>'
    (OUT / "diagnosis.html").write_text(body,encoding="utf-8")
    save(OUT / "report_manifest.json",dict(status="complete",builder_sha256=digest(__file__),
        diagnosis_sha256=digest(OUT / "diagnosis.json"),html_sha256=digest(OUT / "diagnosis.html"),
        audited_accounts=20,new_fits=0,new_replays=0))
    print(json.dumps(dict(summaries=[r for r in summaries if r["case"].startswith("rank")],
        recent_holdings=[d for d in details if d["period"]=="2024_2026"],paired_controls=paired),ensure_ascii=True))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["extract","report"],required=True)
    {"extract":extract,"report":report}[parser.parse_args().stage]()
