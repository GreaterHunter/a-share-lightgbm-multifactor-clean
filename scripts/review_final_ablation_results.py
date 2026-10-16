"""Independent, read-only numerical review; never train, replay, or deploy.

Parquet checks use the quant environment because the bundled report runtime
does not contain a Parquet engine. Output files are separate review artifacts.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/final_ablation_asof_20261013"
DATA = ROOT / "data/processed/final_ablation_asof_20261013"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT / name, float_precision="round_trip")


def audit_account(case_id: str) -> dict:
    path = REPORT / "accounts" / case_id
    daily = pd.read_parquet(path / "daily.parquet")
    orders = pd.read_parquet(path / "orders.parquet")
    holdings = pd.read_parquet(path / "holdings.parquet")
    contributions = pd.read_parquet(path / "contributions.parquet")
    assert daily.index.is_unique and daily.index.is_monotonic_increasing
    assert len(daily) == 1154 and str(daily.index[0].date()) == "2022-01-04"
    assert str(daily.index[-1].date()) == "2026-10-13"
    assert np.isfinite(daily.select_dtypes("number")).all().all()
    assert not holdings.duplicated(["datetime", "instrument"]).any()
    assert daily.cash.min() >= -1e-5
    trades = orders.loc[orders.notional.gt(0)].copy()
    expected_fee = np.maximum(
        5.0, trades.notional * np.where(trades.direction.eq(1), .0005, .0015)
    ) + trades.notional * trades.friction
    fee_error = float(np.max(np.abs(expected_fee - trades.fee)))
    assert fee_error < 1e-7
    costs = trades.groupby("datetime").fee.sum().reindex(daily.index, fill_value=0)
    flows = trades.assign(
        flow=np.where(trades.direction.eq(1), trades.notional, -trades.notional)
    ).groupby("datetime").flow.sum().reindex(daily.index, fill_value=0)
    prior_cash = daily.cash.shift(1, fill_value=1_000_000.)
    prior_nav = daily.account.shift(1, fill_value=1_000_000.)
    market = holdings.groupby("datetime").market_value.sum().reindex(daily.index, fill_value=0)
    pnl = contributions.groupby("datetime").pnl_before_fees.sum().reindex(daily.index, fill_value=0)
    errors = {
        "fee_formula_cny": fee_error,
        "fee_ledger_cny": float(np.max(np.abs(costs - daily.fees_cash))),
        "cash_flow_cny": float(np.max(np.abs(prior_cash - flows - costs - daily.cash))),
        "holdings_nav_cny": float(np.max(np.abs(market + daily.cash - daily.account))),
        "pnl_nav_cny": float(np.max(np.abs(prior_nav + pnl - costs - daily.account))),
        "return_nav_cny": float(np.max(np.abs((1 + daily.net_return).cumprod() * 1_000_000 - daily.account))),
    }
    assert max(errors.values()) < 1e-4, (case_id, errors)
    # Repeat per-name T+1 and round-lot checks against the actual trade ledger.
    balance = {}
    bought_today = {}
    for order in trades.itertuples(index=False):
        stock = order.instrument
        day = pd.Timestamp(order.datetime)
        held = balance.get(stock, 0.)
        if order.direction == 1:
            assert abs(order.raw_quantity / 100 - round(order.raw_quantity / 100)) < 1e-6
            balance[stock] = held + order.adjusted_quantity
            bought_today[(day, stock)] = bought_today.get((day, stock), 0.) + order.adjusted_quantity
        else:
            assert order.adjusted_quantity <= held - bought_today.get((day, stock), 0.) + 1e-5
            balance[stock] = held - order.adjusted_quantity
    nav = np.r_[1., daily.account.to_numpy() / 1_000_000.]
    total = float(nav[-1] - 1)
    cagr = float(nav[-1] ** (252 / len(daily)) - 1)
    drawdown = float(np.min(nav / np.maximum.accumulate(nav) - 1))
    metrics = pd.read_csv(path / "metrics.csv", float_precision="round_trip")
    all_row = metrics.loc[metrics.period.eq("all")].iloc[0]
    assert abs(total - all_row.total_return) < 1e-10
    assert abs(cagr - all_row.annualized_return) < 1e-10
    assert abs(drawdown - all_row.max_drawdown) < 1e-10
    by_year = daily.groupby(daily.index.year).net_return.apply(lambda x: (1 + x).prod() - 1)
    assert abs(float((1 + by_year).prod() - 1) - total) < 1e-10
    return {"case_id": case_id, "errors": errors, "total_return": total,
            "annualized_return": cagr, "max_drawdown": drawdown,
            "fees_cny": float(costs.sum()), "t1_and_buy_lots_verified": True}


def boundary_audit() -> dict:
    frozen = json.loads((REPORT / "frozen_research_version.json").read_text(encoding="utf-8"))
    code = frozen["code_sha256"]
    counts = {}
    final_fits = 0
    folds = 0
    for path in (REPORT / "tasks").glob("*.json"):
        task = json.loads(path.read_text(encoding="utf-8"))
        if task.get("code_sha256") != code:
            continue
        assert task["status"] == "complete", (path.name, task["status"])
        kind = task["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "final_fit":
            record = task["result"]
            assert pd.Timestamp(record["label_end_max"]) <= pd.Timestamp(record["cutoff"])
            assert pd.Timestamp(record["train_signal_end"]) <= pd.Timestamp(record["cutoff"])
            assert pd.Timestamp(record["cutoff"]) <= pd.Timestamp("2026-10-13")
            final_fits += 1
        elif kind == "validation_fold":
            for record in task["result"]:
                assert pd.Timestamp(record["train_label_end_max"]) <= pd.Timestamp(record["train_cutoff"])
                assert pd.Timestamp(record["valid_label_end_max"]) <= pd.Timestamp(record["validation_year"], 12, 31)
                folds += 1
    calendar = pd.DatetimeIndex(pd.read_csv(DATA / "provider/calendars/day.txt", header=None)[0])
    assert calendar[-1] == pd.Timestamp("2026-10-13")
    label_counts = {}
    for horizon in [5, 10, 20, 30, 40]:
        labels = pd.read_parquet(DATA / f"labels_h{horizon}.parquet")
        assert labels.index.is_unique
        assert not np.isinf(labels[["label", "rank_target"]].to_numpy()).any()
        days = labels.index.get_level_values("datetime")
        positions = calendar.get_indexer(days)
        assert (positions >= 0).all()
        ending = positions + horizon + 1
        expected = pd.Series(pd.NaT, index=labels.index, dtype="datetime64[ns]")
        known = ending < len(calendar)
        expected.iloc[np.flatnonzero(known)] = calendar[ending[known]].to_numpy()
        actual = labels.label_end_date
        assert (actual.eq(expected) | (actual.isna() & expected.isna())).all()
        assert actual[labels.label.notna()].notna().all()
        assert actual.dropna().le(pd.Timestamp("2026-10-13")).all()
        label_counts[str(horizon)] = int(labels.label.notna().sum())
    return {"current_scientific_tasks": counts, "final_fit_boundaries_checked": final_fits,
            "validation_candidate_boundaries_checked": folds,
            "label_calendar_mapping_verified": label_counts}


def audit_latest_prediction(frozen: dict) -> dict:
    import lightgbm as lgb

    features = frozen["configuration"]["spec"]["features"]
    ranked = pd.read_parquet(DATA / "rank_features.parquet", columns=features)
    latest = ranked.xs(pd.Timestamp("2026-10-13"), level="datetime")
    assert latest.index.is_unique and not np.isinf(latest.to_numpy()).any()
    model = lgb.Booster(model_file=frozen["latest_model"]["model_path"])
    prediction = pd.Series(model.predict(latest, num_threads=4), index=latest.index)
    assert np.isfinite(prediction).all()
    saved = pd.read_csv(REPORT / "frozen_latest_ranking.csv", index_col=0,
                        float_precision="round_trip").prediction
    assert saved.index.is_unique and set(saved.index) == set(prediction.index)
    error = float((saved - prediction).abs().max())
    assert error < 1e-12
    raw = pd.read_parquet(DATA / "raw_features.parquet", columns=features)
    assert raw.index.equals(ranked.index)
    days = pd.to_datetime(["2010-01-04", "2021-12-31", "2024-12-31", "2026-10-13"])
    sample = raw.loc[raw.index.get_level_values("datetime").isin(days)]
    expected = sample.groupby(level="datetime").rank(method="average", pct=True) - .5
    rank_error = float((ranked.loc[sample.index] - expected).abs().max().max())
    assert rank_error < 1e-6
    return {"latest_model_reload_max_abs_error": error, "latest_feature_rows": len(latest),
            "latest_feature_date": "2026-10-13", "sample_rank_max_abs_error": rank_error,
            "train_information_cutoff": frozen["latest_model"]["cutoff"]}


def main() -> None:
    frozen = json.loads((REPORT / "frozen_research_version.json").read_text(encoding="utf-8"))
    winner = frozen["configuration"]["candidate_id"]
    selection = read_csv("final_selection.csv")
    robust = read_csv("robustness_all.csv")
    seeds = read_csv("randomness.csv")
    baseline = read_csv("reference.csv")
    yearly = read_csv("frozen_vs_reference_yearly.csv")
    base = robust.loc[robust.candidate_id.eq(winner) & robust.period.eq("all") & robust.friction.eq(0)].sort_values("offset")
    assert base.offset.tolist() == [0, 4, 8, 12, 16]
    stressed = robust.loc[robust.candidate_id.eq(winner) & robust.period.eq("all") & robust.friction.eq(.002)]
    assert base.max_drawdown.ge(-.3).all() and stressed.annualized_return.median() > 0
    expected_ranked = selection.loc[selection.passed].sort_values("median_cagr", ascending=False)
    assert expected_ranked.iloc[0].candidate_id == winner
    case_ids = set(base.case_id) | set(stressed.case_id)
    case_ids |= set(baseline.case_id)
    case_ids |= set(seeds.loc[seeds.candidate_id.eq(winner) & seeds.period.eq("all"), "case_id"])
    accounts = [audit_account(cid) for cid in sorted(case_ids)]
    benchmark = pd.read_parquet(REPORT / "accounts" / baseline.case_id.iloc[0] / "daily.parquet").bench
    for cid in case_ids:
        other = pd.read_parquet(REPORT / "accounts" / cid / "daily.parquet").bench
        assert other.index.equals(benchmark.index) and np.array_equal(other.to_numpy(), benchmark.to_numpy())
    phase_zero = base.loc[base.offset.eq(0)].iloc[0]
    alternate_seeds = seeds.loc[seeds.candidate_id.eq(winner) & seeds.period.eq("all")]
    seed_summary = alternate_seeds.groupby("seed").agg(
        median_cagr=("annualized_return", "median"), worst_drawdown=("max_drawdown", "min")
    ).reset_index()
    overlap = {}
    for seed in [2024, 2025]:
        values = read_csv(f"seed_runs/{winner}_seed{seed}_overlap.csv").top_k_overlap
        overlap[str(seed)] = {"mean": float(values.mean()), "minimum": float(values.min())}
    account_dir = REPORT / "accounts" / phase_zero.case_id
    holdings = pd.read_parquet(account_dir / "holdings.parquet")
    daily = pd.read_parquet(account_dir / "daily.parquet")
    weights = holdings.market_value / holdings.datetime.map(daily.account)
    profit = pd.read_parquet(account_dir / "contributions.parquet")
    orders = pd.read_parquet(account_dir / "orders.parquet")
    contribution_rows = []
    for year in range(2022, 2027):
        gross = profit.loc[profit.datetime.dt.year.eq(year)].groupby("instrument").pnl_before_fees.sum()
        costs = orders.loc[orders.datetime.dt.year.eq(year)].groupby("instrument").fee.sum()
        net = gross.sub(costs, fill_value=0).sort_values(ascending=False)
        for stock, value in pd.concat([net.head(3), net.tail(3)]).items():
            contribution_rows.append({"year": year, "instrument": stock, "net_pnl_cny": float(value)})
    coarse = read_csv("portfolio_coarse.csv")
    coarse = coarse.loc[coarse.period.eq("all")]
    matched = coarse.pivot(index=["horizon", "rebalance", "top_k", "top_tier_weight", "n_drop"],
                           columns=["spec", "policy"], values="annualized_return")
    pairs = {}
    for policy in ["fixed", "tuned"]:
        delta = matched[("lgb_15_raw_rank", policy)] - matched[("lgb_15_rank_rank", policy)]
        pairs[f"raw_minus_rank_{policy}"] = {"pairs": len(delta), "positive_pairs": int(delta.gt(0).sum()), "median_cagr_delta": float(delta.median())}
    for spec in ["lgb_15_rank_rank", "lgb_15_raw_rank"]:
        delta = matched[(spec, "tuned")] - matched[(spec, "fixed")]
        pairs[f"tuned_minus_fixed_{spec}"] = {"pairs": len(delta), "positive_pairs": int(delta.gt(0).sum()), "median_cagr_delta": float(delta.median())}
    fine = read_csv("portfolio_fine.csv")
    fine = fine.loc[fine.period.eq("all")]
    without_2024 = robust.loc[robust.candidate_id.eq(winner) & robust.offset.eq(0) & robust.friction.eq(0) & robust.period.eq("excluding_2024_diagnostic")].iloc[0]
    interpretation = [
        "冻结的是研究候选，不自动切换模拟盘。预设筛选仅seed42基础成本五个偏移回撤均不超过30%，20bp摩擦年化中位数为正；没有放宽或事后追加筛选条件。",
        "候选是Top5，而非Top10。第一名占股票目标仓位60%，即账户目标57%；期间价格漂移后的实际单股最高仓位约68.81%。集中度高，不建议仅凭此历史筛选直接部署。",
        "seed2024的最差偏移回撤30.67%，超过研究筛选线，且年化中位数降至12.67%。这说明随机性检验并未证明30%风险线普遍稳健；未改选表现最好的种子。",
        "2025冻结phase0收益-6.01%，沪深300+17.66%，几何超额-20.12%。2022与2024贡献较集中，不能将全期较高收益解释成持续有效。2022净利润约93.73万元，其中SZ300274贡献约48.19万元，说明少数股票对结果有很大影响。",
        "最终基础成本五个调仓偏移年化10.98%至26.17%，中位19.61%。展示phase0年化26.17%时必须同时展示偏移中位数，不应只引用最好起点。",
        "原始输入与调参的少数最好结果不代表普遍更好：相同组合/期限对照中，排名输入在多数配对收益更高；更密集调参没有稳定提升。",
        "半年重训仅对这个候选优于年度与季度，并不证明所有模型都应半年重训。2026当前在役研究模型训练信息截止6月30日，10月13日是行情/报告截止日。",
        "费用已按每笔实际成交金额与最低费用核算。日均换手是双边成交额除前日账户净值；费用占初始资金的累计比例不是年化收益扣减值。",
        "已删除因子消融的历史结果保留，不再参与后续候选扩展。H60与R1/R3没有进入本轮后续实验。",
        "本轮足以结束既定历史筛选并冻结研究版本；不再扩历史网格。之后用未观察的前瞻数据验证，若要部署需另行确认集中度、成交时点及市场规则近似。",
        "2022—2026反复参与研究，不是全新独立测试。回测仍为理想次日开盘成交，统一9.5%涨跌幅过滤、复权公司行动及财务修订版本简化均需披露。",
    ]
    result = {
        "as_of": "2026-10-13", "review_status": "numerical_checks_passed_with_research_limitations",
        "frozen_candidate": winner, "configuration": frozen["configuration"],
        "selection": selection.to_dict("records"), "phase_zero": phase_zero.to_dict(),
        "reference": baseline.loc[baseline.period.eq("all")].iloc[0].to_dict(),
        "phases": base.to_dict("records"), "stress_20bp": stressed.to_dict("records"),
        "yearly": yearly.to_dict("records"), "random_seeds": seed_summary.to_dict("records"),
        "seed_top_k_overlap": overlap, "maximum_actual_stock_weight": float(weights.max()),
        "net_pnl_contributions": contribution_rows, "paired_model_comparisons": pairs,
        "largest_phase_zero_research_result": fine.sort_values("annualized_return", ascending=False).iloc[0].to_dict(),
        "without_2024_diagnostic": without_2024.to_dict(),
        "audit": {"accounts": accounts, "same_benchmark_and_interval": True,
                  **boundary_audit(), **audit_latest_prediction(frozen)},
        "interpretation": interpretation,
        "scientific_code_sha256": frozen["code_sha256"], "snapshot_id": frozen["snapshot_id"],
        "deployment_changed": False, "git_committed": False,
    }
    # Convert missing CSV-only fields to null, not invalid JSON NaN.
    result = json.loads(json.dumps(result, ensure_ascii=False, default=str), parse_constant=lambda _: None)
    (REPORT / "final_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    def table(frame: pd.DataFrame, percent_columns: list[str]) -> str:
        return frame.to_html(index=False, escape=True,
                             formatters={c: lambda v: f"{v:.2%}" for c in percent_columns},
                             float_format=lambda v: f"{v:,.2f}")
    performance = pd.DataFrame([
        {"版本": "冻结研究版：H20/R20/Top5/半年训，偏移0", "全期净年化": phase_zero.annualized_return, "全期最大回撤": -phase_zero.max_drawdown, "累计费用元": phase_zero.total_fees_cny},
        {"版本": "参照：H10/R10/Top10/年度训，偏移0", "全期净年化": result["reference"]["annualized_return"], "全期最大回撤": -result["reference"]["max_drawdown"], "累计费用元": result["reference"]["total_fees_cny"]},
    ])
    annual = yearly[["candidate_id", "period", "total_return", "benchmark_total_return", "total_excess_return", "max_drawdown"]].copy()
    annual["candidate_id"] = annual.candidate_id.fillna("B0参照")
    annual["period"] = annual.period.str.replace(r"^2026.*", "2026截至10月13日", regex=True)
    annual.columns = ["版本", "年份", "区间净收益", "沪深300收益", "几何超额", "区间最大回撤"]
    phases = base[["offset", "annualized_return", "max_drawdown"]].merge(
        stressed[["offset", "annualized_return", "max_drawdown"]], on="offset", suffixes=("_base", "_stress"))
    phases.columns = ["调仓偏移交易日", "基础成本净年化", "基础成本最大回撤", "每边额外20bp净年化", "每边额外20bp最大回撤"]
    selections = selection[["candidate_id", "months", "median_cagr", "worst_drawdown", "stress_median_cagr", "passed"]].copy()
    selections.columns = ["候选", "重训间隔月", "基础偏移年化中位数", "基础偏移最差回撤", "20bp偏移年化中位数", "通过预设筛选"]
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>最终研究结论与独立复核</title><style>body{font:16px system-ui,sans-serif;color:#182331;max-width:1250px;margin:32px auto;padding:0 20px;line-height:1.6}table{border-collapse:collapse;margin:18px 0;font-size:14px}th,td{padding:8px 12px;border-bottom:1px solid #d8dee8;text-align:right}th{background:#eef2f7}td:first-child,th:first-child{text-align:left}h1{font-size:26px}h2{font-size:20px;margin-top:30px}a{color:#2455a4}</style>"
    page += "<h1>最终研究结论与独立复核</h1><p>比较区间：2022年1月4日至2026年10月13日，1154个交易日，初始资金100万元。净收益已含基础交易成本；年化使用252个交易日。此报告不改变原实验、筛选规则或富途部署。</p>"
    page += "<h2>冻结版本与参照</h2>" + table(performance, ["全期净年化", "全期最大回撤"])
    page += "<h2>结论与边界</h2><ol>" + "".join(f"<li>{html.escape(item)}</li>" for item in interpretation) + "</ol>"
    page += "<h2>冻结版本全部预设调仓偏移与摩擦</h2>" + table(phases, list(phases.columns[1:]))
    page += "<h2>逐年实际区间收益，不把2026部分年份外推为全年</h2>" + table(annual, list(annual.columns[2:]))
    page += "<h2>15个重训频率候选的筛选结果</h2>" + table(selections, ["基础偏移年化中位数", "基础偏移最差回撤", "20bp偏移年化中位数"])
    page += "<h2>额外随机种子</h2>" + table(seed_summary, ["median_cagr", "worst_drawdown"])
    page += f"<h2>独立数值验收</h2><p>复核{len(accounts)}个关键账户的交易费用、现金收支、持仓净值、逐股损益、收益复利、买入整手与T+1卖出可用量，最大货币差异低于0.0001元。所有当前科学版本任务完成；训练/验证标签边界及五种期限的真实标签结束交易日校验通过。全量数据与工件哈希另见交付清单。</p>"
    page += "<p><a href='research_report.html'>完整实验表格</a>　<a href='final_review.json'>本报告数值与复核记录</a>　<a href='frozen_research_version.json'>冻结研究配置</a></p></html>"
    (REPORT / "final_review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"candidate": winner, "reviewed_accounts": len(accounts), "boundary_checks": result["audit"]["final_fit_boundaries_checked"],
                      "maximum_account_error_cny": max(max(a["errors"].values()) for a in accounts),
                      "files": [str(REPORT / "final_review.json"), str(REPORT / "final_review.html")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
