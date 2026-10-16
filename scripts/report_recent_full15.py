"""Read-only re-ranking of completed accounts on the fixed 2024--2026 slice."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/experiments"
OLD = BASE / "final_ablation_asof_20261013"
GRID = BASE / "all_full15_offsets_asof_20261013"
STUDY = BASE / "overfit_four_asof_20261013"
HEAD6 = BASE / "s6_head6_40_asof_20261013"
OUT = BASE / "recent_2024_2026_ranking_asof_20261013"
YEARS = ["2024", "2025", "2026"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def compact_account(record, task_path, offset):
    assert record["status"] == "complete"
    result = record["result"]
    directory = Path(result["directory"])
    metrics = directory / "metrics.csv"
    assert digest(metrics) == record["artifacts"][str(metrics)]
    annual = {str(r["period"]): r for r in result["rows"] if str(r["period"]) in YEARS}
    assert sorted(annual) == YEARS
    assert [int(annual[y]["trading_days"]) for y in YEARS] == [242, 243, 185]
    daily = directory / "daily.parquet"
    return dict(offset=int(offset), directory=str(directory), task_path=str(task_path),
        task_sha256=digest(task_path), daily_sha256=record["artifacts"][str(daily)],
        metrics_sha256=record["artifacts"][str(metrics)], years={y: {k: annual[y][k] for k in
            ["total_return", "trading_days", "benchmark_total_return", "total_fees_cny", "turnover_sum"]} for y in YEARS})


def collect():
    from paper_futu import overfit_four_core as core
    protocol = core.Store().protocol
    core.verify_protected(protocol)
    original = read(GRID / "all_full15_phase_report_manifest.json")
    assert digest(GRID / "manifest.json") == original["source_manifest_sha256"]
    assert digest(GRID / "all_full15_phase_ranking.csv") == original["report_sha256"]
    manifest = read(GRID / "manifest.json")
    assert core.frozen.code_hash() == manifest["scientific_code_sha256"]
    assert manifest["snapshot_id"] == protocol["snapshot_id"]
    candidates = []
    for index, candidate in enumerate(manifest["candidates"], 1):
        assert candidate["seed"] == 42 and len(candidate["case_ids"]) == 5
        item = {k: candidate[k] for k in ["candidate_key", "spec", "horizon", "policy", "months", "portfolio", "rebalance"]}
        item.update(variant="V0", origin="original_grid", accounts=[])
        for offset, case in zip(candidate["offsets"], candidate["case_ids"]):
            source = OLD if case in manifest["known"] else GRID
            path = source / "tasks" / f"{case}.json"
            record = read(path)
            assert record["code_sha256"] == manifest["scientific_code_sha256"]
            assert record["snapshot_id"] == manifest["snapshot_id"]
            config = record["config"]
            assert (config["offset"], config["rebalance"], config["portfolio"]) == (offset, candidate["rebalance"], candidate["portfolio"])
            expected = core.frozen.identity(dict(version=core.frozen.VERSION, kind="account", config=config,
                code=manifest["scientific_code_sha256"], data=manifest["snapshot_id"]))[:24]
            assert expected == case
            item["accounts"].append(compact_account(record, path, offset))
        candidates.append(item)
        if index % 300 == 0:
            print(f"COLLECT original {index}/1551", flush=True)
    assert len(candidates) == 1551
    study_report = read(STUDY / "report_manifest.json")
    jobs_path = STUDY / "account_jobs.json"
    assert digest(jobs_path) == study_report["results"][str(jobs_path)]
    new_groups = {}
    for job in read(jobs_path):
        if job["variant"] == "V0" or job["seed"] != 42 or job["friction"] != 0:
            continue
        group = (job["strategy"], job["variant"])
        key = core.frozen.identity(dict(job=job, implementation=protocol["implementation_sha256"],
                                       snapshot=protocol["snapshot_id"]))[:24]
        path = STUDY / "tasks" / f"{key}.json"
        record = read(path)
        assert record["job"] == job
        if group not in new_groups:
            family = core.FAMILIES[job["family"]]
            new_groups[group] = dict(candidate_key="study_"+"_".join(group), spec=family["spec"],
                horizon=family["horizon"], policy=family["policy"], months=12, variant=job["variant"],
                portfolio=job["portfolio"], rebalance=job["rebalance"], origin="overfit_study", accounts=[])
        new_groups[group]["accounts"].append(compact_account(record, path, job["offset"]))
    assert len(new_groups) == 32 and all(len(x["accounts"]) == 5 for x in new_groups.values())
    candidates.extend(new_groups.values())
    h = read(HEAD6 / "verified_accounts.json")
    hm = read(HEAD6 / "report_manifest.json")
    assert digest(HEAD6 / "verified_accounts.json") == hm["source_audit_sha256"]
    item = dict(candidate_key="s6_head6_40", spec="lgb_15_raw_rank", horizon=10, policy="fixed", months=12,
        variant="V0", portfolio=dict(top_k=20, top_tier_size=6, top_tier_weight=.4, n_drop=6), rebalance=5,
        origin="head6_trial", accounts=[])
    for offset in range(5):
        path = HEAD6 / "tasks" / f"offset_{offset}.json"
        record = read(path)
        assert record["job"]["portfolio"] == item["portfolio"] and record["offset"] == offset
        item["accounts"].append(compact_account(record, path, offset))
    candidates.append(item)
    assert len(candidates) == 1584
    assert all(len({a["offset"] for a in c["accounts"]}) == 5 for c in candidates)
    core.verify_protected(protocol)
    save(OUT / "collected.json", dict(status="passed", snapshot_id=protocol["snapshot_id"],
        original_candidates=1551, study_variants=32, head6_candidates=1, candidate_count=1584,
        account_count=7920, source_manifest_sha256=digest(GRID / "manifest.json"),
        source_ranking_sha256=digest(GRID / "all_full15_phase_ranking.csv"),
        study_report_sha256=digest(STUDY / "report_manifest.json"), collector_sha256=digest(__file__),
        candidates=candidates, new_fits=0, new_replays=0, protected_unchanged=True))
    print("COLLECT passed: 1584 configurations, 7920 existing accounts", flush=True)


def model_label(c):
    labels = {"lgb_15_rank_raw": "D", "lgb_15_raw_raw": "E", "ridge_rank_rank": "L",
              "lgb_15_raw_rank_window10": "B-W10", "lgb_15_raw_rank_window5": "B-W5"}
    if c["spec"] == "lgb_15_rank_rank":
        label = "A" if c["policy"] == "tuned" else "C"
    elif c["spec"] == "lgb_15_raw_rank":
        label = "B" if c["policy"] == "fixed" else "B调参"
    else:
        label = labels[c["spec"]]
    if c["variant"] != "V0":
        label += "-"+c["variant"]
    if c["months"] != 12:
        label += f"·{c['months']}月重训"
    return label


def rank():
    import numpy as np
    raw = read(OUT / "collected.json")
    assert raw["status"] == "passed"
    records = []
    for c in raw["candidates"]:
        phase, annual = [], {y: [] for y in YEARS}
        fees, turnover = [], []
        for a in sorted(c["accounts"], key=lambda a: a["offset"]):
            rows = a["years"]
            wealth = np.prod([1+rows[y]["total_return"] for y in YEARS])
            n = sum(rows[y]["trading_days"] for y in YEARS)
            assert n == 670
            phase.append(float(wealth**(252/n)-1))
            fees.append(sum(rows[y]["total_fees_cny"] for y in YEARS))
            turnover.append(sum(rows[y]["turnover_sum"] for y in YEARS)/n)
            for y in YEARS:
                annual[y].append(rows[y]["total_return"])
        p = c["portfolio"]
        equal = bool(np.isclose(p["top_tier_weight"], p["top_tier_size"]/p["top_k"]))
        max_weight = max(p["top_tier_weight"]/p["top_tier_size"],
                         (1-p["top_tier_weight"])/(p["top_k"]-p["top_tier_size"]))
        weights = f"Top{p['top_k']}等权" if equal else f"Top{p['top_k']}，前{p['top_tier_size']}合计{p['top_tier_weight']:.0%}"
        mean, std = float(np.mean(phase)), float(np.std(phase, ddof=0))
        records.append({**{k: c[k] for k in ["candidate_key", "spec", "policy", "months", "variant", "horizon", "rebalance", "origin"]},
            "model": model_label(c), "portfolio": p, "weight_label": weights,
            "single_stock_40pct_or_more": bool(max_weight >= .4-1e-12),
            "mean_cagr": mean, "phase_std_cagr": std, "mean_minus_std": mean-std,
            "phase_cagrs": phase, "mean_year_returns": {y: float(np.mean(annual[y])) for y in YEARS},
            "mean_fees_cny": float(np.mean(fees)), "mean_daily_two_way_turnover": float(np.mean(turnover)),
            "worst_drawdown": None})
    records.sort(key=lambda x: (-x["mean_minus_std"], -x["mean_cagr"], x["candidate_key"]))
    for i, row in enumerate(records, 1):
        row["rank_all"] = i
    filtered = [r for r in records if not r["single_stock_40pct_or_more"]]
    for i, row in enumerate(filtered, 1):
        row["rank_without_single_concentration"] = i
    pure = sorted(filtered, key=lambda x: (-x["mean_cagr"], x["phase_std_cagr"], x["candidate_key"]))
    anchors = [r["candidate_key"] for r in records if r["candidate_key"] in
        ["ac746d54b08767c7bd8c", "a0f4e511799519cd389c", "s6_head6_40"]]
    selected = sorted({r["candidate_key"] for group in [records[:30], filtered[:30], pure[:30]] for r in group} | set(anchors))
    save(OUT / "rankings.json", dict(status="ranked", source_sha256=digest(OUT / "collected.json"),
        sort="mean_compound_annual_return_minus_population_std_across_five_offsets", start="2024-01-02", end="2026-10-13",
        trading_days=670, candidate_count=len(records), unconcentrated_candidate_count=len(filtered),
        rankings=records, top30_all=[r["candidate_key"] for r in records[:30]],
        top30_without_single_concentration=[r["candidate_key"] for r in filtered[:30]],
        top30_pure_cagr_without_single_concentration=[r["candidate_key"] for r in pure[:30]],
        anchors=anchors, selected_for_daily_audit=selected))
    print(json.dumps(dict(candidate_count=len(records), filtered_count=len(filtered), selected_for_drawdown=len(selected),
                         top5=filtered[:5]), ensure_ascii=True), flush=True)


def drawdown():
    import numpy as np
    import pandas as pd
    raw, ranked = read(OUT / "collected.json"), read(OUT / "rankings.json")
    assert digest(OUT / "collected.json") == ranked["source_sha256"]
    cases = {c["candidate_key"]: c for c in raw["candidates"]}
    results, index, benchmark = {}, None, None
    for number, key in enumerate(ranked["selected_for_daily_audit"], 1):
        drawdowns, cagrs, fees = [], [], []
        for account in sorted(cases[key]["accounts"], key=lambda a: a["offset"]):
            path = Path(account["directory"]) / "daily.parquet"
            assert digest(path) == account["daily_sha256"], str(path)
            # Read existing account returns. No replay, schedule reset or cash restart.
            full = pd.read_parquet(path, columns=["net_return", "bench", "account", "fees_cash"])
            daily = full.loc["2024-01-01":"2026-10-13"]
            assert len(daily) == 670 and daily.index.is_unique
            if index is None:
                index, benchmark = daily.index, daily.bench
            assert index.equals(daily.index)
            np.testing.assert_allclose(daily.bench, benchmark, rtol=0, atol=1e-15)
            r = daily.net_return.to_numpy()
            assert np.isfinite(r).all() and (r > -1).all()
            nav = np.r_[1., np.cumprod(1+r)]
            drawdowns.append(float(-np.min(nav/np.maximum.accumulate(nav)-1)))
            cagrs.append(float(nav[-1]**(252/670)-1))
            fees.append(float(daily.fees_cash.sum()))
            prior = full.account.shift(1, fill_value=1e6).loc[daily.index[0]]
            np.testing.assert_allclose(daily.account.to_numpy()/prior, nav[1:], rtol=1e-10, atol=1e-8)
            for year in YEARS:
                actual = float((1+daily.loc[year].net_return).prod()-1)
                np.testing.assert_allclose(actual, account["years"][year]["total_return"], rtol=1e-10, atol=1e-10)
        expected = next(r for r in ranked["rankings"] if r["candidate_key"] == key)
        np.testing.assert_allclose(cagrs, expected["phase_cagrs"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.mean(fees), expected["mean_fees_cny"], rtol=1e-10, atol=1e-6)
        results[key] = dict(worst_drawdown=max(drawdowns), phase_drawdowns=drawdowns,
                            daily_sha256_passed=True, nav_and_year_returns_reconciled=True)
        if number % 10 == 0:
            print(f"DAILY AUDIT {number}/{len(ranked['selected_for_daily_audit'])}", flush=True)
    save(OUT / "daily_audit.json", dict(status="passed", ranking_sha256=digest(OUT / "rankings.json"), results=results,
         start=str(index[0].date()), end=str(index[-1].date()), trading_days=len(index),
         benchmark_total_return=float((1+benchmark).prod()-1)))
    print(f"DAILY AUDIT passed: {len(results)} configurations", flush=True)


def report():
    import pandas as pd
    ranking, audit = read(OUT / "rankings.json"), read(OUT / "daily_audit.json")
    assert digest(OUT / "rankings.json") == audit["ranking_sha256"] and audit["status"] == "passed"
    lookup = {r["candidate_key"]: r for r in ranking["rankings"]}
    for key, data in audit["results"].items():
        lookup[key].update(data)
    groups = {name: [lookup[key] for key in ranking[name]] for name in
        ["top30_without_single_concentration", "top30_all", "top30_pure_cagr_without_single_concentration", "anchors"]}
    output = dict(status="complete", start=ranking["start"], end=ranking["end"], trading_days=670,
        candidate_count=ranking["candidate_count"], unconcentrated_candidate_count=ranking["unconcentrated_candidate_count"],
        sort=ranking["sort"], new_fits=0, new_replays=0, **groups,
        collected_sha256=digest(OUT / "collected.json"), ranking_sha256=digest(OUT / "rankings.json"),
        daily_audit_sha256=digest(OUT / "daily_audit.json"))
    save(OUT / "comparison.json", output)
    def table(rows):
        view = []
        for i, row in enumerate(rows, 1):
            view.append({"排名": i, "模型": row["model"], "预测/调仓": f"{row['horizon']}/{row['rebalance']}",
                "持仓": row["weight_label"], "最多换": row["portfolio"]["n_drop"],
                "2024": f"{row['mean_year_returns']['2024']:.2%}", "2025": f"{row['mean_year_returns']['2025']:.2%}",
                "2026累计": f"{row['mean_year_returns']['2026']:.2%}", "平均年化": f"{row['mean_cagr']:.2%}",
                "起点标准差(pp)": f"{row['phase_std_cagr']*100:.2f}",
                "最差回撤": f"{row['worst_drawdown']:.2%}"+("†" if row["worst_drawdown"] > .3 else "")})
        return pd.DataFrame(view).to_html(index=False, border=0)
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2024至2026完整15因子策略排名</title><style>body{font:15px/1.65 system-ui,'Microsoft YaHei';margin:28px;color:#183047}h1{font-size:26px}h2{font-size:20px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#24475e;color:white}td:nth-child(2),td:nth-child(4){text-align:left}.scroll{overflow-x:auto}p{max-width:1250px}</style>
<h1>2024—2026近期表现：完整15因子策略Top30</h1>
<p>2024-01-02至2026-10-13，共670个交易日；含原交易成本。每个起点先复利年化，再跨5起点取均值和总体标准差。沿用此前排序：平均年化减起点标准差；不是纯收益排名，也不是统计显著性检验。</p>
<p>共1584个模型×持仓配置：原1551组、窗口/正则化/复杂度研究新增32组、前6只40%新增1组。只纳入seed42与基础成本；种子和摩擦压力不重复当新候选。未训练新模型或重跑账户。</p>
<p>截取的是2022年开始的既有账户，2024起点净值归一化；不重置持仓或跨年调仓节奏，不代表2024重新以现金建仓。2026列为截至10月13日累计收益，非全年收益或该年年化。最大回撤也只在这个近期区间重新计算，含期初净值1，取5起点最差值。†表示超过30%，未自动剔除。</p>
<p>模型：A=15因子排名→收益排名、原年度调参；B=15因子原值→收益排名、原固定参数；C=15因子排名→收益排名、原固定参数；D=排名输入→原始收益；E=原值输入→原始收益；L=岭回归排名模型。未注明则年度重训。V1/V2/V3=近3/5/10年训练窗口；V4=近期样本衰减加权；V5/V6/V7=增强L1/L2/联合；V8=较小树。W5/W10为原网格的滑动窗口规格，不与新V2/V3混称完全同一训练规则。</p>
<p>权重百分比占股票目标仓位，股票仓位95%、另留5%现金。“最多换”限制替换股票数量；保留持仓权重调整仍会交易。此表比较策略配置，不是持仓规则完全相同的公平模型能力对照。</p>
<h2>主表：排除目标单股占股票仓位40%及以上的Top30</h2>'''
    body += '<div class="scroll">'+table(groups["top30_without_single_concentration"])+"</div>"
    body += "<h2>全量Top30（包含单股重仓）</h2><div class=\"scroll\">"+table(groups["top30_all"])+"</div>"
    body += "<h2>按平均年化直接排序的分散版Top30</h2><div class=\"scroll\">"+table(groups["top30_pure_cagr_without_single_concentration"])+"</div>"
    body += "<h2>S6三种权重的近期表现与排名</h2><div class=\"scroll\">"+table(groups["anchors"])+"</div>"
    body += '<p>费用数值和每日双边换手见<a href="comparison.json">完整数值</a>；原始任务与账户映射见<a href="collected.json">来源索引</a>；近期净值、收益及费用的复算验收见<a href="daily_audit.json">验收</a>。近期费用为原账户实际费用，并非所有配置都在2024以相同资金重新开始，因此不直接用绝对费用金额衡量成本效率。</p><p>2024—2026已被反复研究。缩短到近期重新排名增加时间区间选择偏差，不能称为独立测试，不据此自动换部署。原有日线理想开盘成交、统一涨跌幅过滤、财务历史修订和公司行动近似的局限不变。</p></html>'
    (OUT / "comparison.html").write_text(body, encoding="utf-8")
    save(OUT / "report_manifest.json", dict(status="complete", builder_sha256=digest(__file__),
        comparison_sha256=digest(OUT / "comparison.json"), html_sha256=digest(OUT / "comparison.html"),
        daily_audited_configurations=len(audit["results"]), source_frozen_directories_unchanged=True,
        new_fits=0, new_replays=0))
    print(json.dumps(groups["top30_without_single_concentration"], ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["collect", "rank", "drawdown", "report"])
    args = parser.parse_args()
    {"collect": collect, "rank": rank, "drawdown": drawdown, "report": report}[args.stage]()
