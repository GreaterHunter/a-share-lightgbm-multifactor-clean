"""Audit/reuse the exact two cached S6 portfolios; no model or replay is run."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/experiments/all_full15_offsets_asof_20261013"
ORIGINAL = ROOT / "reports/experiments/final_ablation_asof_20261013"
STUDY = ROOT / "reports/experiments/overfit_four_asof_20261013"
OUT = ROOT / "reports/experiments/s6_head5_40_asof_20261013"
CASES = {"S6_equal": "ac746d54b08767c7bd8c", "S6_head5_40": "a0f4e511799519cd389c"}
LABELS = {"S6_equal": "S6等权", "S6_head5_40": "S6前5名占40%"}
FIELDS = ["total_return", "annualized_return", "max_drawdown", "total_fees_cny",
          "average_daily_turnover", "average_holdings", "total_excess_return"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    from paper_futu import overfit_four_core as core
    import numpy as np
    import pandas as pd
    store = core.Store()
    core.verify_protected(store.protocol)
    manifest = read(BASE / "manifest.json")
    ranked_manifest = read(BASE / "all_full15_phase_report_manifest.json")
    assert digest(BASE / "all_full15_phase_ranking.csv") == ranked_manifest["report_sha256"]
    assert digest(BASE / "manifest.json") == ranked_manifest["source_manifest_sha256"]
    assert manifest["snapshot_id"] == store.protocol["snapshot_id"]
    assert manifest["scientific_code_sha256"] == core.frozen.code_hash()
    family = store.protocol["families"]["raw10"]
    assert digest(family["prediction_path"]) == family["prediction_sha256"]
    rows, evidence = [], []
    common_index = common_bench = reference_config = None
    for setting, candidate_key in CASES.items():
        candidate = next(c for c in manifest["candidates"] if c["candidate_key"] == candidate_key)
        assert candidate["prediction_path"] == family["prediction_path"]
        assert candidate["spec"] == "lgb_15_raw_rank" and candidate["policy"] == "fixed"
        assert (candidate["horizon"], candidate["rebalance"], candidate["months"], candidate["seed"]) == (10, 5, 12, 42)
        assert candidate["offsets"] == [0, 1, 2, 3, 4]
        expected_weight = .25 if setting == "S6_equal" else .4
        assert candidate["portfolio"] == dict(top_k=20, top_tier_size=5, top_tier_weight=expected_weight, n_drop=6)
        for offset, case_id in zip(candidate["offsets"], candidate["case_ids"]):
            source = ORIGINAL if case_id in manifest["known"] else BASE
            task_path = source / "tasks" / f"{case_id}.json"
            record = read(task_path)
            assert record["status"] == "complete" and record["kind"] == "account"
            assert record["code_sha256"] == manifest["scientific_code_sha256"]
            assert record["snapshot_id"] == manifest["snapshot_id"]
            config = record["config"]
            normalized = json.loads(json.dumps(config))
            normalized["offset"] = 0
            normalized["portfolio"]["top_tier_weight"] = .25
            if reference_config is None:
                reference_config = normalized
            assert normalized == reference_config
            assert config["offset"] == offset and config["portfolio"] == candidate["portfolio"]
            assert config["prediction_sha256"] == family["prediction_sha256"]
            expected_id = core.frozen.identity(dict(version=core.frozen.VERSION, kind="account", config=config,
                code=manifest["scientific_code_sha256"], data=manifest["snapshot_id"]))[:24]
            assert expected_id == case_id
            for name, expected in record["artifacts"].items():
                assert digest(name) == expected, name
            directory = Path(record["result"]["directory"])
            core.verify_account(directory)
            daily = pd.read_parquet(directory / "daily.parquet")
            if common_index is None:
                common_index, common_bench = daily.index, daily.bench
            assert daily.index.equals(common_index)
            np.testing.assert_allclose(daily.bench, common_bench, rtol=0, atol=1e-15)
            for row in record["result"]["rows"]:
                if str(row["period"]) in ["all", "2022", "2023", "2024", "2025", "2026"]:
                    rows.append(dict(setting=setting, offset=offset, period=str(row["period"]),
                        **{field: float(row[field]) for field in FIELDS}))
            evidence.append(dict(setting=setting, offset=offset, case_id=case_id, directory=str(directory),
                task_path=str(task_path), task_sha256=digest(task_path), artifacts=record["artifacts"]))
    assert len(evidence) == 10 and len(rows) == 60
    core.verify_protected(store.protocol)
    result = dict(status="passed", reused_accounts=10, new_fits=0, new_account_replays=0,
        snapshot_id=manifest["snapshot_id"], scientific_code_sha256=manifest["scientific_code_sha256"],
        prediction_sha256=family["prediction_sha256"], input_manifest_sha256=digest(BASE / "manifest.json"),
        ranking_sha256=digest(BASE / "all_full15_phase_ranking.csv"),
        comparison_rules_unchanged_except_head_weight=True, start=str(common_index.min().date()),
        end=str(common_index.max().date()), trading_days=len(common_index), config=reference_config,
        rows=rows, accounts=evidence)
    core.save(OUT / "verified_accounts.json", result)
    print(json.dumps({key: result[key] for key in ["status", "reused_accounts", "new_fits", "new_account_replays", "start", "end"]}))


def report():
    import numpy as np
    import pandas as pd
    audit_result = read(OUT / "verified_accounts.json")
    assert audit_result["status"] == "passed"
    assert digest(BASE / "all_full15_phase_ranking.csv") == audit_result["ranking_sha256"]
    source = pd.DataFrame(audit_result["rows"])
    overall, yearly, phases = [], [], []
    ranking = pd.read_csv(BASE / "all_full15_phase_ranking.csv")
    for setting in CASES:
        frame = source.loc[source.setting.eq(setting) & source.period.eq("all")].sort_values("offset")
        expected = ranking.loc[ranking.candidate_key.eq(CASES[setting])].iloc[0]
        row = dict(setting=setting, mean_cagr=float(frame.annualized_return.mean()),
            phase_std_cagr=float(frame.annualized_return.std(ddof=0)),
            worst_drawdown=float(-frame.max_drawdown.min()), mean_fees_cny=float(frame.total_fees_cny.mean()),
            mean_daily_turnover=float(frame.average_daily_turnover.mean()), average_holdings=float(frame.average_holdings.mean()))
        for name in ["mean_cagr", "phase_std_cagr", "worst_drawdown", "mean_fees_cny"]:
            np.testing.assert_allclose(row[name], expected[name], rtol=1e-12, atol=1e-10)
        overall.append(row)
        phases.extend(frame.to_dict("records"))
        for period in ["2022", "2023", "2024", "2025", "2026"]:
            subset = source.loc[source.setting.eq(setting) & source.period.eq(period)]
            yearly.append(dict(setting=setting, year=period, mean_return=float(subset.total_return.mean()),
                mean_geometric_excess=float(subset.total_excess_return.mean()),
                worst_drawdown=float(-subset.max_drawdown.min()), mean_fees_cny=float(subset.total_fees_cny.mean())))
    output = dict(overall=overall, yearly=yearly, phases=phases, source_audit_sha256=digest(OUT / "verified_accounts.json"))
    (OUT / "comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    def html_table(frame, names, percentages=(), cash=()):
        frame = frame[list(names)].copy()
        if "setting" in frame:
            frame["setting"] = frame.setting.map(LABELS)
        for column in percentages:
            frame[column] = frame[column].map(lambda value: f"{value:.2%}")
        for column in cash:
            frame[column] = frame[column].map(lambda value: f"{value/10000:.2f}万")
        return frame.rename(columns=names).to_html(index=False, border=0, escape=True)
    body = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S6前五名40%与等权对照</title>
<style>body{font:16px/1.7 system-ui,'Microsoft YaHei';max-width:1050px;margin:36px auto;padding:0 20px;color:#183047}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:right;border-bottom:1px solid #ddd}th{background:#24475e;color:white}td:first-child,th:first-child{text-align:left}h1{font-size:26px}h2{font-size:20px;margin-top:30px}</style>
<h1>S6前五名40%与等权对照</h1><p>2022年第一个交易日至2026-10-13。沿用原版S6：原始15因子、10日开盘到开盘标签、年度扩展窗口重训、固定参数、收益排名目标、seed42；每5日调仓，Top20、最多替换6只，股票目标仓位95%。</p>
<p>只改变权重：前5只平分股票仓位40%，其余15只平分60%，即每只分别占账户目标净值7.6%及3.8%。原等权每只目标4.75%。权重会随价格漂移；保留股票目标仓位差仍会交易。</p>
<p>本组已在之前的完整15因子持仓组合研究中完成。此次核对预测、数据、代码、账户配置及所有结果哈希，复算10个账户净值、现金与费用后复用；没有新增训练或重复回测。</p><h2>全期比较</h2>
<p>平均年化为五个起点复利年化的算术平均，最差回撤为五起点最大回撤的最差值。起点标准差是年化收益的跨起点离散度，不是每日收益波动率；费用为初始100万元下的全期费用均值。成本为买入0.05%、卖出0.15%、每笔最低5元。</p>"""
    body += html_table(pd.DataFrame(overall), dict(setting="版本", mean_cagr="平均年化", phase_std_cagr="起点标准差", worst_drawdown="最差回撤", mean_fees_cny="全期平均费用", mean_daily_turnover="平均日双边换手"), percentages=["mean_cagr", "phase_std_cagr", "worst_drawdown", "mean_daily_turnover"], cash=["mean_fees_cny"])
    body += "<p>在这段已反复研究的历史中，40%加权没有优于等权：平均年化略低，最差回撤、费用和起点离散度更高。部分起点或年份可改善，但不足以支持替换。它也不是新的独立样本外证据，不切换模拟盘、不提交Git。</p><h2>逐年净收益</h2><p>2026为截至10月13日累计，不是当年年化。</p>"
    years = pd.DataFrame(yearly).pivot(index="year", columns="setting", values="mean_return").reset_index()
    body += html_table(years, dict(year="年份", S6_equal="S6等权", S6_head5_40="S6前5名占40%"), percentages=["S6_equal", "S6_head5_40"])
    body += "<h2>五个预设调仓起点的复利年化</h2>"
    phase_view = pd.DataFrame(phases).pivot(index="offset", columns="setting", values="annualized_return").reset_index()
    body += html_table(phase_view, dict(offset="起点偏移交易日", S6_equal="S6等权", S6_head5_40="S6前5名占40%"), percentages=["S6_equal", "S6_head5_40"])
    body += '<p>账户、配置、哈希及逐年回撤/费用：<a href="verified_accounts.json">复核依据</a>；<a href="comparison.json">数值对照</a>。日线理想开盘成交、统一9.5%过滤及原公司行动近似等原有局限仍然适用。</p></html>'
    (OUT / "comparison.html").write_text(body, encoding="utf-8")
    (OUT / "report_manifest.json").write_text(json.dumps(dict(status="complete", reused_only=True,
        builder_sha256=digest(__file__), source_audit_sha256=digest(OUT / "verified_accounts.json"),
        comparison_sha256=digest(OUT / "comparison.json"), html_sha256=digest(OUT / "comparison.html")), indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["audit", "report"], required=True)
    audit() if parser.parse_args().stage == "audit" else report()
