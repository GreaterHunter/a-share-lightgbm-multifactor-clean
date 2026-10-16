"""Offline Top10 horizon grid using labels aligned with next-open execution.

This experiment deliberately leaves the Futu deployment and order state alone.
The 2026 slice ends at the last locally verified session (2026-10-13).
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import qlib
import requests
from qlib.config import REG_CN
from qlib.data import D

from paper_futu import backtest_label_grid_2022_2025 as grid
from paper_futu.retrain_cn_latest import digest, protected_state, save_json
from src.build_enhanced_dataset import (
    merge_latest_event,
    prepare_annual_eps_events,
    prepare_general_report_events,
    validate_point_in_time,
)
from src.build_ranked_dataset import cross_sectional_rank
from src.config import PROJECT_ROOT as ROOT
from src.factor_library import ENHANCED_FEATURE_NAMES, FEATURE_EXPRESSIONS
from src.compare_models import summarize_report


NAME = "correct_open_grid_top10_2022_2026_asof_20261013"
DATA = ROOT / "data/processed" / NAME
MODELS = ROOT / "models/experiments" / NAME
REPORT = ROOT / "reports/experiments" / NAME
PROVIDER = ROOT / "data/processed/morning_alpha158_20261013/provider"
HISTORICAL_FEATURES = ROOT / "data/processed/main_until_2026_07/ranked_features.parquet"
FINANCIAL_EVENTS = ROOT / "data/processed/cn_latest_mature_open10_20261012/financial_events.parquet"
FEATURE_SOURCE = DATA / "ranked_features_source.parquet"
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2026-10-13")
YEARS = range(2022, 2027)
SCHEDULES = {
    5: (5,),
    10: (1, 5, 10, 30),
    20: (1, 5, 10, 30),
    30: (1, 5, 10, 30),
    40: (5, 10, 20, 30, 40),
    60: (1, 5, 10, 20, 30, 40, 50, 60),
}
KINDS = ("open_to_open",)


def configure_grid() -> None:
    """Reuse the audited chronological training/account engine in isolation."""
    grid.NAME = NAME
    grid.DATA = DATA
    grid.MODELS = MODELS
    grid.REPORT = REPORT
    grid.PROVIDER = PROVIDER
    grid.FEATURE_SOURCE = FEATURE_SOURCE
    grid.START = START
    grid.END = END
    grid.YEARS = YEARS
    grid.SCHEDULES = SCHEDULES
    grid.KINDS = KINDS
    grid.EXCHANGE_CACHE.clear()


def build_recent_features() -> dict:
    """Extend the frozen historical 15-factor panel with corrected 2026 bars."""
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    old = pd.read_parquet(HISTORICAL_FEATURES, columns=ENHANCED_FEATURE_NAMES).sort_index()
    last_old = old.index.get_level_values("datetime").max()
    if last_old != pd.Timestamp("2026-07-31"):
        raise ValueError(f"Unexpected historical feature endpoint: {last_old}")
    expressions = list(FEATURE_EXPRESSIONS.values()) + ["$close / ($factor + 1e-12)"]
    # July overlap checks that historical and new provider conventions agree.
    start = pd.Timestamp("2026-07-01")
    recent = D.features(D.instruments("csi300"), expressions,
                        start_time=str(start.date()), end_time=str(END.date()), freq="day")
    recent.columns = list(FEATURE_EXPRESSIONS) + ["raw_close"]
    recent = recent.replace([np.inf, -np.inf], np.nan)
    financials = pd.read_parquet(FINANCIAL_EVENTS)
    financials = financials.loc[financials.available_date.le(END)].copy()
    panel = merge_latest_event(recent.reset_index(),
                               prepare_general_report_events(financials),
                               "fundamental_available_date")
    panel = merge_latest_event(panel, prepare_annual_eps_events(financials),
                               "annual_eps_available_date")
    validate_point_in_time(panel)
    price = panel.raw_close.where(panel.raw_close.gt(0))
    panel["book_to_price"] = panel.book_value_per_share / price
    panel["earnings_to_price"] = panel.annual_eps / price
    panel = panel.set_index(["instrument", "datetime"]).sort_index()
    panel = panel[ENHANCED_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    panel = panel.loc[~panel.isna().all(axis=1)]
    ranked = cross_sectional_rank(panel, ENHANCED_FEATURE_NAMES).astype("float32")
    overlap = ranked.loc[ranked.index.get_level_values("datetime") <= last_old]
    reference = old.loc[old.index.get_level_values("datetime") >= start]
    if not overlap.index.equals(reference.index):
        raise ValueError("July overlap universe/stock-date rows changed")
    if not np.allclose(overlap.to_numpy(), reference.to_numpy(), atol=2e-6,
                       rtol=0, equal_nan=True):
        delta = np.abs(overlap.to_numpy() - reference.to_numpy())
        raise ValueError(f"July historical factor ranks changed: max={np.nanmax(delta)}")
    addition = ranked.loc[ranked.index.get_level_values("datetime") > last_old]
    if addition.empty or addition.index.get_level_values("datetime").max() != END:
        raise ValueError("Recent features do not reach the verified data cutoff")
    features = pd.concat([old, addition]).sort_index()
    if features.index.has_duplicates or np.isinf(features.to_numpy()).any():
        raise ValueError("Duplicate or infinite features")
    FEATURE_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_SOURCE)
    record = {"feature_sha256": digest(FEATURE_SOURCE),
              "historical_sha256": digest(HISTORICAL_FEATURES),
              "provider_ready_sha256": digest(PROVIDER / "morning_snapshot_ready.json"),
              "financial_events_sha256": digest(FINANCIAL_EVENTS),
              "historical_end": str(last_old.date()), "as_of": str(END.date()),
              "rows": len(features), "recent_rows": len(addition),
              "feature_count": len(ENHANCED_FEATURE_NAMES),
              "overlap_identical": True,
              "financial_available_max": str(financials.available_date.max().date())}
    save_json(DATA / "feature_source_manifest.json", record)
    print(f"15-factor panel: {len(features):,} stock-dates, through {END.date()}", flush=True)
    return record


def prepare_data(horizons: list[int]) -> None:
    build_recent_features()
    grid.prepare_data(horizons)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    manifest["feature_source"] = json.loads(
        (DATA / "feature_source_manifest.json").read_text(encoding="utf-8"))
    manifest["label_definition"] = "Open(T+H+1) / Open(T+1) - 1"
    save_json(DATA / "manifest.json", manifest)


def train_one_horizon(horizon: int) -> None:
    if not (DATA / "manifest.json").exists():
        raise FileNotFoundError("Run --stage data first")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if digest(FEATURE_SOURCE) != manifest["source_features_sha256"]:
        raise ValueError("The training feature file changed after data preparation")
    grid.train_models([horizon], list(KINDS))


def terminal_benchmark_return() -> float:
    """Repair one missing index day, refusing an unverified benchmark value."""
    cache = DATA / "csi300_20261012_20261013_tencent.json"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        response = requests.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                                params={"param": "sh000300,day,2026-10-12,2026-10-13,2,qfq"},
                                timeout=20)
        response.raise_for_status()
        payload = response.json()
        save_json(cache, payload)
    if payload.get("code") != 0:
        raise ValueError("Tencent CSI300 daily response was unsuccessful")
    bars = payload["data"]["sh000300"].get("day", [])
    observed = {row[0]: float(row[2]) for row in bars}
    if set(observed) != {"2026-10-12", "2026-10-13"}:
        raise ValueError("CSI300 daily quote dates do not match the experiment")
    qlib_index = D.features(["SH000300"], ["$close / $factor"],
                            start_time="2026-10-12", end_time="2026-10-12").iloc[0, 0]
    if not np.isclose(qlib_index, observed["2026-10-12"], rtol=2e-4):
        raise ValueError("External CSI300 close does not match the existing Qlib index")
    last = observed["2026-10-13"] / observed["2026-10-12"] - 1
    save_json(DATA / "csi300_terminal_benchmark_audit.json",
              {"source_url": "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
               "response_sha256": digest(cache), "previous_close": observed["2026-10-12"],
               "last_close": observed["2026-10-13"],
               "last_daily_return": last, "qlib_previous_close": qlib_index,
               "external_close_crosscheck": "https://finance.yahoo.com/quote/000300.SS/history/"})
    return float(last)


def repair_benchmark_and_summarize(horizon: int, rebalance: int,
                                   last_index_return: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = f"h{horizon}_open_to_open_r{rebalance}_open"
    path = REPORT / f"{name}_daily.csv"
    report = pd.read_csv(path, index_col="datetime", parse_dates=True)
    if report.index.max() != END or report.index.has_duplicates:
        raise ValueError("Unexpected final account date or duplicate benchmark rows")
    existing = float(report.at[END, "bench"])
    if not (np.isclose(existing, 0, atol=1e-12) or
            np.isclose(existing, last_index_return, atol=1e-8)):
        raise ValueError("Existing terminal CSI300 return conflicts with verified quote")
    report.at[END, "bench"] = last_index_return
    report["active_return"] = report.net_return - report.bench
    report["benchmark_nav"] = (1 + report.bench).cumprod()
    report["relative_excess_nav"] = report.portfolio_nav / report.benchmark_nav
    report.to_csv(path, encoding="utf-8-sig")
    rows = []
    for year, group in [("all", report), *list(report.groupby(report.index.year))]:
        summary = summarize_report(group, name)
        summary["year"] = year
        summary["horizon"] = horizon
        summary["label_kind"] = "open_to_open"
        summary["rebalance_every"] = rebalance
        summary["deal_price"] = "open"
        summary["total_fees_cny"] = group.fees_cash.sum()
        summary["average_holdings"] = group.holdings_count.mean()
        summary["minimum_holdings"] = group.holdings_count.min()
        summary["maximum_holdings"] = group.holdings_count.max()
        summary["turnover_sum"] = group.turnover.sum()
        rows.append(summary)
    output = pd.concat(rows, ignore_index=True)
    return output.loc[output.year.eq("all")], output.loc[~output.year.eq("all")]


def backtest(horizons: list[int]) -> None:
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    last_index_return = terminal_benchmark_return()
    overall, yearly = [], []
    for horizon in horizons:
        path = REPORT / f"h{horizon}_open_to_open_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Train horizon {horizon} first: {path}")
        prediction = pd.read_parquet(path)
        for rebalance in SCHEDULES[horizon]:
            print(f"Backtest H{horizon} / R{rebalance}, Top10, next-open fills", flush=True)
            grid.backtest_one(
                prediction, horizon, "open_to_open", rebalance, deal_price="open")
            summary, annual = repair_benchmark_and_summarize(
                horizon, rebalance, last_index_return)
            overall.append(summary)
            yearly.append(annual)
            pd.concat(overall, ignore_index=True).to_csv(
                REPORT / "comparison.csv", index=False, encoding="utf-8-sig")
            pd.concat(yearly, ignore_index=True).to_csv(
                REPORT / "yearly.csv", index=False, encoding="utf-8-sig")
    summary = pd.concat(overall, ignore_index=True)
    annual = pd.concat(yearly, ignore_index=True)
    grid.audit_results(horizons, summary, annual)
    annual.pivot(index=["horizon", "rebalance_every"], columns="year",
                 values="total_return").to_csv(REPORT / "yearly_return_matrix.csv",
                                               encoding="utf-8-sig")
    save_json(REPORT / "protocol.json", {
        "as_of": str(END.date()), "start": str(START.date()),
        "years": list(YEARS), "schedules": SCHEDULES,
        "label": "Open(T+H+1) / Open(T+1) - 1",
        "feature_time": "T close", "execution_time": "T+1 open",
        "feature_count": len(ENHANCED_FEATURE_NAMES),
        "annual_expanding_training": True, "validation_folds": 3,
        "tree_candidates": grid.CANDIDATES,
        "top_k": 10, "n_drop": 3, "top_tier_size": 2,
        "top_tier_fraction_invested": 0.4, "risk_degree": 0.95,
        "initial_cash_cny": 1_000_000, "buy_cost": 0.0005,
        "sell_cost": 0.0015, "minimum_cost_cny": 5,
        "round_lot": 100, "benchmark": "SH000300",
        "index_terminal_audit_sha256": digest(DATA / "csi300_terminal_benchmark_audit.json"),
        "data_manifest_sha256": digest(DATA / "manifest.json"),
        "model_training_summary_sha256": digest(REPORT / "training_summary.csv"),
        "comparison_sha256": digest(REPORT / "comparison.csv"),
        "yearly_sha256": digest(REPORT / "yearly.csv"),
        "program_sha256": digest(ROOT / "paper_futu/backtest_correct_open_grid_2022_2026.py"),
    })
    lines = ["# 正确开盘标签：Top10 离线预测／调仓网格（2022—2026）", "",
             "标签：T 日收盘后的特征预测 Open(T+H+1)/Open(T+1)-1；次日开盘交易，真实收益由连续账户逐日复算。",
             "2026 年只统计到 2026-10-13，绝不是完整日历年；年表该列为区间净收益，不是年化。",
             "本地Qlib指数缺少10月13日行情，使用与10月12日Qlib收盘价交叉核验的腾讯指数日线补齐当日基准收益（见数据审计JSON）。",
             "12 个价量因子＋3 个时点可得基本面因子；未加入 Alpha158。",
             "2010 年起扩展训练；每个测试年使用之前三个验证年，按平均 Rank IC 选树数，随后从头拟合。",
             "100 万元初始资金、Top10 分层加权（前两名各占投入资金20%，其余各7.5%）、最多更换 3 只、目标仓位95%；买入 0.05%、卖出 0.15%、每笔最低 5 元。",
             "理想日线开盘成交；未模拟盘口价差、市场冲击、分钟级容量；历史组合已被反复研究，并非全新盲测。", "",
             "| 预测/调仓 | 2022 | 2023 | 2024 | 2025 | 2026截至10/13 | 全期年化 | 最大回撤 | 全期费用(元) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.sort_values(["horizon", "rebalance_every"]).itertuples():
        sub = annual.loc[annual.model.eq(row.model)].set_index("year")
        values = " | ".join(f"{sub.loc[y, 'total_return']:.2%}" for y in YEARS)
        lines.append(f"| {row.horizon}/{row.rebalance_every} | {values} | "
                     f"{row.annualized_return:.2%} | {row.max_drawdown:.2%} | "
                     f"{row.total_fees_cny:,.0f} |")
    (REPORT / "research_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audited {len(summary)} configurations; report: {REPORT}", flush=True)


def audit_existing(horizons: list[int]) -> None:
    """Recheck saved accounts and finalize provenance without trading replay."""
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    summary = pd.read_csv(REPORT / "comparison.csv")
    annual = pd.read_csv(REPORT / "yearly.csv")
    grid.audit_results(horizons, summary, annual)
    last_index_return = terminal_benchmark_return()
    for row in summary.itertuples():
        path = REPORT / f"{row.model}_daily.csv"
        account = pd.read_csv(path, index_col="datetime", parse_dates=True)
        if account.index.max() != END or not np.isclose(
                account.at[END, "bench"], last_index_return, atol=1e-10):
            raise ValueError(f"Terminal benchmark still missing: {row.model}")
        if not np.isclose((1 + account.net_return).prod() - 1,
                          row.total_return, atol=1e-10):
            raise ValueError(f"Daily account return does not match summary: {row.model}")
    save_json(REPORT / "protocol.json", {
        "as_of": str(END.date()), "years": list(YEARS), "schedules": SCHEDULES,
        "label": "Open(T+H+1) / Open(T+1) - 1", "feature_time": "T close",
        "execution_time": "T+1 open", "feature_count": len(ENHANCED_FEATURE_NAMES),
        "annual_expanding_training": True, "validation_folds": 3,
        "tree_candidates": grid.CANDIDATES, "top_k": 10, "n_drop": 3,
        "top_tier_size": 2, "top_tier_fraction_invested": 0.4,
        "risk_degree": 0.95, "initial_cash_cny": 1_000_000,
        "buy_cost": 0.0005, "sell_cost": 0.0015,
        "minimum_cost_cny": 5, "round_lot": 100,
        "benchmark": "SH000300",
        "terminal_index_return": last_index_return,
        "index_terminal_audit_sha256": digest(DATA / "csi300_terminal_benchmark_audit.json"),
        "data_manifest_sha256": digest(DATA / "manifest.json"),
        "model_training_summary_sha256": digest(REPORT / "training_summary.csv"),
        "comparison_sha256": digest(REPORT / "comparison.csv"),
        "yearly_sha256": digest(REPORT / "yearly.csv"),
        "program_sha256": digest(ROOT / "paper_futu/backtest_correct_open_grid_2022_2026.py"),
    })
    print(f"Validated {len(summary)} saved strategy accounts and final provenance", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["data", "train", "backtest", "audit", "all"], default="all")
    parser.add_argument("--horizons", type=int, nargs="+", choices=sorted(SCHEDULES),
                        default=sorted(SCHEDULES))
    args = parser.parse_args()
    for directory in (DATA, MODELS, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    configure_grid()
    before = protected_state()
    try:
        if args.stage in {"data", "all"}:
            prepare_data(args.horizons)
        if args.stage in {"train", "all"}:
            for horizon in args.horizons:
                train_one_horizon(horizon)
        if args.stage in {"backtest", "all"}:
            backtest(args.horizons)
        if args.stage in {"audit", "all"}:
            audit_existing(args.horizons)
    finally:
        after = protected_state()
        save_json(REPORT / "protected_state_check.json",
                  {"unchanged": before == after, "before": before, "after": after})
        if before != after:
            raise RuntimeError("Futu deployment or pending order state changed")
    print(f"Completed {args.stage}: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
