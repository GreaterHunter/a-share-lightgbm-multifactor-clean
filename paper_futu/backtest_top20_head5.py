"""Compare Top20 head-five weights using frozen annual open-to-open models."""

from __future__ import annotations

import argparse
import gc
import json
import warnings

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN

from paper_futu import backtest_correct_open_grid_2022_2026 as source
from paper_futu import backtest_label_grid_2022_2025 as engine
from paper_futu.retrain_cn_latest import digest, protected_state, save_json
from src.compare_models import summarize_report
from src.config import PROJECT_ROOT


OUTPUT = PROJECT_ROOT / "reports/experiments/top20_head5_open_grid_2022_2026_asof_20261013"
VARIANTS = {
    "top10_head2_40": {"top_k": 10, "top_tier_size": 2, "top_tier_weight": 0.40},
    "top20_equal": {"top_k": 20, "top_tier_size": 5, "top_tier_weight": 0.25},
    "top20_head5_50": {"top_k": 20, "top_tier_size": 5, "top_tier_weight": 0.50},
}


def source_name(horizon: int, rebalance: int) -> str:
    return f"h{horizon}_open_to_open_r{rebalance}_open"


def load_account(path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="datetime", parse_dates=True)
    required = ["net_return", "cost", "bench", "account", "fees_cash", "traded_notional", "holdings_count"]
    if frame.index.has_duplicates or frame.empty or frame.index.max() != source.END:
        raise ValueError(f"Unexpected account dates: {path}")
    if not np.isfinite(frame[required].to_numpy()).all():
        raise ValueError(f"Nonfinite account data: {path}")
    if not np.allclose((1 + frame.net_return).cumprod() * 1_000_000,
                       frame.account, atol=1e-5, rtol=1e-6):
        raise ValueError(f"Account value does not compound from returns: {path}")
    prior = frame.account.shift(1, fill_value=1_000_000)
    if not np.allclose(frame.fees_cash, frame.cost * prior, atol=1e-5, rtol=1e-6):
        raise ValueError(f"Cash fees do not agree with daily cost rates: {path}")
    return frame


def verify_sources(horizons: list[int]) -> dict:
    protocol = json.loads((source.REPORT / "protocol.json").read_text(encoding="utf-8"))
    if protocol["label"] != "Open(T+H+1) / Open(T+1) - 1":
        raise ValueError("Source model label is not open-to-open")
    if protocol["as_of"] != str(source.END.date()) or protocol["feature_count"] != 15:
        raise ValueError("Source data period or feature specification changed")
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): digest(path)
        for path in [source.REPORT / "protocol.json", source.DATA / "manifest.json",
                     source.REPORT / "training_summary.csv"]
    }
    for horizon in horizons:
        key = f"h{horizon}_open_to_open"
        folds_path = source.REPORT / f"{key}_selection_folds.csv"
        folds = pd.read_csv(folds_path)
        for split in ("train", "valid"):
            if not (pd.to_datetime(folds[f"{split}_label_end_max"]) <=
                    pd.to_datetime(folds[f"{split}_end"])).all():
                raise ValueError(f"Source validation fold crosses cutoff: {key}")
        source_hashes[str(folds_path.relative_to(PROJECT_ROOT))] = digest(folds_path)
        prediction = source.REPORT / f"{key}_predictions.parquet"
        source_hashes[str(prediction.relative_to(PROJECT_ROOT))] = digest(prediction)
        for year in source.YEARS:
            model = source.MODELS / key / f"model_{year}.txt"
            metadata_path = model.with_name(f"metadata_{year}.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if digest(model) != metadata["model_sha256"]:
                raise ValueError(f"Source fitted model hash changed: {model}")
            if (pd.Timestamp(metadata["train_label_end_max"]) >
                    pd.Timestamp(metadata["training_information_cutoff"]) or
                    max(metadata["validation_years"]) >= year):
                raise ValueError(f"Source annual model contains later outcomes: {model}")
            source_hashes[str(model.relative_to(PROJECT_ROOT))] = metadata["model_sha256"]
            source_hashes[str(metadata_path.relative_to(PROJECT_ROOT))] = digest(metadata_path)
    return source_hashes


def summarize_account(frame: pd.DataFrame, horizon: int, rebalance: int,
                      variant: str) -> list[dict]:
    name = f"h{horizon}_r{rebalance}_{variant}"
    rows = []
    for period, group in [("2022_2026", frame),
                          ("2022_2025", frame.loc[frame.index.year <= 2025]),
                          *[(str(year), group) for year, group in frame.groupby(frame.index.year)]]:
        row = summarize_report(group, name).iloc[0].to_dict()
        row.update({"period": period, "horizon": horizon, "rebalance_every": rebalance,
                    "variant": variant, **VARIANTS[variant], "n_drop": 3,
                    "start_date": str(group.index.min().date()),
                    "end_date": str(group.index.max().date()),
                    "total_fees_cny": float(group.fees_cash.sum()),
                    "average_holdings": float(group.holdings_count.mean()),
                    "minimum_holdings": int(group.holdings_count.min()),
                    "maximum_holdings": int(group.holdings_count.max())})
        rows.append(row)
    return rows


def write_comparisons(rows: list[dict]) -> pd.DataFrame:
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "metrics_all_periods.csv", index=False, encoding="utf-8-sig")
    records = []
    for (horizon, rebalance, period), group in results.groupby(["horizon", "rebalance_every", "period"]):
        if set(group.variant) != set(VARIANTS):
            continue
        if not np.allclose(group.benchmark_total_return, group.benchmark_total_return.iloc[0], atol=1e-12):
            raise ValueError("Portfolio variants have different benchmark returns")
        by_variant = group.set_index("variant")
        record = {"horizon": horizon, "rebalance_every": rebalance, "period": period,
                  "benchmark_total_return": float(group.benchmark_total_return.iloc[0])}
        for variant, row in by_variant.iterrows():
            for metric in ("total_return", "annualized_return", "max_drawdown",
                           "total_fees_cny", "average_daily_turnover", "average_holdings"):
                record[f"{variant}_{metric}"] = row[metric]
        record["tiered20_minus_top10_annualized_pp"] = 100 * (
            by_variant.loc["top20_head5_50", "annualized_return"] -
            by_variant.loc["top10_head2_40", "annualized_return"])
        record["tiered20_minus_equal20_annualized_pp"] = 100 * (
            by_variant.loc["top20_head5_50", "annualized_return"] -
            by_variant.loc["top20_equal", "annualized_return"])
        records.append(record)
    comparison = pd.DataFrame(records)
    comparison.to_csv(OUTPUT / "comparison.csv", index=False, encoding="utf-8-sig")
    for period in ("2022_2025", "2022_2026"):
        comparison.loc[comparison.period.eq(period)].to_csv(
            OUTPUT / f"comparison_{period}.csv", index=False, encoding="utf-8-sig")
    annual = results.loc[results.period.isin([str(y) for y in source.YEARS])]
    annual.pivot(index=["horizon", "rebalance_every", "variant"], columns="period",
                 values="total_return").to_csv(OUTPUT / "yearly_return_matrix.csv", encoding="utf-8-sig")
    write_readable_tables(results, comparison)
    return results


def write_readable_tables(results: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Keep the displayed percent units distinct from raw decimal metrics."""
    for period in ("2022_2025", "2022_2026"):
        records = []
        for row in comparison.loc[comparison.period.eq(period)].itertuples():
            records.append({
                "预测/调仓": f"{row.horizon}日/{row.rebalance_every}日",
                "Top10年化收益": f"{row.top10_head2_40_annualized_return:.2%}",
                "Top10最大回撤": f"{-row.top10_head2_40_max_drawdown:.2%}",
                "Top20前5加权年化收益": f"{row.top20_head5_50_annualized_return:.2%}",
                "Top20前5加权最大回撤": f"{-row.top20_head5_50_max_drawdown:.2%}",
                "Top20等权年化收益": f"{row.top20_equal_annualized_return:.2%}",
                "Top20等权最大回撤": f"{-row.top20_equal_max_drawdown:.2%}",
                "加权Top20相对Top10年化变化(百分点)": f"{row.tiered20_minus_top10_annualized_pp:+.2f}",
                "加权Top20全期累计费用(元)": f"{row.top20_head5_50_total_fees_cny:,.0f}",
                "加权Top20双边日均换手": f"{row.top20_head5_50_average_daily_turnover:.2%}",
                "加权Top20实际平均持仓数": f"{row.top20_head5_50_average_holdings:.2f}",
            })
        pd.DataFrame(records).to_csv(OUTPUT / f"display_comparison_{period}.csv",
                                      index=False, encoding="utf-8-sig")
    records = []
    annual = results.loc[results.period.isin([str(year) for year in source.YEARS])]
    for (horizon, rebalance, variant), group in annual.groupby(["horizon", "rebalance_every", "variant"]):
        labels = {"top10_head2_40": "Top10前2加权", "top20_equal": "Top20等权",
                  "top20_head5_50": "Top20前5加权"}
        record = {"预测/调仓": f"{horizon}日/{rebalance}日", "组合": labels[variant]}
        for year in source.YEARS:
            period = str(year)
            matches = group.loc[group.period.eq(period)]
            label = "2026截至10月13日" if year == 2026 else period
            record[label] = f"{matches.iloc[0].total_return:.2%}" if len(matches) else ""
        records.append(record)
    pd.DataFrame(records).to_csv(OUTPUT / "display_yearly_returns.csv", index=False, encoding="utf-8-sig")


def validate_rollups(results: pd.DataFrame, horizons: list[int]) -> None:
    expected = sum(len(source.SCHEDULES[h]) for h in horizons) * len(VARIANTS) * 7
    if len(results) != expected or results.duplicated(["horizon", "rebalance_every", "variant", "period"]).any():
        raise ValueError("Incomplete or duplicate comparison grid")
    for _, group in results.groupby(["horizon", "rebalance_every", "variant"]):
        annual = group.loc[group.period.isin([str(y) for y in source.YEARS])]
        full = group.loc[group.period.eq("2022_2026")].iloc[0]
        if len(annual) != 5 or not np.isclose((1 + annual.total_return).prod() - 1, full.total_return, atol=1e-10):
            raise ValueError("Annual returns do not compound to full period")
        if not np.isclose(annual.total_fees_cny.sum(), full.total_fees_cny, atol=1e-5):
            raise ValueError("Annual fees disagree with full period fees")


def run(horizons: list[int]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = protected_state()
    source_hashes = verify_sources(horizons)
    source.configure_grid()
    engine.REPORT = OUTPUT
    qlib.init(provider_uri=str(source.PROVIDER), region=REG_CN, kernels=1)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    rows = []
    reference_verified = False
    try:
        for horizon in horizons:
            prediction = pd.read_parquet(source.REPORT / f"h{horizon}_open_to_open_predictions.parquet")
            if prediction.index.has_duplicates or not np.isfinite(prediction.prediction.to_numpy()).all():
                raise ValueError("Duplicate stock-date rows or invalid model scores")
            for rebalance in source.SCHEDULES[horizon]:
                baseline_path = source.REPORT / f"{source_name(horizon, rebalance)}_daily.csv"
                baseline = load_account(baseline_path)
                source_hashes[str(baseline_path.relative_to(PROJECT_ROOT))] = digest(baseline_path)
                rows.extend(summarize_account(baseline, horizon, rebalance, "top10_head2_40"))
                if not reference_verified:
                    reference_name = f"reference_replay_h{horizon}_r{rebalance}"
                    engine.backtest_one(prediction, horizon, "open_to_open", rebalance,
                                        output_name=reference_name)
                    replay = load_account(OUTPUT / f"{reference_name}_daily.csv")
                    columns = ["account", "net_return", "fees_cash", "traded_notional", "holdings_count"]
                    if (not replay.index.equals(baseline.index) or not np.allclose(
                            replay[columns], baseline[columns], atol=1e-6, rtol=1e-10)):
                        raise ValueError("Parameterized default portfolio differs from saved Top10 account")
                    reference_verified = True
                    print("Saved Top10 account independently reproduced", flush=True)
                for variant in ("top20_equal", "top20_head5_50"):
                    name = f"h{horizon}_r{rebalance}_{variant}"
                    print(f"Backtest H{horizon}/R{rebalance}: {variant}", flush=True)
                    engine.backtest_one(prediction, horizon, "open_to_open", rebalance,
                                        output_name=name, **VARIANTS[variant])
                    path = OUTPUT / f"{name}_daily.csv"
                    report = load_account(path)
                    if not report.index.equals(baseline.index):
                        raise ValueError("Different portfolio observation dates")
                    # The source account includes the separately verified final index bar.
                    report["bench"] = baseline.bench
                    report["active_return"] = report.net_return - report.bench
                    report["benchmark_nav"] = (1 + report.bench).cumprod()
                    report["relative_excess_nav"] = report.portfolio_nav / report.benchmark_nav
                    report.to_csv(path, encoding="utf-8-sig")
                    rows.extend(summarize_account(report, horizon, rebalance, variant))
                write_comparisons(rows)
            engine.EXCHANGE_CACHE.clear()
            del prediction
            gc.collect()
        results = write_comparisons(rows)
        validate_rollups(results, horizons)
        for path, original_hash in source_hashes.items():
            if digest(PROJECT_ROOT / path) != original_hash:
                raise ValueError(f"Input changed during experiment: {path}")
        save_json(OUTPUT / "protocol.json", {
            "start": "2022-01-04", "as_of": str(source.END.date()),
            "horizons": horizons, "schedules": {h: source.SCHEDULES[h] for h in horizons},
            "variants": VARIANTS, "label": "Open(T+H+1) / Open(T+1) - 1",
            "signal_time": "T close", "execution": "sell and buy position differences at T+1 open",
            "models_retrained": False, "prediction_reuse": "Frozen annual expanding models; three prior validation years",
            "features": "12 price/volume plus 3 point-in-time fundamental factors, cross-sectional ranks",
            "head5_account_weight_each": 0.095, "remaining15_account_weight_each": 0.95 * 0.5 / 15,
            "risk_degree": 0.95, "n_drop": 3, "initial_cash_cny": 1_000_000,
            "costs": {"buy": 0.0005, "sell": 0.0015, "minimum_per_order_cny": 5},
            "round_lot": 100, "opening_limits": engine.execution_limit_expressions("open"),
            "weight_policy": "At each rebalance reweight the retained basket by latest model scores; between signals weights drift with prices",
            "replacement_policy": "At most three names replaced per rebalance; Top20 is a target size, not unconditional replacement with latest raw Top20",
            "source_hashes": source_hashes, "source_top10_replay_verified": reference_verified,
            "annual_rollups_verified": True, "fees_verified": True,
            "limitations": "2026 is partial through October13; ideal daily opening fills omit spread and impact; historical experiments repeatedly studied; fixed costs and common lot/limit assumptions inherited",
            "program_sha256": digest(PROJECT_ROOT / "paper_futu/backtest_top20_head5.py"),
            "engine_sha256": digest(PROJECT_ROOT / "paper_futu/backtest_label_grid_2022_2025.py"),
        })
        full = results.loc[(results.period == "2022_2026") & (results.variant == "top20_head5_50")]
        print(full.sort_values("annualized_return", ascending=False)[[
            "horizon", "rebalance_every", "annualized_return", "max_drawdown",
            "total_fees_cny", "average_daily_turnover", "average_holdings"]].to_string(index=False), flush=True)
    finally:
        after = protected_state()
        save_json(OUTPUT / "protected_state_check.json", {"unchanged": before == after, "before": before, "after": after})
        if before != after:
            raise RuntimeError("Active paper deployment or pending plans changed during experiment")
    print(f"Completed: {OUTPUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, choices=sorted(source.SCHEDULES),
                        default=sorted(source.SCHEDULES))
    args = parser.parse_args()
    run(sorted(set(args.horizons)))


if __name__ == "__main__":
    main()
