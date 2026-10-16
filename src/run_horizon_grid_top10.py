"""Compare prediction horizons, rebalance schedules, and portfolio sizes.

This is a separate research experiment. It never reads or changes the deployed
Futu paper configuration, model, signal, or scheduler.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from src.analyze_portfolio_robustness import run_tiered_backtest
from src.compare_models import enrich_daily_report, summarize_report, summarize_report_by_year
from src.config import PROJECT_ROOT, load_main_strategy_config, load_model_config, load_research_config
from src.factor_library import ENHANCED_FEATURE_NAMES
from src.run_horizon_10_experiment import build_label_expression
from src.run_temporal_ablation import select_period
from src.train_walk_forward_cv import build_validation_folds, fit_selection_fold, summarize_iteration_candidates


NAME = "horizon_grid_top10_2022_2025"
REPORT_DIR = PROJECT_ROOT / "reports" / "experiments" / NAME
MODEL_DIR = PROJECT_ROOT / "models" / "experiments" / NAME
FEATURE_PATH = PROJECT_ROOT / "data/processed/main_until_2026_07/ranked_features.parquet"
HORIZONS = (10, 20, 30)
SUPPORTED_HORIZONS = (5, *HORIZONS, 40, 60)
REBALANCES = (1, 10, 30)
TEST_YEARS = range(2022, 2026)
END = "2025-12-31"


def init_data(max_horizon: int, period_end: str, test_years: tuple[int, ...]) -> dict:
    research = load_research_config()
    qlib.init(provider_uri=str(PROJECT_ROOT / research["provider_uri"]), region=REG_CN, kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    if calendar.max() < pd.Timestamp(period_end):
        raise ValueError(f"The Qlib calendar ends before {period_end}")
    last_training_end = pd.Timestamp(f"{max(test_years)-1}-12-31")
    future_dates = calendar[calendar > last_training_end]
    if len(future_dates) < max_horizon:
        raise ValueError(
            f"The Qlib provider has only {len(future_dates)} trading days after "
            f"{last_training_end.date()}; {max_horizon} are required for the final training label"
        )
    return research


def build_dataset(
    horizon: int, research: dict, period_end: str, test_years: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Return all dated features and the labeled subset used only for fitting."""
    features = pd.read_parquet(FEATURE_PATH, columns=ENHANCED_FEATURE_NAMES).sort_index()
    dates = features.index.get_level_values("datetime")
    features = features.loc[dates <= pd.Timestamp(period_end)]
    features = features.loc[~features[ENHANCED_FEATURE_NAMES].isna().all(axis=1)]
    if features.index.duplicated().any() or features.empty:
        raise ValueError("Feature panel is empty or contains duplicate stock-dates")

    raw_name = f"future_return_{horizon}d"
    target_name = f"target_rank_{horizon}d"
    # The latest test year needs predictions, not observed future returns.
    label_end = min(pd.Timestamp(period_end), pd.Timestamp(f"{max(test_years)-1}-12-31"))
    labels = D.features(
        instruments=D.instruments(research["universe"]),
        fields=[build_label_expression(horizon, price_basis="legacy_open_to_close")],
        start_time=research["train_start"],
        end_time=str(label_end.date()),
        freq="day",
    )
    labels.columns = [raw_name]
    dataset = features.join(labels, how="left")
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset[target_name] = (
        dataset.groupby(level="datetime")[raw_name].rank(method="average", pct=True) - 0.5
    ).astype("float32")
    fitting = dataset.dropna(subset=[raw_name, target_name]).copy()
    if fitting.empty or fitting.index.duplicated().any():
        raise ValueError(f"No valid {horizon}-day labeled training observations")
    return dataset, fitting, raw_name, target_name


def train_horizon(
    horizon: int, research: dict, strategy: dict, parameters: dict,
    report_dir: Path, model_dir: Path, test_years: tuple[int, ...], period_end: str,
) -> Path:
    dataset, fitting, raw_name, target_name = build_dataset(
        horizon, research, period_end, test_years,
    )
    fold_cache: dict[str, list[dict]] = {}
    all_folds: list[dict] = []
    selections: list[pd.DataFrame] = []
    training: list[dict] = []
    predictions: list[pd.DataFrame] = []
    horizon_models = model_dir / f"horizon_{horizon}d"
    horizon_models.mkdir(parents=True, exist_ok=True)
    candidates = [int(value) for value in strategy["iteration_candidates"]]

    for year in test_years:
        current: list[dict] = []
        folds = build_validation_folds(year, int(strategy["validation_folds"]), 2010)
        for fold in folds:
            validation_year = fold["validation_year"]
            if validation_year not in fold_cache:
                print(f"H{horizon} validation {validation_year}: fitting 2010-{int(validation_year)-1}", flush=True)
                fold_cache[validation_year] = fit_selection_fold(
                    fitting, fold, ENHANCED_FEATURE_NAMES, target_name, raw_name,
                    parameters, candidates, horizon, int(research["min_cross_section_size"]),
                )
            current.extend({**row, "test_year": year} for row in fold_cache[validation_year])
        all_folds.extend(current)
        summary = summarize_iteration_candidates(current)
        summary.insert(0, "test_year", year)
        selections.append(summary)
        chosen = int(summary.iloc[0]["candidate_iteration"])

        # The final model is freshly fitted; no observations from its test year enter fitting.
        train = select_period(fitting, "2010-01-01", f"{year-1}-12-31", horizon, purge_end=True)
        test_dates = dataset.index.get_level_values("datetime")
        test = dataset.loc[(test_dates >= pd.Timestamp(f"{year}-01-01")) &
                           (test_dates <= min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(period_end)))]
        if test.empty or train.index.get_level_values("datetime").max().year >= year:
            raise ValueError(f"Invalid training/test boundary for {year}")
        params = deepcopy(parameters)
        params["n_jobs"] = 4
        params["n_estimators"] = chosen
        print(f"H{horizon} test {year}: refit {len(train):,} rows, {chosen} trees; predict {len(test):,} rows", flush=True)
        model = lgb.LGBMRegressor(**params)
        model.fit(train[ENHANCED_FEATURE_NAMES], train[target_name])
        model.booster_.save_model(str(horizon_models / f"model_{year}.txt"))
        prediction = test[[raw_name]].copy()
        prediction["prediction"] = model.predict(test[ENHANCED_FEATURE_NAMES], num_threads=4)
        if not np.isfinite(prediction["prediction"]).all():
            raise ValueError(f"Non-finite predictions in {year}")
        predictions.append(prediction)
        training.append({"horizon": horizon, "test_year": year, "train_start": "2010-01-01",
                         "train_end": f"{year-1}-12-31", "last_retained_signal": str(train.index.get_level_values("datetime").max().date()),
                         "train_rows": len(train), "test_rows": len(test), "chosen_trees": chosen,
                         "validation_years": ",".join(f["validation_year"] for f in folds),
                         "validation_mean_rank_ic": float(summary.iloc[0]["mean_validation_rank_ic"])})
        del model, train, test
        gc.collect()

    combined = pd.concat(predictions).sort_index()
    if combined.index.duplicated().any() or set(combined.index.get_level_values("datetime").year) != set(test_years):
        raise ValueError("Annual predictions are incomplete or duplicated")
    path = report_dir / f"horizon_{horizon}d_predictions.parquet"
    combined.to_parquet(path)
    pd.DataFrame(all_folds).to_csv(report_dir / f"horizon_{horizon}d_validation_folds.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(report_dir / f"horizon_{horizon}d_tree_selection.csv", index=False)
    pd.DataFrame(training).to_csv(report_dir / f"horizon_{horizon}d_training.csv", index=False)
    print(f"Saved H{horizon} annual predictions: {path}", flush=True)
    return path


def backtest_grid(
    strategy: dict, research: dict, horizons: tuple[int, ...],
    rebalance_schedules: dict[int, tuple[int, ...]], report_dir: Path, period_end: str,
    top_k: int = 10, result_tag: str = "", extra_cost_bps: int = 0,
) -> None:
    if extra_cost_bps < 0:
        raise ValueError("Extra cost must be nonnegative")
    surcharge = extra_cost_bps / 10_000
    suffix = f"_{result_tag}" if result_tag else ""
    if extra_cost_bps:
        suffix += f"_extra_{extra_cost_bps}bp"
    config = {**research, **strategy, "top_k": top_k, "n_drop": 3,
              "top_tier_size": 2, "top_tier_weight": 0.4,
              "buy_cost": 0.0005 + surcharge, "sell_cost": 0.0015 + surcharge,
              "minimum_cost": 5.0, "risk_degree": 0.95, "initial_cash": 10_000_000}
    overall = []
    yearly = []
    for horizon in horizons:
        path = report_dir / f"horizon_{horizon}d_predictions.parquet"
        predictions = pd.read_parquet(path)
        codes = sorted(predictions.index.get_level_values("instrument").unique())
        for interval in rebalance_schedules[horizon]:
            name = f"H{horizon}_rebalance_{interval}d_top{top_k}_drop3{suffix}"
            print(f"Backtesting {name}", flush=True)
            raw = run_tiered_backtest(
                predictions, config, rebalance_every=interval, top_k=top_k,
                top_tier_size=2, top_tier_weight=0.4,
                period_end=period_end, quote_codes=codes,
            )
            report = enrich_daily_report(raw, name)
            summary = summarize_report(report, name)
            by_year = summarize_report_by_year(report, name)
            for frame in (summary, by_year):
                frame.insert(1, "label_horizon", horizon)
                frame.insert(2, "rebalance_every", interval)
                frame.insert(3, "top_k", top_k)
                frame["gross_compounded_return"] = float((1 + report["return"]).prod() - 1) if frame is summary else np.nan
            # Qlib records cumulative CNY order value and charged commission/tax.
            summary["total_traded_cny"] = float(report["total_turnover"].iloc[-1])
            summary["total_fees_cny"] = float(report["total_cost"].iloc[-1])
            summary["gross_net_compound_gap_pp"] = 100 * (float((1 + report["return"]).prod()) - float((1 + report["net_return"]).prod()))
            summary["signal_days"] = len(pd.DatetimeIndex(predictions.index.get_level_values("datetime").unique())[::interval])
            for year, year_frame in report.groupby(report.index.year):
                mask = by_year["year"].eq(year)
                # The source columns are cumulative since 2022; first difference
                # the full account history before taking each calendar-year sum.
                by_year.loc[mask, "total_traded_cny"] = report["total_turnover"].diff().fillna(report["total_turnover"]).loc[year_frame.index].sum()
                by_year.loc[mask, "total_fees_cny"] = report["total_cost"].diff().fillna(report["total_cost"]).loc[year_frame.index].sum()
                by_year.loc[mask, "gross_compounded_return"] = (1 + year_frame["return"]).prod() - 1
            report.to_parquet(report_dir / f"{name}_daily.parquet")
            overall.append(summary)
            yearly.append(by_year)
            pd.concat(overall, ignore_index=True).to_csv(report_dir / f"comparison{suffix}.csv", index=False, encoding="utf-8-sig")
            pd.concat(yearly, ignore_index=True).to_csv(report_dir / f"yearly{suffix}.csv", index=False, encoding="utf-8-sig")
            print(summary[["model", "annualized_return", "annualized_excess_return", "max_drawdown", "total_cost_rate_sum", "average_daily_turnover"]].to_string(index=False), flush=True)
    print(f"Comparison: {report_dir / f'comparison{suffix}.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "train", "backtest", "stress"), default="all")
    parser.add_argument("--horizons", type=int, nargs="+", choices=SUPPORTED_HORIZONS, default=HORIZONS)
    parser.add_argument("--rebalance-intervals", type=int, nargs="+", default=REBALANCES)
    parser.add_argument(
        "--schedule", action="append", default=[], metavar="HORIZON:INTERVALS",
        help="Override one horizon, e.g. 60:1,5,10,20,30,40,50,60",
    )
    parser.add_argument("--experiment-name", default=NAME)
    parser.add_argument("--test-start-year", type=int, default=TEST_YEARS.start)
    parser.add_argument("--test-end-year", type=int, default=TEST_YEARS.stop - 1)
    parser.add_argument("--period-end", default=END)
    parser.add_argument("--top-k", type=int, choices=(5, 10), default=10)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--extra-cost-bps", type=int, default=10)
    args = parser.parse_args()
    horizons = tuple(dict.fromkeys(args.horizons))
    rebalance_intervals = tuple(dict.fromkeys(args.rebalance_intervals))
    if args.test_end_year < args.test_start_year:
        parser.error("Test end year must not precede start year")
    test_years = tuple(range(args.test_start_year, args.test_end_year + 1))
    period_end = pd.Timestamp(args.period_end)
    if period_end.year != args.test_end_year or period_end < pd.Timestamp(f"{args.test_start_year}-01-01"):
        parser.error("Period end must fall in the final test year")
    period_end_text = str(period_end.date())
    if not rebalance_intervals or any(interval <= 0 for interval in rebalance_intervals):
        parser.error("All rebalance intervals must be positive")
    rebalance_schedules = {horizon: rebalance_intervals for horizon in horizons}
    for spec in args.schedule:
        horizon_text, separator, intervals_text = spec.partition(":")
        if not separator or not horizon_text.isdigit() or int(horizon_text) not in rebalance_schedules:
            parser.error(f"Invalid schedule horizon: {spec}")
        try:
            intervals = tuple(dict.fromkeys(int(value) for value in intervals_text.split(",")))
        except ValueError:
            parser.error(f"Invalid rebalance schedule: {spec}")
        if not intervals or any(interval <= 0 for interval in intervals):
            parser.error(f"All rebalance intervals must be positive: {spec}")
        rebalance_schedules[int(horizon_text)] = intervals
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.experiment_name):
        parser.error("Experiment name must contain only lowercase letters, numbers, _ or -")
    if args.result_tag and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.result_tag):
        parser.error("Result tag must contain only lowercase letters, numbers, _ or -")
    report_dir = PROJECT_ROOT / "reports" / "experiments" / args.experiment_name
    model_dir = PROJECT_ROOT / "models" / "experiments" / args.experiment_name
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    report_dir.mkdir(parents=True, exist_ok=True)
    research = init_data(max(horizons), period_end_text, test_years)
    strategy = load_main_strategy_config()
    parameters = deepcopy(load_model_config()["parameters"])
    parameters["n_jobs"] = 4
    protocol_suffix = f"_{args.result_tag}" if args.result_tag else ""
    if args.stage == "stress":
        protocol_suffix += f"_extra_{args.extra_cost_bps}bp"
    protocol_path = report_dir / f"protocol{protocol_suffix}.json"
    protocol_path.write_text(json.dumps({
        "exploratory_comparison": True, "test_years": list(test_years),
        "period_end": period_end_text,
        "label_horizons": list(horizons),
        "rebalance_intervals_by_horizon": {str(h): list(rebalance_schedules[h]) for h in horizons},
        "model": "LightGBM ranked 15 enhanced features; annual expanding refit; 3 preceding validation years",
        "label": "T+1 open to T+H close; target is same-date cross-sectional return percentile minus 0.5",
        "portfolio": f"Top{args.top_k}; first 2 hold 40% of invested capital; others 60%; max 3 stock replacements per rebalance; 95% exposure",
        "top_k": args.top_k,
        "execution": "T close signal; next trading day open execution; Qlib account; position differences only",
        "fees": {"buy": 0.0005, "sell": 0.0015, "minimum_cny": 5},
        "additional_cost_per_buy_and_sell_bps": args.extra_cost_bps if args.stage == "stress" else 0,
        "unmodeled": ["spread", "slippage", "market impact", "live broker constraints"],
        "deployed_paper_strategy_modified": False,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unspecified"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.stage in ("all", "train"):
        for horizon in horizons:
            train_horizon(
                horizon, research, strategy, parameters, report_dir, model_dir,
                test_years, period_end_text,
            )
    if args.stage in ("all", "backtest"):
        backtest_grid(
            strategy, research, horizons, rebalance_schedules, report_dir, period_end_text,
            top_k=args.top_k, result_tag=args.result_tag,
        )
    if args.stage == "stress":
        if args.extra_cost_bps <= 0:
            raise ValueError("Stress run requires positive extra cost bps")
        backtest_grid(
            strategy, research, horizons, rebalance_schedules, report_dir, period_end_text,
            top_k=args.top_k, result_tag=args.result_tag,
            extra_cost_bps=args.extra_cost_bps,
        )


if __name__ == "__main__":
    main()
