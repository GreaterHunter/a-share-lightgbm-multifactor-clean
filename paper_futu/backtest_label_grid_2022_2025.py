"""Rebuild annual Top10 horizon/label comparisons with chronological validation.

All results are offline. Daily features known at T close execute no earlier
than T+1. Close-price execution is a theoretical sensitivity, not a broker fill.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import importlib
import json
import os
import subprocess
import sys
from unittest.mock import patch
from functools import partial
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from paper_futu.retrain_cn_latest import digest, protected_state, save_json
from src.build_ranked_dataset import cross_sectional_rank
from src.compare_models import build_rebalance_signal, enrich_daily_report, summarize_report
from src.config import PROJECT_ROOT as ROOT, load_model_config
from src.factor_library import ENHANCED_FEATURE_NAMES
from src.train_walk_forward_cv import build_validation_folds, summarize_iteration_candidates

NAME = "label_grid_top10_2022_2025"
DATA = ROOT / "data/processed" / NAME
MODELS = ROOT / "models/experiments" / NAME
REPORT = ROOT / "reports/experiments" / NAME
PROVIDER = ROOT / "data/processed/cn_latest_mature_open10_20261012/provider"
FEATURE_SOURCE = ROOT / "data/processed/main_until_2026_07/ranked_features.parquet"
END = pd.Timestamp("2025-12-31")
START = pd.Timestamp("2022-01-01")
YEARS = range(2022, 2026)
SCHEDULES = {5: (3, 5, 10), 10: (1, 3, 5, 10, 30), 15: (5, 10, 15),
             20: (1, 5, 10, 30), 30: (1, 5, 10, 30),
             40: (5, 10, 20, 30, 40), 60: (1, 5, 10, 20, 30, 40, 50, 60)}
KINDS = ("open_to_close", "open_to_open")
CANDIDATES = [10, 20, 40, 80, 160, 320, 640]
EXCHANGE_CACHE: dict[tuple, object] = {}


def daily_ic(prediction: pd.Series, outcomes: pd.Series, minimum: int = 30) -> pd.DataFrame:
    """Vectorized daily Pearson/Spearman; ties use the same average-rank rule."""
    pair = pd.DataFrame({"prediction": prediction, "outcome": outcomes}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    count = pair.groupby(level="datetime").size()
    result = pd.DataFrame({"observations": count})
    for name, values in [("ic", pair), ("rank_ic", pair.groupby(level="datetime").rank(method="average"))]:
        centered = values - values.groupby(level="datetime").transform("mean")
        product = (centered.prediction * centered.outcome).groupby(level="datetime").sum()
        square = centered.pow(2).groupby(level="datetime").sum()
        denominator = np.sqrt(square.prediction * square.outcome)
        result[name] = (product / denominator.where(denominator > 0)).where(count >= minimum)
    result.index.name = "datetime"
    return result


def expression(horizon: int, kind: str) -> str:
    if kind == "open_to_close":
        return f"Ref($close, -{horizon}) / Ref($open, -1) - 1"
    if kind == "open_to_open":
        return f"Ref($open, -{horizon + 1}) / Ref($open, -1) - 1"
    raise ValueError("Unknown label kind")


def execution_limit_expressions(deal_price: str) -> tuple[str, str]:
    """Do not use the day's final price change to filter morning trades.

    The 9.5% threshold remains the project's conservative common assumption;
    it is not a reconstruction of board/ST-specific exchange price limits.
    """
    if deal_price not in {"open", "close"}:
        raise ValueError("Unsupported execution price")
    change = f"(${deal_price} / Ref($close, 1) - 1)"
    return f"{change} >= 0.095", f"{change} <= -0.095"


def prepare_data(horizons: list[int]) -> None:
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    features = pd.read_parquet(FEATURE_SOURCE, columns=ENHANCED_FEATURE_NAMES)
    features = features.loc[features.index.get_level_values("datetime") <= END].sort_index()
    if features.index.duplicated().any() or np.isinf(features.to_numpy()).any():
        raise ValueError("Invalid feature panel")
    features.to_parquet(DATA / "ranked_features.parquet")
    pairs = [(h, k) for h in horizons for k in KINDS]
    print(f"Querying {len(pairs)} labels on one feature/universe snapshot", flush=True)
    labels = D.features(D.instruments("csi300"), [expression(h, k) for h, k in pairs],
                        start_time="2010-01-01", end_time=str(END.date()), freq="day")
    labels.columns = [f"label_{h}_{k}" for h, k in pairs]
    labels = labels.reindex(features.index)
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    unique_dates = pd.DatetimeIndex(features.index.get_level_values("datetime").unique()).sort_values()
    positions = calendar.get_indexer(unique_dates)
    if (positions < 0).any():
        raise ValueError("Feature date missing from market calendar")
    for h, k in pairs:
        ends = positions + h + int(k == "open_to_open")
        dates = pd.Series(pd.NaT, index=unique_dates, dtype="datetime64[ns]")
        valid = ends < len(calendar)
        dates.iloc[np.flatnonzero(valid)] = calendar[ends[valid]].to_numpy()
        column = f"end_{h}_{k}"
        labels[column] = labels.index.get_level_values("datetime").map(dates)
        labels.loc[labels[column].isna() | labels[column].gt(END), f"label_{h}_{k}"] = np.nan
    labels.to_parquet(DATA / "labels.parquet")
    save_json(DATA / "manifest.json", {"features_sha256": digest(DATA / "ranked_features.parquet"),
              "labels_sha256": digest(DATA / "labels.parquet"), "source_features_sha256": digest(FEATURE_SOURCE),
              "label_horizons": horizons, "end": END, "rows": len(features), "provider": str(PROVIDER),
              "daily_inference_features_preserved": True})


def mature(frame: pd.DataFrame, start: str, cutoff: str) -> pd.DataFrame:
    dates = frame.index.get_level_values("datetime")
    result = frame.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(cutoff))
                       & frame.label_end_date.le(pd.Timestamp(cutoff))].dropna(subset=["label", "target"])
    if result.empty or not result.label_end_date.le(pd.Timestamp(cutoff)).all():
        raise ValueError("Invalid training or validation information boundary")
    return result


def signal_execution_years(calendar: pd.DatetimeIndex, start: pd.Timestamp,
                          end: pd.Timestamp) -> pd.Series:
    trade_dates = calendar[(calendar >= start) & (calendar <= end)]
    positions = calendar.get_indexer(trade_dates)
    if not len(positions) or (positions <= 0).any():
        raise ValueError("No previous signal session for the execution interval")
    signal_dates = calendar[positions - 1]
    return pd.Series(trade_dates.year, index=signal_dates, name="execution_year")


def train_models(horizons: list[int], kinds: list[str]) -> None:
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    features = pd.read_parquet(DATA / "ranked_features.parquet")
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    execution_year = signal_execution_years(calendar, START, END)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    parameters = deepcopy(load_model_config()["parameters"])
    parameters.update(n_jobs=4, random_state=42)
    summary_path = REPORT / "training_summary.csv"
    training_records = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    for horizon in horizons:
        for kind in kinds:
            key = f"h{horizon}_{kind}"
            fold_path = REPORT / f"{key}_selection_folds.csv"
            records = pd.read_csv(fold_path).to_dict("records") if fold_path.exists() else []
            if records and any(row["data_sha256"] != manifest["labels_sha256"] for row in records):
                raise ValueError("Cached validation results use a different data snapshot")
            # Project only this label and its maturity date; the full label grid
            # is large, and keeping it resident does not help this training job.
            labels = pd.read_parquet(DATA / "labels.parquet", columns=[
                f"label_{horizon}_{kind}", f"end_{horizon}_{kind}"])
            frame = features.join(labels).rename(
                columns={f"label_{horizon}_{kind}": "label", f"end_{horizon}_{kind}": "label_end_date"})
            del labels
            frame["target"] = cross_sectional_rank(frame, ["label"])["label"].astype("float32")
            yearly_predictions = []
            selections = []
            for year in YEARS:
                folds = build_validation_folds(year, 3, 2010)
                for fold in folds:
                    validation_year = int(fold["validation_year"])
                    cached = [row for row in records if int(row["validation_year"]) == validation_year]
                    if len(cached) == len(CANDIDATES):
                        continue
                    records = [row for row in records if int(row["validation_year"]) != validation_year]
                    train = mature(frame, fold["train_start"], fold["train_end"])
                    valid = mature(frame, fold["valid_start"], fold["valid_end"])
                    print(f"{key}: selection {validation_year}, {len(train):,} training rows", flush=True)
                    model = lgb.LGBMRegressor(**{**parameters, "n_estimators": max(CANDIDATES)})
                    model.fit(train[ENHANCED_FEATURE_NAMES], train.target)
                    for candidate in CANDIDATES:
                        score = pd.Series(model.predict(valid[ENHANCED_FEATURE_NAMES], num_iteration=candidate), index=valid.index)
                        ic = daily_ic(score, valid.label)
                        records.append({**fold, "validation_year": validation_year,
                            "candidate_iteration": candidate, "data_sha256": manifest["labels_sha256"],
                            "train_label_end_max": str(train.label_end_date.max().date()),
                            "valid_label_end_max": str(valid.label_end_date.max().date()),
                            "valid_mean_ic": float(ic.ic.mean()), "valid_mean_rank_ic": float(ic.rank_ic.mean()),
                            "valid_mse": float((score - valid.target).pow(2).mean())})
                    pd.DataFrame(records).to_csv(fold_path, index=False)
                    del train, valid, model
                    gc.collect()
                selection = summarize_iteration_candidates([row for row in records
                    if int(row["validation_year"]) in [int(fold["validation_year"]) for fold in folds]])
                chosen = int(selection.iloc[0].candidate_iteration)
                selection["test_year"] = year
                selections.append(selection)
                train = mature(frame, "2010-01-01", f"{year-1}-12-31")
                test_signal_dates = execution_year.index[execution_year.eq(year)]
                test = features.loc[features.index.get_level_values("datetime").isin(test_signal_dates)]
                directory = MODELS / key
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"model_{year}.txt"
                metadata_path = directory / f"metadata_{year}.json"
                reuse = False
                if path.exists() and metadata_path.exists():
                    old = json.loads(metadata_path.read_text(encoding="utf-8"))
                    reuse = (old["labels_sha256"] == manifest["labels_sha256"]
                             and old["features_sha256"] == manifest["features_sha256"]
                             and old.get("parameters") == {**parameters, "n_estimators": chosen}
                             and old["chosen_iteration"] == chosen and old["model_sha256"] == digest(path))
                print(f"{key}: test {year}, {chosen} trees, {len(train):,} mature rows" + (" (cached)" if reuse else ""), flush=True)
                if reuse:
                    booster = lgb.Booster(model_file=str(path))
                else:
                    model = lgb.LGBMRegressor(**{**parameters, "n_estimators": chosen})
                    model.fit(train[ENHANCED_FEATURE_NAMES], train.target)
                    model.booster_.save_model(str(path))
                    booster = lgb.Booster(model_file=str(path))
                    check = test[ENHANCED_FEATURE_NAMES].iloc[::100]
                    if not np.allclose(model.predict(check), booster.predict(check, num_threads=4), atol=1e-12, rtol=0):
                        raise ValueError("Saved model predictions changed")
                    del model
                prediction = pd.DataFrame({"prediction": booster.predict(test[ENHANCED_FEATURE_NAMES], num_threads=4)}, index=test.index)
                prediction["execution_year"] = year
                if not np.isfinite(prediction.prediction).all():
                    raise ValueError("Nonfinite annual predictions")
                yearly_predictions.append(prediction)
                record = {"horizon": horizon, "label_kind": kind, "test_year": year,
                          "training_information_cutoff": f"{year-1}-12-31", "train_rows": len(train),
                          "last_training_signal": str(train.index.get_level_values("datetime").max().date()),
                          "train_label_end_max": str(train.label_end_date.max().date()),
                          "chosen_iteration": chosen, "mean_selection_rank_ic": float(selection.iloc[0].mean_validation_rank_ic),
                          "validation_years": [int(f["validation_year"]) for f in folds],
                          "label_expression": expression(horizon, kind), "parameters": {**parameters, "n_estimators": chosen},
                          "model_sha256": digest(path), **manifest}
                save_json(metadata_path, record)
                training_records = [row for row in training_records if not (
                    int(row["horizon"]) == horizon and row["label_kind"] == kind
                    and int(row["test_year"]) == year)]
                training_records.append({name: value for name, value in record.items() if name != "validation_years"})
                del train, test, booster
                gc.collect()
            combined = pd.concat(yearly_predictions).sort_index()
            if combined.index.duplicated().any():
                raise ValueError("Duplicate annual boundary prediction")
            combined.to_parquet(REPORT / f"{key}_predictions.parquet")
            test_ic = daily_ic(combined.prediction, frame.label.reindex(combined.index))
            test_ic.to_csv(REPORT / f"{key}_test_daily_ic.csv", encoding="utf-8-sig")
            pd.concat(selections, ignore_index=True).to_csv(REPORT / f"{key}_tree_selection.csv", index=False)
            pd.DataFrame(training_records).sort_values(["horizon", "label_kind", "test_year"]).to_csv(summary_path, index=False)
            del frame
            gc.collect()


def train_in_separate_processes(horizons: list[int], kinds: list[str]) -> None:
    """Release Arrow/LightGBM native allocations after every horizon/label job."""
    for horizon in horizons:
        for kind in kinds:
            command = [sys.executable, "-m", "paper_futu.backtest_label_grid_2022_2025",
                       "--stage", "train", "--horizons", str(horizon),
                       "--label-kinds", kind, "--in-process"]
            print(f"Starting isolated training: H{horizon} {kind}", flush=True)
            log_path = REPORT / f"h{horizon}_{kind}_training.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise RuntimeError(f"Training H{horizon} {kind} failed; inspect {log_path}")


def backtest_one(prediction: pd.DataFrame, horizon: int, kind: str,
                 rebalance: int, deal_price: str = "open", *,
                 top_k: int = 10, n_drop: int = 3, top_tier_size: int = 2,
                 top_tier_weight: float = 0.4,
                 output_name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.exchange import Exchange
    from qlib.contrib.evaluate import backtest_daily
    from src.portfolio_strategy import TieredTopKWeightStrategy

    signal, signal_dates = build_rebalance_signal(prediction, rebalance)
    name = output_name or f"h{horizon}_{kind}_r{rebalance}_{deal_price}"
    first_signal = signal_dates[0]
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    first_trade = calendar[calendar > first_signal].min()
    strategy = TieredTopKWeightStrategy(signal=signal, top_k=top_k, n_drop=n_drop,
        top_tier_size=top_tier_size, top_tier_weight=top_tier_weight,
        risk_degree=0.95, only_tradable=True)
    codes = tuple(sorted(set(prediction.index.get_level_values("instrument"))))
    exchange_key = (first_signal, END, deal_price, codes)
    if exchange_key not in EXCHANGE_CACHE:
        EXCHANGE_CACHE[exchange_key] = Exchange(freq="day", start_time=first_signal,
            end_time=END, codes=list(codes), deal_price=deal_price,
            open_cost=0.0005, close_cost=0.0015, min_cost=5, trade_unit=100,
            limit_threshold=execution_limit_expressions(deal_price))
    backtest_module = importlib.import_module("qlib.backtest.backtest")
    # Disable console progress only; this does not change simulation behavior.
    with patch.object(backtest_module, "tqdm", partial(backtest_module.tqdm, disable=True)):
        raw, positions = backtest_daily(start_time=first_signal, end_time=END, strategy=strategy,
            executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False),
            account=1_000_000, benchmark="SH000300", exchange_kwargs={
                "exchange": EXCHANGE_CACHE[exchange_key]})
    warmup = raw.loc[raw.index < first_trade]
    if not warmup.turnover.eq(0).all():
        raise ValueError("Unexpected trading on initialization day")
    raw = raw.loc[raw.index >= first_trade].copy()
    if raw.turnover.iloc[0] <= 0:
        raise ValueError("Initial investment did not execute")
    initial = D.features(["SH000300"], ["$open", "$close"], start_time=first_trade,
                         end_time=first_trade, freq="day").iloc[0]
    raw.loc[first_trade, "bench"] = initial["$close"] / initial["$open"] - 1 if deal_price == "open" else 0.0
    report = enrich_daily_report(raw, name)
    previous_nav = report.portfolio_nav.shift(1, fill_value=1) * 1_000_000
    report["fees_cash"] = report.cost * previous_nav
    report["traded_notional"] = report.turnover * previous_nav
    report["holdings_count"] = pd.Series({date: len(position.get_stock_list())
        for date, position in positions.items()}).reindex(report.index, fill_value=0)
    if not np.allclose(report.portfolio_nav * 1_000_000, report.account, rtol=1e-6):
        raise ValueError("Account NAV and compounded net returns disagree")
    if not np.isfinite(report[["net_return", "bench", "cost"]].to_numpy()).all():
        raise ValueError("Invalid backtest values")
    report.to_csv(REPORT / f"{name}_daily.csv", encoding="utf-8-sig")
    rows = []
    for year, group in [("all", report), *list(report.groupby(report.index.year))]:
        summary = summarize_report(group, name)
        summary["year"] = year
        summary["horizon"] = horizon
        summary["label_kind"] = kind
        summary["rebalance_every"] = rebalance
        summary["deal_price"] = deal_price
        summary["total_fees_cny"] = group.fees_cash.sum()
        summary["average_holdings"] = group.holdings_count.mean()
        summary["minimum_holdings"] = group.holdings_count.min()
        summary["maximum_holdings"] = group.holdings_count.max()
        summary["turnover_sum"] = group.turnover.sum()
        rows.append(summary)
    output = pd.concat(rows, ignore_index=True)
    return output.loc[output.year.eq("all")], output.loc[~output.year.eq("all")]


def backtest_grid(horizons: list[int]) -> None:
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    # Quotes and execution constraints are shared, never accounts or holdings.
    EXCHANGE_CACHE.clear()
    overall, yearly, close_cases = [], [], []
    for horizon in horizons:
        for kind in KINDS:
            prediction = pd.read_parquet(REPORT / f"h{horizon}_{kind}_predictions.parquet")
            for rebalance in SCHEDULES[horizon]:
                print(f"Replay H{horizon} {kind}, R{rebalance}, Top10 next open", flush=True)
                summary, annual = backtest_one(prediction, horizon, kind, rebalance)
                overall.append(summary)
                yearly.append(annual)
                pd.concat(overall).to_csv(REPORT / "comparison.csv", index=False, encoding="utf-8-sig")
                pd.concat(yearly).to_csv(REPORT / "yearly.csv", index=False, encoding="utf-8-sig")
            if horizon == 10:
                summary, annual = backtest_one(prediction, horizon, kind, 10, deal_price="close")
                close_cases.append(pd.concat([summary, annual]))
    if close_cases:
        pd.concat(close_cases).to_csv(REPORT / "next_close_execution_sensitivity.csv", index=False, encoding="utf-8-sig")
    annual = pd.concat(yearly, ignore_index=True)
    pivot = annual.pivot(index=["horizon", "rebalance_every", "label_kind"], columns="year", values="total_return")
    pivot.to_csv(REPORT / "yearly_return_matrix.csv", encoding="utf-8-sig")
    summary = pd.concat(overall, ignore_index=True)
    pairs = summary.loc[summary.label_kind.eq("open_to_close")].merge(
        summary.loc[summary.label_kind.eq("open_to_open")], on=["horizon", "rebalance_every"],
        suffixes=("_oc", "_oo"), validate="one_to_one")
    pairs["oc_minus_oo_annual_return_pp"] = (pairs.annualized_return_oc - pairs.annualized_return_oo) * 100
    pairs.to_csv(REPORT / "label_pairs.csv", index=False, encoding="utf-8-sig")
    audit_results(horizons, summary, annual)
    plot_yearly_returns(horizons, annual)
    write_report(summary, annual, close_cases)


def audit_results(horizons: list[int], summary: pd.DataFrame, annual: pd.DataFrame) -> None:
    """Independently check chronological model boundaries and return rollups."""
    expected_cases = sum(len(SCHEDULES[h]) for h in horizons) * len(KINDS)
    if len(summary) != expected_cases or len(annual) != expected_cases * len(YEARS):
        raise ValueError("Incomplete strategy/year grid")
    if summary.duplicated(["horizon", "rebalance_every", "label_kind"]).any():
        raise ValueError("Duplicate strategy summary")
    if not np.allclose(summary.benchmark_total_return, summary.benchmark_total_return.iloc[0], atol=1e-12, rtol=0):
        raise ValueError("Strategies used different benchmark intervals")
    for row in summary.itertuples():
        years = annual.loc[annual.model.eq(row.model)].sort_values("year")
        if list(years.year) != list(YEARS):
            raise ValueError("Missing yearly results")
        if not np.isclose((1 + years.total_return).prod() - 1, row.total_return, atol=1e-10):
            raise ValueError("Yearly returns do not compound to the full-period return")
        if not np.isclose(years.total_fees_cny.sum(), row.total_fees_cny, atol=1e-5):
            raise ValueError("Yearly and full-period fees disagree")
    model_count = 0
    for horizon in horizons:
        for kind in KINDS:
            key = f"h{horizon}_{kind}"
            folds = pd.read_csv(REPORT / f"{key}_selection_folds.csv")
            if not (pd.to_datetime(folds.train_label_end_max) <= pd.to_datetime(folds.train_end)).all():
                raise ValueError("Selection training label crosses its information cutoff")
            if not (pd.to_datetime(folds.valid_label_end_max) <= pd.to_datetime(folds.valid_end)).all():
                raise ValueError("Selection validation label crosses its information cutoff")
            for year in YEARS:
                path = MODELS / key / f"model_{year}.txt"
                metadata = json.loads((MODELS / key / f"metadata_{year}.json").read_text(encoding="utf-8"))
                if pd.Timestamp(metadata["train_label_end_max"]) > pd.Timestamp(metadata["training_information_cutoff"]):
                    raise ValueError("Final training outcome crosses its information cutoff")
                if max(metadata["validation_years"]) >= year or digest(path) != metadata["model_sha256"]:
                    raise ValueError("Future validation year or changed saved model")
                model_count += 1
            # Compute any missing IC artifacts from the already saved annual
            # predictions; never refit a model merely to evaluate it.
            ic_path = REPORT / f"{key}_test_daily_ic.csv"
            if not ic_path.exists():
                prediction = pd.read_parquet(REPORT / f"{key}_predictions.parquet", columns=["prediction"])
                label_column = f"label_{horizon}_{kind}"
                labels = pd.read_parquet(DATA / "labels.parquet", columns=[label_column])
                daily_ic(prediction.prediction, labels[label_column].reindex(prediction.index)).to_csv(
                    ic_path, encoding="utf-8-sig")
    save_json(REPORT / "audit.json", {"strategy_cases": len(summary), "yearly_cases": len(annual),
        "annual_models": model_count, "model_boundaries_verified": True,
        "model_hashes_verified": True, "common_benchmark_verified": True,
        "annual_return_compounding_verified": True, "fee_rollups_verified": True,
        "opening_limit_filter_uses_no_same_day_close": True,
        "live_deployment_not_modified_by_pipeline": True})


def plot_yearly_returns(horizons: list[int], annual: pd.DataFrame) -> None:
    """A shared-scale heatmap makes the 64 year-by-year comparisons readable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ordered = [(h, r) for h in sorted(horizons) for r in SCHEDULES[h]]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, len(ordered) * 0.35)))
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=100)
    for ax, kind in zip(axes, KINDS):
        table = annual.loc[annual.label_kind.eq(kind)].pivot(
            index=["horizon", "rebalance_every"], columns="year", values="total_return")
        table = table.reindex(pd.MultiIndex.from_tuples(ordered)).reindex(columns=list(YEARS)) * 100
        pixels = ax.imshow(table.to_numpy(), aspect="auto", cmap="RdYlGn", norm=norm)
        ax.set_xticks(range(4), list(YEARS))
        ax.set_yticks(range(len(ordered)), [f"H{h} / R{r}" for h, r in ordered])
        ax.set_title(kind.replace("_", " "))
        for i in range(len(ordered)):
            for j in range(4):
                ax.text(j, i, f"{table.iloc[i,j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.suptitle("Top10 annual net returns | same next-open execution, 2022-2025", y=0.995)
    fig.tight_layout(rect=[0, 0.025, 0.89, 0.98])
    bar = fig.colorbar(pixels, cax=fig.add_axes([0.91, 0.14, 0.02, 0.70]), extend="both")
    bar.set_label("Net return (%)")
    fig.text(0.04, 0.008, "Annual refit; 3 prior CV years; 1m cash; costs included, spread/impact omitted. Exploratory history.", fontsize=8)
    fig.savefig(REPORT / "yearly_returns.png", dpi=160)
    plt.close(fig)


def verify_cached_replay() -> None:
    """Fresh and reused quote objects must produce identical separate accounts."""
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    path = REPORT / "h10_open_to_close_r10_open_daily.csv"
    expected = pd.read_csv(path, index_col="datetime", parse_dates=True)
    prediction = pd.read_parquet(REPORT / "h10_open_to_close_predictions.parquet")
    EXCHANGE_CACHE.clear()
    columns = ["account", "net_return", "fees_cash", "traded_notional", "holdings_count"]
    for repetition in range(2):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            backtest_one(prediction, 10, "open_to_close", 10)
        replay = pd.read_csv(path, index_col="datetime", parse_dates=True)
        if not replay.index.equals(expected.index) or not np.allclose(
            expected[columns], replay[columns], rtol=1e-10, atol=1e-9):
            raise ValueError(f"Account/quote state leaked into repetition {repetition}")
    audit = json.loads((REPORT / "audit.json").read_text(encoding="utf-8"))
    audit.update(fresh_and_reused_quote_accounts_agree=True)
    save_json(REPORT / "audit.json", audit)
    print("Independent fresh/cached account replay verified", flush=True)


def finalize_protocol() -> None:
    path = REPORT / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol.update({
        "costs": {"buy": 0.0005, "sell": 0.0015, "minimum_per_order_cny": 5,
                  "spread": "omitted", "impact": "omitted"},
        "portfolio": {"top_k": 10, "n_drop": 3, "risk_degree": 0.95,
                      "top_tier_size": 2, "top_tier_weight_within_invested": 0.4,
                      "initial_cash_cny": 1_000_000, "trade_unit_assumption": 100},
        "data_manifest": json.loads((DATA / "manifest.json").read_text(encoding="utf-8")),
        "program_sha256": digest(ROOT / "paper_futu/backtest_label_grid_2022_2025.py"),
        "runtime": {"python": sys.version, "qlib": qlib.__version__, "lightgbm": lgb.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__},
    })
    save_json(path, protocol)


def write_report(summary: pd.DataFrame, yearly: pd.DataFrame, close_cases: list) -> None:
    text = ["# 2022—2025 Top10预测期限、调仓和标签对照", "",
            "每个测试年仅使用此前已成熟标签；2010年起扩展训练、三个此前年度验证折、按平均Rank IC选树数。",
            "100万元连续账户、Top10（前2名40%、其余60%）、95%仓位、最多换3只；买0.05%、卖0.15%、最低5元。",
            "各模型使用同一股票池、特征、数据快照及候选树数。年份交界按实际执行年选择当时可用模型。",
            "2022-01-04首次执行；跨年换模型但不清空账户，最后交易日为2025-12-31，末日按收盘市值估值、不强制平仓。",
            "两种训练收益：旧标签Close(T+H)/Open(T+1)-1，新标签Open(T+H+1)/Open(T+1)-1；模型均拟合当日股票收益排名减0.5。",
            "", "以下为日历年实际净收益，不是将不足一年的收益乘252得到的数值。", "",
            "| H预测/R调仓 | 标签 | 2022 | 2023 | 2024 | 2025 | 4年年化（252日口径） | 全期最大回撤 |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.sort_values(["horizon", "rebalance_every", "label_kind"]).itertuples():
        sub = yearly.loc[(yearly.horizon == row.horizon) & (yearly.rebalance_every == row.rebalance_every)
                         & (yearly.label_kind == row.label_kind)].set_index("year")
        values = " | ".join(f"{sub.loc[y,'total_return']:.2%}" for y in YEARS)
        text.append(f"| {row.horizon}/{row.rebalance_every} | {row.label_kind} | {values} | {row.annualized_return:.2%} | {row.max_drawdown:.2%} |")
    paired = summary.loc[summary.label_kind.eq("open_to_close")].merge(
        summary.loc[summary.label_kind.eq("open_to_open")], on=["horizon", "rebalance_every"],
        suffixes=("_oc", "_oo"), validate="one_to_one")
    wins = int((paired.annualized_return_oc > paired.annualized_return_oo).sum())
    median_difference = float((paired.annualized_return_oc - paired.annualized_return_oo).median())
    text.extend(["", "## 相同预测期/调仓期的标签对照", "",
        f"旧开盘→收盘标签在{wins}/{len(paired)}组的全期年化收益较高；两者年化差的中位数为{median_difference * 100:.2f}个百分点。",
        "这些配置之间高度相关，胜场数不是独立样本的显著性检验，也不能据此保证未来表现。", "",
        "## 交易摩擦与实际持仓", "",
        "以下费用为4年逐笔费用的现金合计，不是无成本与含成本策略之间的净值损失；忽略价差和冲击。",
        "双边日均换手 = 当日买入成交額+卖出成交额 / 前日账户净值，未交易日为0。", "",
        "| H/R | 标签 | 4年累计费用（元） | 双边日均换手 | 实际持仓平均数 | 最少/最多 |",
        "|---|---|---:|---:|---:|---:|"])
    for row in summary.sort_values(["horizon", "rebalance_every", "label_kind"]).itertuples():
        text.append(f"| {row.horizon}/{row.rebalance_every} | {row.label_kind} | {row.total_fees_cny:,.0f} | "
                    f"{row.average_daily_turnover:.2%} | {row.average_holdings:.2f} | "
                    f"{row.minimum_holdings}/{row.maximum_holdings} |")
    if close_cases:
        text.extend(["", "## 主模型的理想次日收盘成交敏感性", "",
            "仅改变执行价格；仍使用前一日完整特征以及原本的开盘起点标签，不是新建了收盘→收盘预测模型。",
            "不代表14:59或15:01在富途能按该价格成交，不能据此自动变更交易程序。", "",
            "| 训练标签 | 2022 | 2023 | 2024 | 2025 | 全期年化 | 最大回撤 | 4年费用（元） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"])
        close = pd.concat(close_cases, ignore_index=True)
        for row in close.loc[close.year.eq("all")].itertuples():
            sub = close.loc[close.model.eq(row.model) & ~close.year.eq("all")].set_index("year")
            sub.index = sub.index.astype(int)
            values = " | ".join(f"{sub.loc[y,'total_return']:.2%}" for y in YEARS)
            text.append(f"| {row.label_kind} | {values} | {row.annualized_return:.2%} | "
                        f"{row.max_drawdown:.2%} | {row.total_fees_cny:,.0f} |")
    text.extend(["", "## 收盘执行的区别与接口约束", "",
        "- 这张主表全部按次日开盘成交；标签止于未来收盘，不意味着订单在收盘成交。",
        "- 另保存10日预测/10日调仓的理想次日收盘敏感性：使用昨日完整因子，当天收盘价格成交，因此不使用尚未知道的当日收盘特征。",
        "- 标签是预测目标，不是账户收益。即使保留开盘→收盘目标，账户也必须按实际成交、持续持仓和费用计算；不需要为迁就标签去改成交时间。",
        "- 14:59在收盘集合竞价时段，富途API文档说明A股模拟交易不支持竞价；不能把这一假设直接接到模拟盘。",
        "- 15:01为盘后撮合等待；交易所固定价格撮合15:05—15:30，API行情状态支持不等于已确认模拟订单支持。",
        "- 2022—2025主板股票并没有2026年扩展后的普遍盘后固定价格机制，不把当年全股票同日盘后成交算成可执行策略。",
        "- 日线撮合仍统一100股单位及涨跌停阈值，缺少分钟价格、价差与冲击模型；100万元也不能保证成交。",
        "- Top10是目标股票数；被限制卖出的旧持仓可能暂时残留，资金不足/整手/停牌也可能使实际数量不足10，另报告实际数量。",
        "- 最多换3只限制股票名称替换，不限制保留股票的权重再平衡；换手统计为买卖成交额之和/前日账户净值。",
        "- 交易费用统一固定假设，未还原历史印花税变化；没有市场冲击、买卖价差、分钟成交滑点和个股差异化涨跌停规则。",
        "- 开盘可交易性使用开盘相对前收盘的涨跌幅，不能使用当天最终收盘涨跌幅提前过滤；旧表可能使用过后者，因此本轮不与旧表混用。",
        "- 历史数据和策略已反复研究，结果是探索性回放，不能称为全新独立测试。",
        "- 模拟盘部署、活动计划和补仓绑定未由本实验修改。", "",
        "参考：[富途交易时段](https://openapi.futunn.com/futu-api-doc/qa/trade.html)、[富途行情状态](https://openapi.futunn.com/futu-api-doc/quote/quote.html)、"
        "[深交所2026交易规则](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)"])
    (REPORT / "research_note.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["data", "train", "backtest", "all"], default="all")
    parser.add_argument("--horizons", type=int, nargs="+", choices=sorted(SCHEDULES), default=sorted(SCHEDULES))
    parser.add_argument("--label-kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for path in [DATA, MODELS, REPORT]:
        path.mkdir(parents=True, exist_ok=True)
    before = protected_state()
    if not args.in_process:
        save_json(REPORT / "protocol.json", {"horizons": args.horizons, "schedules": SCHEDULES, "years": list(YEARS),
            "kinds": KINDS, "start": START, "end": END, "initial_cash": 1_000_000,
            "top_k": 10, "n_drop": 3, "rebalance_anchor": "Previous session before the first 2022 execution",
            "prediction_feature_time": "T close", "earliest_execution_time": "T+1",
            "model_update": "Annual expanding, three prior validation years, fresh final fit",
            "limit_filter": "Execution-price change versus previous close; 9.5% common conservative threshold",
            "tree_candidates": CANDIDATES, "protected_before": before})
    try:
        if args.stage in ["data", "all"]:
            prepare_data(args.horizons)
        if args.stage in ["train", "all"]:
            if args.in_process:
                train_models(args.horizons, args.label_kinds)
            else:
                train_in_separate_processes(args.horizons, args.label_kinds)
        if args.stage in ["backtest", "all"]:
            backtest_grid(args.horizons)
            if 10 in args.horizons:
                verify_cached_replay()
            finalize_protocol()
    finally:
        after = protected_state()
        save_json(REPORT / "protected_state_check.json", {"unchanged": before == after, "before": before, "after": after})
        if before != after:
            raise RuntimeError("External deployment/plan changed; inspect before continuing")
    print(f"Completed {args.stage}: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
