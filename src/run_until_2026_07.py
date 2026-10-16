"""Extend the frozen 10-day strategy through July 2026, without overwriting it.

The saved 2022-2024 annual models are reused. Fresh 2025/2026 models use the
same expanding-window, three-fold selection rule. Only supervised fitting
purges year-end labels: predictions are generated for all available feature
dates, and IC evaluation independently censors labels ending after the cutoff.
"""

from __future__ import annotations

import argparse
import gc
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
import requests
from qlib.config import REG_CN
from qlib.data import D

from src.analyze_portfolio_robustness import run_tiered_backtest
from src.build_enhanced_dataset import (
    merge_latest_event,
    prepare_annual_eps_events,
    prepare_general_report_events,
    validate_point_in_time,
)
from src.build_ranked_dataset import cross_sectional_rank
from src.compare_models import (
    enrich_daily_report,
    summarize_report,
    summarize_report_by_year,
)
from src.config import (
    PROJECT_ROOT,
    load_main_strategy_config,
    load_model_config,
    load_research_config,
)
from src.evaluate_model import calculate_daily_model_ic
from src.factor_library import (
    ENHANCED_FEATURE_NAMES,
    FEATURE_EXPRESSIONS,
    FUNDAMENTAL_FEATURE_NAMES,
)
from src.run_temporal_ablation import select_period
from src.train_walk_forward_cv import (
    build_validation_folds,
    fit_selection_fold,
    latest_training_period,
    summarize_iteration_candidates,
)


EXPERIMENT = "main_until_2026_07"
CUTOFF = pd.Timestamp("2026-07-31")
HORIZON = 10
RAW_LABEL = "future_return_10d"
RANK_LABEL = "target_rank_10d"
REPORT_DIR = PROJECT_ROOT / "reports" / "experiments" / EXPERIMENT
DATA_DIR = PROJECT_ROOT / "data" / "processed" / EXPERIMENT
MODEL_DIR = PROJECT_ROOT / "models" / "experiments" / EXPERIMENT
SOURCE_RANKED = PROJECT_ROOT / "data/processed/ranked_enhanced_multifactor_dataset.parquet"
SOURCE_MODELS = PROJECT_ROOT / "models/experiments/main_walk_forward_cv"
PREDICTION_PATH = REPORT_DIR / "predictions.parquet"
FEATURE_PATH = DATA_DIR / "ranked_features.parquet"
DATASET_PATH = DATA_DIR / "supervised_dataset.parquet"


def init_qlib() -> None:
    research = load_research_config()
    qlib.init(
        provider_uri=str(PROJECT_ROOT / research["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    if calendar.max() < CUTOFF:
        raise ValueError(f"行情日历仅到 {calendar.max().date()}，未覆盖 {CUTOFF.date()}")


def ensure_directories() -> None:
    for path in [REPORT_DIR, DATA_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_2026_disclosures() -> pd.DataFrame:
    """Use CNINFO's actual disclosure field, never its scheduled date."""
    existing = pd.read_parquet(
        PROJECT_ROOT / "data/external/fundamental_disclosure_dates.parquet"
    )
    fundamentals = pd.read_parquet(PROJECT_ROOT / "data/external/fundamentals.parquet")
    lookup = {code[-6:]: code for code in fundamentals["instrument"].unique()}
    frames = [existing]
    source_url = "https://www.cninfo.com.cn/new/information/getPrbookInfo"
    for period_end in ["2026-03-31", "2026-06-30"]:
        cache = DATA_DIR / f"cninfo_{period_end}.parquet"
        if cache.exists():
            frame = pd.read_parquet(cache)
        else:
            response = requests.post(
                source_url,
                params={
                    "sectionTime": period_end,
                    "firstTime": "",
                    "lastTime": "",
                    "market": "szsh",
                    "stockCode": "",
                    "orderClos": "",
                    "isDesc": "",
                    "pagesize": "10000",
                    "pagenum": "1",
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("hasNextPage"):
                raise ValueError("巨潮响应需要分页，不能使用不完整的披露日期")
            records = payload.get("prbookinfos", [])
            if not records:
                raise ValueError(f"巨潮没有返回 {period_end} 披露记录")
            raw = pd.DataFrame(records)
            required = {"seccode", "f006d_0102", "f001d_0102"}
            if not required.issubset(raw.columns):
                raise ValueError(f"巨潮响应字段变化：{raw.columns.tolist()}")
            # Save the unmodified response for an audit of actual vs scheduled dates.
            (DATA_DIR / f"cninfo_{period_end}_raw.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            frame = pd.DataFrame(
                {
                    "instrument": raw["seccode"].astype(str).str.zfill(6).map(lookup),
                    "report_date": pd.to_datetime(raw["f001d_0102"], errors="coerce"),
                    "disclosure_date": pd.to_datetime(raw["f006d_0102"], errors="coerce"),
                }
            ).dropna(subset=["instrument", "report_date"])
            if not frame["report_date"].eq(pd.Timestamp(period_end)).all():
                raise ValueError("巨潮响应的报告期与请求不一致")
            frame["source_url"] = source_url
            frame["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
            frame = frame.drop_duplicates(["instrument", "report_date"], keep="last")
            frame.to_parquet(cache, index=False)
        print(
            f"Actual disclosures {period_end}: {len(frame)} rows; "
            f"available by cutoff: {int((frame.disclosure_date < CUTOFF).sum())}",
            flush=True,
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["retrieved_at_utc"] = pd.to_datetime(
        combined["retrieved_at_utc"], format="mixed", errors="coerce", utc=True
    ).dt.tz_localize(None)
    combined = combined.drop_duplicates(["instrument", "report_date"], keep="last")
    combined.to_parquet(DATA_DIR / "disclosures_through_2026.parquet", index=False)
    return combined


def load_financial_events(disclosures: pd.DataFrame) -> pd.DataFrame:
    fundamentals = pd.read_parquet(PROJECT_ROOT / "data/external/fundamentals.parquet")
    financials = fundamentals.merge(
        disclosures[["instrument", "report_date", "disclosure_date"]],
        on=["instrument", "report_date"],
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["disclosure_date"])
    financials["available_date"] = financials["disclosure_date"] + pd.Timedelta(days=1)
    if (financials["available_date"] <= financials["report_date"]).any():
        raise ValueError("财务数据可用日期不晚于报告期")
    # Events announced after the experiment cutoff have no place in this panel.
    return financials.loc[financials["available_date"] <= CUTOFF].copy()


def label_end_dates(dates: pd.DatetimeIndex, horizon: int = HORIZON) -> pd.Series:
    """Return the actual exchange-calendar date on which each label matures."""
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    positions = calendar.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("样本日期不在交易日历中")
    ends = positions + horizon
    values = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    valid = ends < len(calendar)
    values.loc[dates[valid]] = calendar[ends[valid]].to_numpy()
    return values


def censor_evaluation_labels(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep every prediction but hide outcomes that mature after the cutoff."""
    result = frame.copy()
    dates = pd.DatetimeIndex(result.index.get_level_values("datetime").unique())
    ends = label_end_dates(dates)
    late = ends.index[ends.isna() | ends.gt(cutoff)]
    result.loc[result.index.get_level_values("datetime").isin(late), RAW_LABEL] = np.nan
    return result


def build_data() -> None:
    init_qlib()
    disclosures = download_2026_disclosures()
    historical = pd.read_parquet(SOURCE_RANKED, columns=ENHANCED_FEATURE_NAMES).sort_index()
    dates = historical.index.get_level_values("datetime")
    if dates.max() < pd.Timestamp("2025-12-31"):
        raise ValueError("现有排名特征未完整覆盖2025年")
    historical = historical.loc[dates <= pd.Timestamp("2025-12-31")]

    instruments = D.instruments(load_research_config()["universe"])
    expressions = list(FEATURE_EXPRESSIONS.values()) + ["$close / ($factor + 1e-12)"]
    recent = D.features(
        instruments=instruments,
        fields=expressions,
        start_time="2026-01-01",
        end_time=str(CUTOFF.date()),
        freq="day",
    )
    recent.columns = list(FEATURE_EXPRESSIONS) + ["raw_close"]
    recent = recent.replace([np.inf, -np.inf], np.nan)
    recent = recent.loc[~recent[list(FEATURE_EXPRESSIONS)].isna().all(axis=1)].copy()
    panel = recent.reset_index()
    financials = load_financial_events(disclosures)
    panel = merge_latest_event(
        panel,
        prepare_general_report_events(financials),
        "fundamental_available_date",
    )
    panel = merge_latest_event(
        panel,
        prepare_annual_eps_events(financials),
        "annual_eps_available_date",
    )
    validate_point_in_time(panel)
    price = panel["raw_close"].where(panel["raw_close"] > 0)
    panel["book_to_price"] = panel["book_value_per_share"] / price
    panel["earnings_to_price"] = panel["annual_eps"] / price
    panel[ENHANCED_FEATURE_NAMES] = panel[ENHANCED_FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan
    )
    panel = panel.set_index(["instrument", "datetime"]).sort_index()

    quality = []
    for year, group in panel.groupby(panel.index.get_level_values("datetime").year):
        for factor in FUNDAMENTAL_FEATURE_NAMES:
            quality.append(
                {
                    "year": year,
                    "factor": factor,
                    "rows": len(group),
                    "coverage_ratio": group[factor].notna().mean(),
                    "latest_used_disclosure_available_date": group[
                        "annual_eps_available_date"
                        if factor == "earnings_to_price"
                        else "fundamental_available_date"
                    ].max(),
                }
            )
    pd.DataFrame(quality).to_csv(REPORT_DIR / "fundamental_coverage.csv", index=False)
    panel[["fundamental_report_date", "fundamental_available_date", "annual_eps_report_date", "annual_eps_available_date"]].to_parquet(
        DATA_DIR / "financial_alignment_2026.parquet"
    )
    recent_ranked = cross_sectional_rank(panel, ENHANCED_FEATURE_NAMES).astype("float32")
    features = pd.concat([historical, recent_ranked]).sort_index()
    if features.index.duplicated().any():
        raise ValueError("扩展特征存在重复股票日期")
    if np.isinf(features.to_numpy()).any():
        raise ValueError("扩展特征含无穷值")
    features.to_parquet(FEATURE_PATH)

    labels = D.features(
        instruments=instruments,
        fields=[f"Ref($close, -{HORIZON}) / Ref($open, -1) - 1"],
        start_time="2010-01-01",
        end_time=str(CUTOFF.date()),
        freq="day",
    )
    labels.columns = [RAW_LABEL]
    dataset = features.join(labels, how="left").replace([np.inf, -np.inf], np.nan)
    dataset = censor_evaluation_labels(dataset, CUTOFF)
    supervised = dataset.dropna(subset=[RAW_LABEL]).copy()
    supervised[RANK_LABEL] = cross_sectional_rank(supervised, [RAW_LABEL])[RAW_LABEL].astype("float32")
    supervised.to_parquet(DATASET_PATH)
    labels.to_parquet(DATA_DIR / "labels_uncensored.parquet")
    daily_counts = features.groupby(level="datetime").size().rename("feature_rows")
    daily_counts.to_csv(REPORT_DIR / "daily_feature_counts.csv")
    if features.index.get_level_values("datetime").max() != CUTOFF:
        raise ValueError("最新特征未覆盖7月最后交易日")
    metadata = {
        "cutoff": str(CUTOFF.date()),
        "feature_rows": len(features),
        "supervised_rows": len(supervised),
        "last_prediction_feature_date": str(features.index.get_level_values("datetime").max().date()),
        "last_mature_label_date": str(supervised.index.get_level_values("datetime").max().date()),
        "historical_feature_source": str(SOURCE_RANKED),
        "historical_features_preserved": True,
        "last_2026_disclosure_report_period": str(disclosures.report_date.max().date()),
        "fitting_boundary": "purge final 10 signal days in each training/validation period",
        "prediction_boundary": "keep all feature dates; only IC labels are maturity-censored",
    }
    (REPORT_DIR / "data_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Dataset ready:", json.dumps(metadata, indent=2), flush=True)
    gc.collect()


def predict_year(
    booster: lgb.Booster,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    dates = features.index.get_level_values("datetime")
    frame = features.loc[dates.year == year]
    if frame.empty:
        raise ValueError(f"{year}年没有预测特征")
    result = pd.DataFrame(
        {"prediction": booster.predict(frame[ENHANCED_FEATURE_NAMES], num_threads=4)},
        index=frame.index,
    )
    result = result.join(labels[[RAW_LABEL]], how="left")
    return censor_evaluation_labels(result, min(pd.Timestamp(f"{year}-12-31"), CUTOFF))


def train_models() -> None:
    init_qlib()
    strategy = load_main_strategy_config()
    model_config = load_model_config()
    research = load_research_config()
    features = pd.read_parquet(FEATURE_PATH)
    dataset = pd.read_parquet(DATASET_PATH)
    labels = pd.read_parquet(DATA_DIR / "labels_uncensored.parquet")
    parameters = deepcopy(model_config["parameters"])
    parameters["n_jobs"] = 4
    candidates = [int(value) for value in strategy["iteration_candidates"]]
    predictions = []
    training_records = []
    candidate_records = []
    fold_records = []
    validation_cache = {}
    original_summary = pd.read_csv(PROJECT_ROOT / "reports/experiments/main_walk_forward_cv/training_summary.csv")

    for year in range(2022, 2027):
        print(f"\nPreparing annual model {year}", flush=True)
        if year <= 2024:
            model_path = SOURCE_MODELS / f"ranked_enhanced_10d_test_{year}.txt"
            booster = lgb.Booster(model_file=str(model_path))
            record = original_summary.loc[original_summary.test_year == year].iloc[0].to_dict()
            record["model_origin"] = "saved frozen annual model"
        else:
            current_records = []
            folds = build_validation_folds(year, int(strategy["validation_folds"]), 2010)
            for fold in folds:
                validation_year = fold["validation_year"]
                if validation_year not in validation_cache:
                    print(
                        f"  Selection fold: train {fold['train_start']} to {fold['train_end']}; "
                        f"validate {validation_year}",
                        flush=True,
                    )
                    validation_cache[validation_year] = fit_selection_fold(
                        dataset=dataset,
                        fold=fold,
                        feature_names=ENHANCED_FEATURE_NAMES,
                        target_name=RANK_LABEL,
                        raw_label_name=RAW_LABEL,
                        model_parameters=parameters,
                        iteration_candidates=candidates,
                        label_horizon=HORIZON,
                        min_cross_section_size=int(research["min_cross_section_size"]),
                    )
                    pd.DataFrame(validation_cache[validation_year]).to_csv(
                        REPORT_DIR / f"validation_fold_{validation_year}.csv", index=False
                    )
                    gc.collect()
                for cached in validation_cache[validation_year]:
                    current_records.append({**cached, "test_year": year})
            summary = summarize_iteration_candidates(current_records)
            summary.insert(0, "test_year", year)
            candidate_records.append(summary)
            fold_records.extend(current_records)
            chosen = int(summary.iloc[0]["candidate_iteration"])
            print(summary.to_string(index=False), flush=True)
            print(f"  Selected tree count: {chosen}", flush=True)
            train_start, train_end = latest_training_period(year, 2010)
            final_train = select_period(dataset, train_start, train_end, HORIZON, purge_end=True)
            fit_parameters = deepcopy(parameters)
            fit_parameters["n_estimators"] = chosen
            model = lgb.LGBMRegressor(**fit_parameters)
            model.fit(
                final_train[ENHANCED_FEATURE_NAMES],
                final_train[RANK_LABEL],
                callbacks=[lgb.log_evaluation(period=0)],
            )
            model_path = MODEL_DIR / f"ranked_enhanced_10d_test_{year}.txt"
            model.booster_.save_model(str(model_path))
            booster = model.booster_
            record = {
                "test_year": year,
                "chosen_iteration": chosen,
                "final_train_start": train_start,
                "final_train_end": train_end,
                "final_train_rows": len(final_train),
                "validation_fold_count": len(folds),
                "selection_mean_validation_rank_ic": float(summary.iloc[0]["mean_validation_rank_ic"]),
                "model_origin": "fresh fit, frozen selection rule",
            }
            del final_train, model
            gc.collect()
        if booster.feature_name() != ENHANCED_FEATURE_NAMES:
            raise ValueError("模型特征名称或顺序与输入不一致")
        predicted = predict_year(booster, features, labels, year)
        record["prediction_rows"] = len(predicted)
        record["prediction_start"] = str(predicted.index.get_level_values("datetime").min().date())
        record["prediction_end"] = str(predicted.index.get_level_values("datetime").max().date())
        record["model_path"] = str(model_path)
        predictions.append(predicted)
        training_records.append(record)
        predicted.to_parquet(REPORT_DIR / f"predictions_{year}.parquet")
        pd.DataFrame(training_records).to_csv(REPORT_DIR / "training_summary.csv", index=False)
        print(f"  Saved predictions: {len(predicted):,} rows through {record['prediction_end']}", flush=True)
        del booster, predicted
        gc.collect()

    combined = pd.concat(predictions).sort_index()
    if combined.index.duplicated().any() or not np.isfinite(combined.prediction).all():
        raise ValueError("预测存在重复索引或非有限值")
    combined.to_parquet(PREDICTION_PATH)
    pd.DataFrame(fold_records).to_csv(REPORT_DIR / "validation_folds.csv", index=False)
    pd.concat(candidate_records, ignore_index=True).to_csv(REPORT_DIR / "iteration_candidate_summary.csv", index=False)
    config_snapshot = {
        "main_strategy": strategy,
        "model_parameters": parameters,
        "cutoff": str(CUTOFF.date()),
        "annual_test_years": [2022, 2023, 2024, 2025, 2026],
        "prediction_boundary": "all feature dates; no test-end purge",
    }
    (REPORT_DIR / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")
    print("Training complete", flush=True)


def run_backtest() -> None:
    init_qlib()
    config = {**load_research_config(), **load_main_strategy_config()}
    predictions = pd.read_parquet(PREDICTION_PATH)
    # Keep the published research prefix exactly reproducible, including its
    # historical signal-date policy. New-year inference uses feature-only dates.
    frozen_path = (
        PROJECT_ROOT
        / "reports/experiments/main_walk_forward_cv/ranked_enhanced_10d_test_predictions.parquet"
    )
    frozen = pd.read_parquet(frozen_path)[["prediction", RAW_LABEL]]
    common = frozen.index.intersection(predictions.index)
    if not np.allclose(
        frozen.loc[common, "prediction"], predictions.loc[common, "prediction"],
        rtol=0, atol=1e-12,
    ):
        raise ValueError("旧模型预测与已发布的研究前缀不一致")
    new_dates = predictions.index.get_level_values("datetime")
    predictions = pd.concat(
        [frozen, predictions.loc[new_dates >= pd.Timestamp("2025-01-01")]]
    ).sort_index()
    if predictions.index.duplicated().any():
        raise ValueError("连续回测拼接了重复股票日期")
    predictions.to_parquet(REPORT_DIR / "backtest_predictions.parquet")
    (REPORT_DIR / "backtest_protocol.json").write_text(
        json.dumps(
            {
                "historical_prefix": str(frozen_path),
                "historical_predictions_identical": True,
                "new_prediction_years": [2025, 2026],
                "continuous_account": True,
                "cutoff": str(CUTOFF.date()),
                "new_year_end_prediction_policy": "keep all feature dates; censor only IC outcomes",
            }, indent=2,
        ), encoding="utf-8",
    )
    name = "ranked_enhanced_10d_until_2026_07"
    raw = run_tiered_backtest(
        prediction_frame=predictions,
        config=config,
        rebalance_every=int(config["rebalance_every"]),
        top_k=int(config["top_k"]),
        top_tier_size=int(config["top_tier_size"]),
        top_tier_weight=float(config["top_tier_weight"]),
        period_end=str(CUTOFF.date()),
        quote_codes=sorted(predictions.index.get_level_values("instrument").unique()),
    )
    report = enrich_daily_report(raw, name)
    if report.index.max() != CUTOFF:
        raise ValueError("账户回测未运行至7月末")
    report.to_csv(REPORT_DIR / "backtest_daily.csv", encoding="utf-8-sig")
    summary = summarize_report(report, name)
    yearly = summarize_report_by_year(report, name)
    yearly["start_date"] = [str(report.loc[report.index.year == year].index.min().date()) for year in yearly.year]
    yearly["end_date"] = [str(report.loc[report.index.year == year].index.max().date()) for year in yearly.year]
    yearly["period_type"] = np.where(yearly.year == 2026, "January-July, partial year", "calendar year")
    summary.to_csv(REPORT_DIR / "backtest_comparison.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(REPORT_DIR / "backtest_yearly.csv", index=False, encoding="utf-8-sig")

    ic_records = []
    for year, frame in predictions.groupby(predictions.index.get_level_values("datetime").year):
        # Late labels were censored before grouping; corr only uses matured outcomes.
        daily_ic = calculate_daily_model_ic(
            frame,
            min_cross_section_size=int(config["min_cross_section_size"]),
            label_name=RAW_LABEL,
        )
        daily_ic.to_csv(REPORT_DIR / f"daily_ic_{year}.csv", index=False)
        ic_records.append(
            {
                "year": year,
                "valid_days": len(daily_ic),
                "mean_ic": daily_ic.ic.mean(),
                "mean_rank_ic": daily_ic.rank_ic.mean(),
                "positive_rank_ic_ratio": daily_ic.rank_ic.gt(0).mean(),
            }
        )
    pd.DataFrame(ic_records).to_csv(REPORT_DIR / "test_ic_summary.csv", index=False)
    subperiod_records = []
    for label, start, end in [
        ("2022-2024, frozen research prefix", "2022-01-01", "2024-12-31"),
        ("2025-2026 July", "2025-01-01", str(CUTOFF.date())),
        ("2026 January-July", "2026-01-01", str(CUTOFF.date())),
    ]:
        sub = report.loc[start:end]
        sub_summary = summarize_report(sub, name)
        sub_summary.insert(0, "period", label)
        subperiod_records.append(sub_summary)
    pd.concat(subperiod_records, ignore_index=True).to_csv(REPORT_DIR / "subperiod_comparison.csv", index=False)
    print("\nOverall account results:\n" + summary.to_string(index=False), flush=True)
    print("\nYearly account results:\n" + yearly.to_string(index=False), flush=True)
    print("\nYearly prediction IC:\n" + pd.DataFrame(ic_records).to_string(index=False), flush=True)
    print(f"\nResults saved: {REPORT_DIR}", flush=True)


def save_performance_chart() -> Path:
    """Plot the saved account, without rerunning or changing its simulation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = pd.read_csv(REPORT_DIR / "backtest_daily.csv", index_col="datetime", parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(report.index, report.portfolio_nav, label="LightGBM strategy, net", color="#126b55", linewidth=1.7)
    axes[0].plot(report.index, report.benchmark_nav, label="CSI 300 benchmark", color="#4169ad", linewidth=1.5)
    axes[0].set_title("Annual rolling strategy through 2026-07-31")
    axes[0].set_ylabel("Growth of 1")
    axes[0].legend(loc="upper left")
    nav = report.portfolio_nav
    drawdown = nav / nav.cummax().clip(lower=1) - 1
    axes[1].fill_between(report.index, drawdown * 100, 0, color="#c67b3b", alpha=0.55)
    axes[1].set_ylabel("Portfolio drawdown (%)")
    for axis in axes:
        axis.axvline(pd.Timestamp("2025-01-01"), color="#808080", linewidth=1, linestyle="--")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output = REPORT_DIR / "nav_and_drawdown.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def audit_saved_backtest() -> None:
    """Independently recompute metrics and verify the published research prefix."""
    report = pd.read_csv(REPORT_DIR / "backtest_daily.csv", index_col="datetime", parse_dates=True)
    summary = pd.read_csv(REPORT_DIR / "backtest_comparison.csv").iloc[0]
    old = pd.read_csv(
        PROJECT_ROOT / "reports/main_strategy/backtest_daily.csv",
        index_col="datetime", parse_dates=True,
    )
    rate_columns = ["return", "cost", "bench", "net_return", "turnover"]
    differences = (report.loc[old.index, rate_columns] - old[rate_columns]).abs().max()
    account_difference = float((report.loc[old.index, "account"] - old.account).abs().max())
    if differences.max() > 1e-12 or account_difference > 1e-6:
        raise ValueError("连续账户未复现原2022—2024回测前缀")
    if not np.isfinite(report[["account", "net_return", "bench", "cost"]].to_numpy()).all():
        raise ValueError("账户数据存在非有限值")
    if (report.account <= 0).any() or (report.cost < 0).any():
        raise ValueError("账户净资产或费用异常")
    if report.index.max() != CUTOFF or report.index.duplicated().any():
        raise ValueError("账户日期边界或唯一性异常")
    wealth = float((1 + report.net_return).prod())
    benchmark_wealth = float((1 + report.bench).prod())
    rate = 252 / len(report)
    nav = np.r_[1.0, (1 + report.net_return).cumprod().to_numpy()]
    drawdown = float((nav / np.maximum.accumulate(nav) - 1).min())
    active = report.net_return - report.bench
    recomputed = {
        "total_return": wealth - 1,
        "annualized_return": wealth ** rate - 1,
        "total_excess_return": wealth / benchmark_wealth - 1,
        "annualized_excess_return": (wealth / benchmark_wealth) ** rate - 1,
        "information_ratio": float(active.mean() / active.std() * np.sqrt(252)),
        "max_drawdown": drawdown,
    }
    for key, value in recomputed.items():
        if not np.isclose(value, summary[key], rtol=0, atol=1e-12):
            raise ValueError(f"指标无法独立复算：{key}")
    alignment = pd.read_parquet(DATA_DIR / "financial_alignment_2026.parquet")
    alignment_dates = alignment.index.get_level_values("datetime")
    for column in ["fundamental_available_date", "annual_eps_available_date"]:
        if (alignment[column].notna() & alignment[column].gt(alignment_dates)).any():
            raise ValueError("2026财务数据提前可用")
    audit = {
        "passed": True,
        "last_account_date": str(report.index.max().date()),
        "trading_days": len(report),
        "historical_prefix_max_rate_difference": float(differences.max()),
        "historical_prefix_max_account_currency_difference": account_difference,
        "financial_availability_check": "passed",
        "independently_recomputed_metrics": recomputed,
    }
    (REPORT_DIR / "result_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Independent account and financial timing checks passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "data", "train", "backtest"], default="all")
    args = parser.parse_args()
    ensure_directories()
    if args.stage in {"all", "data"}:
        build_data()
    if args.stage in {"all", "train"}:
        train_models()
    if args.stage in {"all", "backtest"}:
        run_backtest()
        audit_saved_backtest()
        save_performance_chart()


if __name__ == "__main__":
    main()
