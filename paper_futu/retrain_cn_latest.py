"""One-off latest-mature-label refit, with an earlier-cutoff offline comparison.

No order API or deployment writer is used here.  All snapshots, models and
reports stay in a separate experiment.  The raw label is T+1 open to T+11 open.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.storage.file_storage import FileFeatureStorage
import requests

from paper_futu.cn_common import HISTORICAL_PROVIDER, ROOT, STATE_DIR, load_deployment
from src.build_enhanced_dataset import (
    merge_latest_event, prepare_annual_eps_events, prepare_general_report_events,
    validate_point_in_time,
)
from src.build_ranked_dataset import cross_sectional_rank
from src.compare_models import build_rebalance_signal, enrich_daily_report, summarize_report
from src.config import load_model_config, load_research_config
from src.evaluate_model import calculate_daily_model_ic
from src.factor_library import ENHANCED_FEATURE_NAMES, FEATURE_EXPRESSIONS
from paper_futu.build_cn_signal import read_saved_bars
from src.forward_labels import label_contract, label_end_dates as contract_end_dates, select_mature_period
from src.train_lightgbm import build_feature_importance
from src.train_walk_forward_cv import build_validation_folds, summarize_iteration_candidates

HORIZON = 10
LABEL = label_contract(HORIZON)["raw_label"]
TARGET = label_contract(HORIZON)["target"]
LABEL_EXPRESSION = label_contract(HORIZON)["label_expression"]
EARLIER_CUTOFF = pd.Timestamp("2026-06-30")
COMPARISON_START = pd.Timestamp("2026-07-01")
FEATURE_SOURCE = ROOT / "data/processed/main_until_2026_07/ranked_features.parquet"
FINANCIAL_SOURCE = ROOT / "data/external/fundamentals.parquet"
DISCLOSURE_SOURCE = ROOT / "data/processed/main_until_2026_07/disclosures_through_2026.parquet"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def protected_state() -> dict[str, str]:
    model_dir, _ = load_deployment()
    paths = [ROOT / "configs/futu_cn_paper.json", STATE_DIR / "active_plan.json",
             STATE_DIR / "catchup_plan.json", model_dir / "model.txt", model_dir / "metadata.json"]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths if path.exists()}


def experiment_paths(as_of: pd.Timestamp) -> dict[str, Path]:
    name = f"cn_latest_mature_open10_{as_of:%Y%m%d}"
    return {
        "data": ROOT / "data/processed" / name,
        "model": ROOT / "models/experiments" / name,
        "report": ROOT / "reports/experiments" / name,
    }


def label_end_dates(index: pd.MultiIndex, calendar: pd.DatetimeIndex) -> pd.Series:
    return contract_end_dates(index, calendar, HORIZON)


def select_mature(frame: pd.DataFrame, start: str | pd.Timestamp,
                  cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Use exact observed exit dates, including when later labels exist in memory."""
    result = select_mature_period(frame, start, cutoff).dropna(subset=[LABEL, TARGET])
    if result.empty:
        raise ValueError("No usable mature labels remain")
    return result


def make_provider(paths: dict[str, Path], as_of: pd.Timestamp, symbols: list[str]) -> Path:
    """Copy the quote snapshot; append verified local bars without touching live data."""
    provider = paths["data"] / "provider"
    if (provider / "snapshot_ready.json").exists():
        return provider
    provider.mkdir(parents=True, exist_ok=True)
    for name in ["calendars", "instruments"]:
        shutil.copytree(HISTORICAL_PROVIDER / name, provider / name, dirs_exist_ok=True)
    for code in sorted(set(symbols + ["SH000300"])):
        source = HISTORICAL_PROVIDER / "features" / code.lower()
        if not source.exists():
            raise FileNotFoundError(f"Missing historical quote folder: {code}")
        shutil.copytree(source, provider / "features" / code.lower(), dirs_exist_ok=True)
    calendar_path = provider / "calendars/day.txt"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    historical_end = calendar.max()
    calendar = calendar[calendar <= as_of]
    live_paths = [path for path in sorted(STATE_DIR.glob("*/bars.parquet"))
                  if historical_end < pd.Timestamp(path.parent.name) <= as_of]
    if as_of > historical_end and (not live_paths or pd.Timestamp(live_paths[-1].parent.name) != as_of):
        raise ValueError("No verified local closing bars for the requested cutoff")
    expected = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day_future.txt").read_text().splitlines()))
    expected = expected[(expected > historical_end) & (expected <= as_of)]
    observed = pd.DatetimeIndex([path.parent.name for path in live_paths])
    if not expected.equals(observed):
        raise ValueError("Local closing bars have missing or extra trading dates")
    calendar = calendar.union(observed).sort_values()
    calendar_path.write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    # Qlib's file-storage adapter needs the global mount configuration even
    # when its provider_uri is explicitly supplied for the isolated copy.
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1)
    local_manifest = []
    for path in live_paths:
        bars = read_saved_bars(path).sort_index()
        if bars.index.duplicated().any():
            raise ValueError("Duplicate local closing bars")
        date = pd.Timestamp(path.parent.name)
        snapshots = pd.read_csv(path.parent / "snapshots.csv").set_index("code")
        for (instrument, stamp), row in bars.iterrows():
            if pd.Timestamp(stamp) != date:
                raise ValueError("Local bar timestamp disagrees with its folder")
            quote = snapshots.loc[instrument[:2] + "." + instrument[2:]]
            if not bool(quote.suspension) and (not np.isfinite(row.factor) or row.factor <= 0
                    or not np.isclose(row.open / row.factor, quote.open_price, rtol=1e-5)
                    or not np.isclose(row.close / row.factor, quote.last_price, rtol=1e-5)):
                raise ValueError(f"Local adjusted/raw price mismatch: {instrument}")
            values = {**row.to_dict(), "change": float(quote.last_price / quote.prev_close_price - 1),
                      "amount": float(quote.turnover) / 1000}
            for field, value in values.items():
                storage = FileFeatureStorage(instrument, field, "day", provider_uri=str(provider))
                storage.write([float(value)], index=int(calendar.get_loc(date)))
        # Historical membership stops at the source's last bar. Extend only
        # members verified by the index publisher on this local signal date.
        members = pd.read_csv(path.parent / "members.csv")
        membership_path = provider / "instruments/csi300.txt"
        membership = pd.read_csv(membership_path, sep="\t", header=None,
                                 names=["instrument", "start", "end"])
        for instrument in members.instrument:
            rows = membership.index[membership.instrument.eq(instrument)]
            last = rows[-1] if len(rows) else None
            if last is not None and pd.Timestamp(membership.at[last, "end"]) == calendar[calendar < date].max():
                membership.at[last, "end"] = str(date.date())
            else:
                membership.loc[len(membership)] = [instrument, str(date.date()), str(date.date())]
        membership.to_csv(membership_path, sep="\t", header=False, index=False)
        benchmark_path = ROOT / f"reports/experiments/recent_30d_{date:%Y%m%d}/benchmark_snapshot.csv"
        benchmark = pd.read_csv(benchmark_path).iloc[0]
        if pd.Timestamp(benchmark.update_time).normalize() != date:
            raise ValueError("Benchmark closing snapshot is stale")
        index_position = int(calendar.get_loc(date))
        close_storage = FileFeatureStorage("SH000300", "close", "day", provider_uri=str(provider))
        _, previous = close_storage[index_position - 1]
        factor = float(previous / benchmark.prev_close_price)
        benchmark_values = {
            "open": benchmark.open_price * factor, "close": benchmark.last_price * factor,
            "high": benchmark.high_price * factor, "low": benchmark.low_price * factor,
            "factor": factor, "volume": benchmark.volume / factor,
            "change": benchmark.last_price / benchmark.prev_close_price - 1,
        }
        for field, value in benchmark_values.items():
            FileFeatureStorage("SH000300", field, "day", provider_uri=str(provider)).write([value], index=index_position)
        local_manifest.append({"date": str(date.date()), "bar_sha256": digest(path),
                               "membership_sha256": digest(path.parent / "members.csv"),
                               "benchmark_sha256": digest(benchmark_path)})
    save_json(provider / "snapshot_ready.json", {"as_of": as_of, "historical_end": historical_end,
                                                "symbols": len(symbols), "local_bars": local_manifest})
    return provider


def load_financials(paths: dict[str, Path], as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Check actual Q3 disclosures; only locally verified financial values enter features."""
    values = pd.read_parquet(FINANCIAL_SOURCE)
    disclosures = pd.read_parquet(DISCLOSURE_SOURCE)
    lookup = {code[-6:]: code for code in values.instrument.unique()}
    source = "https://www.cninfo.com.cn/new/information/getPrbookInfo"
    cache = paths["data"] / "cninfo_2026_09_30.json"
    request_error = None
    try:
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = requests.post(source, params={
                "sectionTime": "2026-09-30", "firstTime": "", "lastTime": "", "market": "szsh",
                "stockCode": "", "orderClos": "", "isDesc": "", "pagesize": "10000", "pagenum": "1",
            }, timeout=45)
            response.raise_for_status()
            payload = response.json()
            save_json(cache, payload)
        if payload.get("hasNextPage") or not payload.get("prbookinfos"):
            raise ValueError("CNINFO Q3 disclosure response is incomplete")
        raw = pd.DataFrame(payload["prbookinfos"])
        recent = pd.DataFrame({
            "instrument": raw.seccode.astype(str).str.zfill(6).map(lookup),
            "report_date": pd.to_datetime(raw.f001d_0102),
            "disclosure_date": pd.to_datetime(raw.f006d_0102, errors="coerce"),
        }).dropna(subset=["instrument", "report_date"])
        if not recent.report_date.eq(pd.Timestamp("2026-09-30")).all():
            raise ValueError("CNINFO returned the wrong report period")
        disclosures = pd.concat([disclosures, recent], ignore_index=True).drop_duplicates(
            ["instrument", "report_date"], keep="last")
    except (requests.RequestException, ValueError, KeyError, AttributeError) as error:
        request_error = str(error)
    disclosures = disclosures[["instrument", "report_date", "disclosure_date"]].drop_duplicates(
        ["instrument", "report_date"], keep="last")
    financials = values.merge(disclosures, on=["instrument", "report_date"], how="inner",
                              validate="one_to_one").dropna(subset=["disclosure_date"])
    financials["available_date"] = financials.disclosure_date + pd.Timedelta(days=1)
    if financials.available_date.le(financials.report_date).any():
        raise ValueError("Financial values available before their report period ended")
    financials = financials.loc[financials.available_date.le(as_of)].copy()
    announced = disclosures.loc[disclosures.disclosure_date.le(as_of)].copy()
    missing = announced.merge(values[["instrument", "report_date"]], on=["instrument", "report_date"],
                              how="left", indicator=True)
    missing = missing.loc[missing._merge.eq("left_only")].drop(columns="_merge")
    missing.to_csv(paths["report"] / "disclosed_but_unverified_values.csv", index=False, encoding="utf-8-sig")
    financials.to_parquet(paths["data"] / "financial_events.parquet", index=False)
    return financials, {
        "financial_source_sha256": digest(FINANCIAL_SOURCE),
        "disclosure_source_sha256": digest(DISCLOSURE_SOURCE),
        "latest_verified_report_period": str(financials.report_date.max().date()),
        "announced_rows_without_verified_values": len(missing),
        "new_disclosure_query_url": source, "new_disclosure_query_error": request_error,
        "fallback": "Use each stock's last verified report if newer values cannot be verified",
        "historical_revision_limitation": "Disclosure dates are aligned; financial source is not a full as-reported revision archive",
    }


def build_data(paths: dict[str, Path], as_of: pd.Timestamp) -> None:
    print("Building frozen quote/feature snapshot", flush=True)
    historical = pd.read_parquet(FEATURE_SOURCE, columns=ENHANCED_FEATURE_NAMES).sort_index()
    historical = historical.loc[historical.index.get_level_values("datetime") <= as_of]
    last_cached = historical.index.get_level_values("datetime").max()
    local = pd.read_parquet(STATE_DIR / str(as_of.date()) / "features_ranked.parquet")
    symbols = sorted(set(historical.index.get_level_values("instrument"))
                     | set(local.index.get_level_values("instrument")))
    provider = make_provider(paths, as_of, symbols)
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    if calendar.max() != as_of:
        raise ValueError("Snapshot calendar does not end at the fixed cutoff")
    financials, financial_audit = load_financials(paths, as_of)
    market = D.instruments("csi300")
    expressions = list(FEATURE_EXPRESSIONS.values()) + ["$close / ($factor + 1e-12)"]
    recent = D.features(market, expressions, start_time=str((last_cached + pd.Timedelta(days=1)).date()),
                        end_time=str(as_of.date()), freq="day")
    recent.columns = list(FEATURE_EXPRESSIONS) + ["raw_close"]
    recent = recent.replace([np.inf, -np.inf], np.nan)
    recent = recent.loc[~recent[list(FEATURE_EXPRESSIONS)].isna().all(axis=1)].reset_index()
    recent = merge_latest_event(recent, prepare_general_report_events(financials), "fundamental_available_date")
    recent = merge_latest_event(recent, prepare_annual_eps_events(financials), "annual_eps_available_date")
    validate_point_in_time(recent)
    price = recent.raw_close.where(recent.raw_close > 0)
    recent["book_to_price"] = recent.book_value_per_share / price
    recent["earnings_to_price"] = recent.annual_eps / price
    recent = recent.set_index(["instrument", "datetime"]).replace([np.inf, -np.inf], np.nan).sort_index()
    coverage = recent[ENHANCED_FEATURE_NAMES].notna().mean().rename("coverage_ratio")
    coverage.to_csv(paths["report"] / "recent_feature_coverage.csv")
    recent[["fundamental_report_date", "fundamental_available_date", "annual_eps_report_date",
            "annual_eps_available_date"]].to_parquet(paths["data"] / "recent_financial_alignment.parquet")
    ranked = cross_sectional_rank(recent, ENHANCED_FEATURE_NAMES).astype("float32")
    # The live builder verified these exact-day formulas and filtered suspension.
    # Reuse that verified cross-section; never rank today's missing suspended row.
    ranked = ranked.loc[ranked.index.get_level_values("datetime") != as_of]
    local = local[ENHANCED_FEATURE_NAMES].astype("float32")
    features = pd.concat([historical, ranked, local]).sort_index()
    if features.index.duplicated().any() or np.isinf(features.to_numpy()).any():
        raise ValueError("Duplicate feature index or infinite feature values")
    if features.index.get_level_values("datetime").max() != as_of:
        raise ValueError("Inference features do not include the latest completed session")
    if not features.stack().between(-0.5, 0.5).all():
        raise ValueError("Unexpected ranked-feature scale")
    features.to_parquet(paths["data"] / "ranked_features.parquet")
    print(f"Computing aligned labels for {len(features):,} stock-days", flush=True)
    labels = D.features(market, [LABEL_EXPRESSION], start_time="2010-01-01", end_time=str(as_of.date()), freq="day")
    labels.columns = [LABEL]
    dataset = features.join(labels, how="left").replace([np.inf, -np.inf], np.nan)
    dataset["label_end_date"] = label_end_dates(dataset.index, calendar)
    dataset.loc[dataset.label_end_date.isna() | dataset.label_end_date.gt(as_of), LABEL] = np.nan
    supervised = dataset.dropna(subset=[LABEL]).copy()
    supervised = supervised.loc[~supervised[ENHANCED_FEATURE_NAMES].isna().all(axis=1)]
    supervised[TARGET] = cross_sectional_rank(supervised, [LABEL])[LABEL].astype("float32")
    supervised.to_parquet(paths["data"] / "supervised_dataset.parquet")
    dataset[[LABEL, "label_end_date"]].to_parquet(paths["data"] / "evaluation_labels.parquet")
    last_signal = supervised.index.get_level_values("datetime").max()
    expected_last = calendar[-HORIZON - 2]
    if last_signal != expected_last:
        raise ValueError(f"Latest mature signal is {last_signal}, expected {expected_last}")
    audit = {
        "as_of": str(as_of.date()), "label_expression": LABEL_EXPRESSION,
        "label_forward_calendar_steps": HORIZON + 1,
        "prediction_feature_rows": len(features), "supervised_rows": len(supervised),
        "latest_feature_date": str(as_of.date()), "latest_mature_signal_date": str(last_signal.date()),
        "training_label_end_max": str(supervised.label_end_date.max().date()),
        "mature_missing_label_rows": int((dataset.label_end_date.le(as_of) & dataset[LABEL].isna()).sum()),
        "unmatured_prediction_rows_preserved": int(dataset.label_end_date.isna().sum()),
        "historical_feature_sha256": digest(FEATURE_SOURCE), "provider": str(provider.relative_to(ROOT)),
        "dataset_sha256": digest(paths["data"] / "supervised_dataset.parquet"),
        "ranked_features_sha256": digest(paths["data"] / "ranked_features.parquet"),
        **financial_audit,
    }
    save_json(paths["report"] / "data_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    gc.collect()


def train_models(paths: dict[str, Path], as_of: pd.Timestamp) -> None:
    dataset_path = paths["data"] / "supervised_dataset.parquet"
    dataset = pd.read_parquet(dataset_path)
    parameters = deepcopy(load_model_config()["parameters"])
    parameters["n_jobs"] = 4
    parameters["random_state"] = 42
    candidates = [10, 20, 40, 80, 160, 320, 640]
    records = []
    for fold in build_validation_folds(2026, 3, 2010):
        train = select_mature(dataset, fold["train_start"], fold["train_end"])
        valid = select_mature(dataset, fold["valid_start"], fold["valid_end"])
        print(f"Selection year {fold['validation_year']}: {len(train):,} train / {len(valid):,} validation", flush=True)
        model = lgb.LGBMRegressor(**{**parameters, "n_estimators": max(candidates)})
        model.fit(train[ENHANCED_FEATURE_NAMES], train[TARGET])
        for candidate in candidates:
            prediction = pd.DataFrame({"prediction": model.predict(valid[ENHANCED_FEATURE_NAMES],
                                                                     num_iteration=candidate),
                                       LABEL: valid[LABEL]}, index=valid.index)
            daily = calculate_daily_model_ic(prediction, int(load_research_config()["min_cross_section_size"]), LABEL)
            records.append({**fold, "candidate_iteration": candidate, "train_rows": len(train),
                            "valid_rows": len(valid), "train_label_end_max": str(train.label_end_date.max().date()),
                            "valid_label_end_max": str(valid.label_end_date.max().date()),
                            "valid_mean_ic": float(daily.ic.mean()), "valid_mean_rank_ic": float(daily.rank_ic.mean()),
                            "valid_mse": float(np.mean((prediction.prediction.to_numpy() - valid[TARGET].to_numpy()) ** 2))})
        pd.DataFrame(records).to_csv(paths["report"] / "validation_folds.csv", index=False)
        del train, valid, model
        gc.collect()
    selections = summarize_iteration_candidates(records)
    selections.to_csv(paths["report"] / "iteration_candidates.csv", index=False)
    chosen = int(selections.iloc[0].candidate_iteration)
    print(f"Selected {chosen} trees by three-fold mean Rank IC", flush=True)
    _, deployed_metadata = load_deployment()
    for name, cutoff in [("refreshed_at_20260630", EARLIER_CUTOFF), ("latest", as_of)]:
        training = select_mature(dataset, "2010-01-01", cutoff)
        print(f"Fresh fit {name}: {len(training):,} rows, label outcomes through {cutoff.date()}", flush=True)
        model = lgb.LGBMRegressor(**{**parameters, "n_estimators": chosen})
        model.fit(training[ENHANCED_FEATURE_NAMES], training[TARGET])
        directory = paths["model"] / name
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.txt"
        model.booster_.save_model(str(model_path))
        check = training[ENHANCED_FEATURE_NAMES].iloc[::max(1, len(training) // 2000)]
        loaded = lgb.Booster(model_file=str(model_path))
        original_prediction = model.predict(check)
        restored_prediction = loaded.predict(check, num_threads=4)
        if not np.allclose(original_prediction, restored_prediction, rtol=0, atol=1e-12):
            raise ValueError("Saved/reloaded model predictions differ")
        build_feature_importance(model, ENHANCED_FEATURE_NAMES).to_csv(
            paths["report"] / f"{name}_feature_importance.csv", index=False)
        metadata = {
            **label_contract(HORIZON),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(), "model_year": 2026,
            "training_mode": "one_off_expanding_with_mature_open_to_open_labels",
            "train_start": "2010-01-01", "information_cutoff": str(cutoff.date()),
            "last_retained_signal_date": str(training.index.get_level_values("datetime").max().date()),
            "last_label_end_date": str(training.label_end_date.max().date()), "train_rows": len(training),
            "label_horizon": HORIZON, "label_expression": LABEL_EXPRESSION,
            "target": TARGET, "chosen_iteration": chosen, "validation_years": [2023, 2024, 2025],
            "selection_mean_rank_ic": float(selections.iloc[0].mean_validation_rank_ic),
            "features": ENHANCED_FEATURE_NAMES, "parameters": {**parameters, "n_estimators": chosen},
            "model_sha256": digest(model_path), "dataset_sha256": digest(dataset_path),
            "includes_2026_mature_outcomes": True, "saved_prediction_check_max_abs_error": float(np.max(np.abs(original_prediction-restored_prediction))),
            "portfolio": deployed_metadata["portfolio"], "deployment_status": "offline_only",
        }
        save_json(directory / "metadata.json", metadata)
        del training, model, loaded
        gc.collect()


def run_comparison_backtest(predictions: pd.DataFrame, name: str,
                            paths: dict[str, Path], as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.evaluate import backtest_daily
    from src.portfolio_strategy import TieredTopKWeightStrategy

    signal, signal_dates = build_rebalance_signal(predictions, rebalance_every=10)
    strategy = TieredTopKWeightStrategy(
        signal=signal, top_k=10, n_drop=3, top_tier_size=2, top_tier_weight=0.4,
        only_tradable=True, risk_degree=0.95,
    )
    print(f"Offline replay {name}: {len(signal_dates)} rebalance signals", flush=True)
    raw, positions = backtest_daily(
        # WeightStrategyBase sizes orders using yesterday's close. Qlib's
        # exchange must include the signal day or the initial target is zero.
        # This cash-only warm-up day is excluded from reported performance.
        start_time=EARLIER_CUTOFF, end_time=as_of, strategy=strategy,
        executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False),
        account=1_000_000, benchmark="SH000300",
        exchange_kwargs={"codes": sorted(set(predictions.index.get_level_values("instrument"))),
                         "deal_price": "open", "limit_threshold": float(load_research_config()["limit_threshold"]),
                         "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5.0, "trade_unit": 100},
    )
    warmup = raw.loc[raw.index < COMPARISON_START]
    if not warmup.turnover.eq(0).all() or not np.allclose(warmup.account, 1_000_000):
        raise ValueError("Warm-up session unexpectedly traded or changed starting capital")
    raw = raw.loc[raw.index >= COMPARISON_START].copy()
    positions = {date: position for date, position in positions.items() if pd.Timestamp(date) >= COMPARISON_START}
    if raw.empty or raw.turnover.iloc[0] <= 0:
        raise ValueError("Initial July 1 opening investment was not executed")
    # Both strategies start in cash and enter at the first session's open.
    # Match the benchmark's first-day investment interval to that same open.
    benchmark = D.features(["SH000300"], ["$open", "$close"], start_time=str(COMPARISON_START.date()),
                            end_time=str(as_of.date()), freq="day").droplevel("instrument")
    raw.loc[raw.index[0], "bench"] = benchmark.iloc[0]["$close"] / benchmark.iloc[0]["$open"] - 1
    report = enrich_daily_report(raw, name)
    previous_nav = report.portfolio_nav.shift(1, fill_value=1.0) * 1_000_000
    report["cost_cash"] = report.cost * previous_nav
    report["traded_notional"] = report.turnover * previous_nav
    holding_rows = []
    for date, position in positions.items():
        for instrument in position.get_stock_list():
            holding_rows.append({"datetime": date, "instrument": instrument,
                                 "adjusted_amount": position.get_stock_amount(instrument),
                                 "cash": position.get_cash(), "account_value": position.calculate_value()})
    holdings = pd.DataFrame(holding_rows)
    counts = holdings.groupby("datetime").instrument.nunique().reindex(report.index, fill_value=0)
    report["holdings_count"] = counts
    if not np.isfinite(report[["net_return", "bench", "cost", "turnover"]].to_numpy()).all():
        raise ValueError("Nonfinite backtest returns or costs")
    if not np.allclose(report.portfolio_nav * 1_000_000, report["account"], rtol=1e-6):
        raise ValueError("Compounded returns disagree with Qlib account values")
    if (counts > 10).any():
        raise ValueError("Backtest unexpectedly exceeded Top10 holdings")
    report.to_csv(paths["report"] / f"{name}_daily_backtest.csv", encoding="utf-8-sig")
    holdings.to_parquet(paths["report"] / f"{name}_holdings.parquet", index=False)
    summary = summarize_report(report, name)
    summary["cost_cash_total"] = report.cost_cash.sum()
    summary["traded_notional_total"] = report.traded_notional.sum()
    summary["average_holdings"] = counts.mean()
    summary["minimum_holdings"] = counts.min()
    return report, summary


def compare_models(paths: dict[str, Path], as_of: pd.Timestamp) -> None:
    qlib.init(provider_uri=str(paths["data"] / "provider"), region=REG_CN, kernels=1)
    features = pd.read_parquet(paths["data"] / "ranked_features.parquet")
    dates = features.index.get_level_values("datetime")
    features = features.loc[dates >= EARLIER_CUTOFF].copy()
    labels = pd.read_parquet(paths["data"] / "evaluation_labels.parquet")
    missing = labels.loc[labels.label_end_date.le(as_of) & labels[LABEL].isna()]
    missing.groupby(missing.index.get_level_values("datetime").year).size().rename(
        "mature_but_price_unavailable_rows").to_csv(paths["report"] / "missing_labels_by_year.csv")
    baseline_dir = paths["model"] / "frozen_baseline"
    baseline_metadata = json.loads((baseline_dir / "metadata.json").read_text(encoding="utf-8"))
    results, backtests = [], []
    for name, directory in [("frozen_old", baseline_dir), ("refreshed_at_20260630", paths["model"] / "refreshed_at_20260630")]:
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        if booster.feature_name() != ENHANCED_FEATURE_NAMES:
            raise ValueError("Model feature order differs")
        prediction = pd.DataFrame({"prediction": booster.predict(features, num_threads=4)}, index=features.index)
        prediction = prediction.join(labels, how="left")
        if not np.isfinite(prediction.prediction).all():
            raise ValueError("Nonfinite model predictions")
        prediction.to_parquet(paths["report"] / f"{name}_comparison_predictions.parquet")
        raw, summary = run_comparison_backtest(prediction, name, paths, as_of)
        evaluation = prediction.loc[prediction.index.get_level_values("datetime") >= COMPARISON_START]
        daily_ic = calculate_daily_model_ic(evaluation, int(load_research_config()["min_cross_section_size"]), LABEL)
        daily_ic.to_csv(paths["report"] / f"{name}_mature_daily_ic.csv", index=False)
        summary["mature_ic_days"] = int(daily_ic.rank_ic.notna().sum())
        summary["mean_ic"] = daily_ic.ic.mean()
        summary["mean_rank_ic"] = daily_ic.rank_ic.mean()
        summary["rank_icir_non_annualized"] = daily_ic.rank_ic.mean() / daily_ic.rank_ic.std()
        results.append(summary)
        backtests.append(raw)
    if not backtests[0].index.equals(backtests[1].index) or not np.allclose(backtests[0].bench, backtests[1].bench):
        raise ValueError("Comparison calendar or benchmark differs")
    comparison = pd.concat(results, ignore_index=True)
    comparison.to_csv(paths["report"] / "model_comparison.csv", index=False, encoding="utf-8-sig")
    verify_benchmark(paths, as_of, float(comparison.iloc[0].benchmark_total_return))
    latest_features = features.loc[features.index.get_level_values("datetime") == as_of]
    latest = lgb.Booster(model_file=str(paths["model"] / "latest/model.txt"))
    old = lgb.Booster(model_file=str(baseline_dir / "model.txt"))
    ranking = pd.DataFrame({"prediction": latest.predict(latest_features, num_threads=4),
                            "old_prediction": old.predict(latest_features, num_threads=4)}, index=latest_features.index)
    raw_latest = pd.read_parquet(STATE_DIR / str(as_of.date()) / "features_raw.parquet")
    snapshots = pd.read_csv(STATE_DIR / str(as_of.date()) / "snapshots.csv").set_index("code")
    ranking["code"] = ranking.index.get_level_values("instrument").map(lambda value: value[:2] + "." + value[2:])
    ranking["name"] = ranking.code.map(snapshots["name"])
    ranking["raw_close"] = raw_latest.raw_close.reindex(ranking.index)
    ranking["eligible"] = ~ranking.code.map(snapshots.suspension).fillna(True) & ~ranking.name.str.contains("ST", na=False)
    ranking["new_rank"] = ranking.prediction.rank(ascending=False, method="first").astype(int)
    ranking["old_rank"] = ranking.old_prediction.rank(ascending=False, method="first").astype(int)
    ranking = ranking.sort_values(["prediction", "code"], ascending=[False, True])
    ranking.to_csv(paths["report"] / "latest_signal_ranking.csv", encoding="utf-8-sig")
    eligible = ranking.loc[ranking.eligible]
    new_top = eligible.head(10).code.tolist()
    old_top = eligible.sort_values(["old_prediction", "code"], ascending=[False, True]).head(10).code.tolist()
    signal_summary = {"signal_date": str(as_of.date()), "prediction_rows": len(ranking),
                      "new_top10": new_top, "old_top10": old_top,
                      "top10_overlap": len(set(new_top).intersection(old_top)),
                      "not_an_execution_plan": True, "unmatured_latest_rows_predicted": True}
    save_json(paths["report"] / "latest_signal_summary.json", signal_summary)
    save_json(paths["report"] / "comparison_assumptions.json", {
        "start": str(COMPARISON_START.date()), "end": str(as_of.date()), "initial_cash": 1_000_000,
        "portfolio": {**baseline_metadata["portfolio"], "initial_cash": 1_000_000},
        "benchmark": "SH000300", "benchmark_first_day": "First-session open to close",
        "execution": "Ideal next-session open; daily closes mark remaining holdings",
        "research_status": "Historical replay of previously studied dates, not a new untouched final test",
        "combined_changes": "New open-exit label plus 2026 mature training samples; no separate causal attribution",
        "constraints": "Inherited Qlib daily engine uses uniform 100-share units and conservative common limit threshold; simulator/board rules can differ",
        "small_accounts": "5k/10k feasibility is reported separately; these returns are for 1 million cash",
        "live_fill_audit": "reports/paper_futu/opening_execution_gap_2026-10-13.csv; single-day simulation observation only",
    })
    save_comparison_chart(paths, backtests)
    write_research_note(paths, comparison, signal_summary)
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(signal_summary, ensure_ascii=False, indent=2), flush=True)


def verify_benchmark(paths: dict[str, Path], as_of: pd.Timestamp, measured: float) -> None:
    """Independently check the price-index interval against Tencent raw daily bars."""
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    cache = paths["data"] / "benchmark_tencent_raw.json"
    audit = {"source_url": url, "symbol": "sh000300", "start": str(COMPARISON_START.date()),
             "end": str(as_of.date()), "measured_open_to_last_close_return": measured}
    try:
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = requests.get(url, params={"_var": "kline_dayqfq",
                "param": f"sh000300,day,{COMPARISON_START.date()},{as_of.date()},640,qfq"}, timeout=30)
            response.raise_for_status()
            text = response.text
            payload = json.loads(text[text.find("={") + 1:] if "={" in text else text)
            save_json(cache, payload)
        data = payload["data"]["sh000300"]
        bars = pd.DataFrame(data.get("day") or data["qfqday"]).iloc[:, :6]
        bars.columns = ["datetime", "open", "close", "high", "low", "volume"]
        bars["datetime"] = pd.to_datetime(bars.datetime)
        bars = bars.loc[bars.datetime.between(COMPARISON_START, as_of)].sort_values("datetime")
        bars[["open", "close"]] = bars[["open", "close"]].astype(float)
        expected_calendar = pd.DatetimeIndex(D.calendar(start_time=COMPARISON_START, end_time=as_of, freq="day"))
        if not pd.DatetimeIndex(bars.datetime).equals(expected_calendar):
            raise ValueError("Independent benchmark bars do not cover the comparison calendar")
        independent = float(bars.close.iloc[-1] / bars.open.iloc[0] - 1)
        audit.update({"verified": True, "rows": len(bars), "independent_interval_return": independent,
                      "absolute_return_difference": abs(independent - measured),
                      "first_open_points": float(bars.open.iloc[0]), "last_close_points": float(bars.close.iloc[-1])})
        bars.to_csv(paths["report"] / "benchmark_independent_bars.csv", index=False)
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
        audit.update({"verified": False, "error": str(error)})
    save_json(paths["report"] / "benchmark_verification.json", audit)
    if audit.get("verified") and audit["absolute_return_difference"] > 0.001:
        raise ValueError("Independent benchmark return disagrees materially with Qlib snapshot")


def write_research_note(paths: dict[str, Path], comparison: pd.DataFrame, signals: dict) -> None:
    audit = json.loads((paths["report"] / "data_audit.json").read_text(encoding="utf-8"))
    latest = json.loads((paths["model"] / "latest/metadata.json").read_text(encoding="utf-8"))
    rows = ["# 最新成熟标签单次重训与历史对照", "",
            f"数据截止：{audit['as_of']}；最新完整标签信号：{audit['latest_mature_signal_date']}。",
            f"最终拟合 {latest['train_rows']:,} 行，三折选择 {latest['chosen_iteration']} 棵树。",
            "", "标签：T+1 开盘至 T+11 开盘收益，按当日股票横截面转为排名减0.5。",
            "", "对照从2026-07-01的100万元现金开始，到2026-10-12收盘估值；新对照模型仅使用2026-06-30前已知标签。",
            "", "| 模型 | 区间净收益 | 沪深300 | 几何超额 | 最大回撤 | 成本金额 | 平均持仓数 | Rank IC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in comparison.itertuples():
        rows.append(f"| {item.model} | {item.total_return:.2%} | {item.benchmark_total_return:.2%} | "
                    f"{item.total_excess_return:.2%} | {item.max_drawdown:.2%} | {item.cost_cash_total:.2f} | "
                    f"{item.average_holdings:.2f} | {item.mean_rank_ic:.4f} |")
    verification = json.loads((paths["report"] / "benchmark_verification.json").read_text(encoding="utf-8"))
    rows.extend(["", "![净值及回撤对照](performance_comparison.png)", "",
                 f"最新信号Top10与旧模型重合 {signals['top10_overlap']} 只。详见 latest_signal_ranking.csv。",
                 f"基准独立校验状态：{verification['verified']}，详见 benchmark_verification.json。",
                 "", "## 口径与局限", "",
                 "- 最新最终模型没有用训练覆盖期回测收益作为能力证明；历史比较使用另一个较早截止版本。",
                 "- 收益差异包含标签修改和训练数据更新的综合效果；历史区间已被研究过，不是全新独立测试。",
                 "- 日线回测按理想开盘成交；09:31限价模拟成交不同。现有引擎仍统一100股单位和涨跌停阈值，科创板等规则差异尚未重建。",
                 "- Qlib权重下单器需要前一日行情：加入6月30日现金初始化，7月1日已实际建仓；统计收益仍从7月1日起。",
                 "- 财务披露日期进行了对齐，但本地财务值并非完整的历史修订版本档案。",
                 f"- 已验证财务值最新报告期为 {audit['latest_verified_report_period']}；{audit['announced_rows_without_verified_values']} 条已披露报告缺少可验证新值，沿用旧报告。",
                 "- 当前模拟盘模型、订单计划和后台调度未由本实验修改；结果未提交Git。",
                 "", "复现：python -m paper_futu.retrain_cn_latest --as-of 2026-10-12 --stage all"])
    (paths["report"] / "research_note.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_comparison_chart(paths: dict[str, Path], reports: list[pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
    colors = ["#345ea8", "#188568"]
    labels = ["Frozen old model", "Refit as of 2026-06-30"]
    for report, color, label in zip(reports, colors, labels):
        nav = report.portfolio_nav
        axes[0].plot(nav.index, nav, color=color, label=label, linewidth=1.6)
        peak = nav.cummax().clip(lower=1)
        axes[1].plot(nav.index, (nav / peak - 1) * 100, color=color, linewidth=1.2)
    axes[0].plot(reports[0].index, reports[0].benchmark_nav, color="#888888", label="CSI 300", linewidth=1.3)
    axes[0].set_ylabel("Growth of 1 (net of costs)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[0].set_title("2026-07-01 to 2026-10-12 | Top10, 10-session rebalance | CNY 1 million")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(paths["report"] / "performance_comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True,
                        help="Completed session with verified local bars, features and benchmark snapshot")
    parser.add_argument("--stage", choices=["data", "train", "compare", "all"], default="all")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of).normalize()
    if as_of <= EARLIER_CUTOFF:
        raise ValueError("Cutoff must follow the historical comparison cutoff")
    paths = experiment_paths(as_of)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    before = protected_state()
    frozen_dir = paths["model"] / "frozen_baseline"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    deployed_dir, _ = load_deployment()
    for name in ["model.txt", "metadata.json"]:
        destination = frozen_dir / name
        if not destination.exists():
            shutil.copy2(deployed_dir / name, destination)
    frozen_metadata = json.loads((frozen_dir / "metadata.json").read_text(encoding="utf-8"))
    if digest(frozen_dir / "model.txt") != frozen_metadata["model_sha256"]:
        raise ValueError("Frozen baseline model hash changed")
    save_json(paths["report"] / "protected_state_before.json", before)
    try:
        if args.stage in ["data", "all"]:
            build_data(paths, as_of)
        if args.stage in ["train", "all"]:
            train_models(paths, as_of)
        if args.stage in ["compare", "all"]:
            compare_models(paths, as_of)
    finally:
        after = protected_state()
        save_json(paths["report"] / "protected_state_check.json", {"unchanged": before == after,
                                                                   "before": before, "after": after})
        if before != after:
            raise RuntimeError("Protected deployment or plan changed during this experiment; review external activity")
    print(f"Completed stage {args.stage}: {paths['report']}", flush=True)


if __name__ == "__main__":
    main()
