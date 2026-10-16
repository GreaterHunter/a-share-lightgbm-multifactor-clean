"""Extend frozen annual predictions and report the last 30 continuous-account sessions."""

from __future__ import annotations

import hashlib
import json
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.storage.file_storage import FileFeatureStorage

from paper_futu.build_cn_signal import disclosed_financials
from paper_futu.cn_common import HISTORICAL_PROVIDER, MODEL_DIR, ROOT, STATE_DIR
from src.analyze_portfolio_robustness import run_tiered_backtest
from src.build_enhanced_dataset import merge_latest_event, prepare_annual_eps_events, prepare_general_report_events, validate_point_in_time
from src.build_ranked_dataset import cross_sectional_rank
from src.compare_models import calculate_max_drawdown, enrich_daily_report
from src.config import load_main_strategy_config, load_research_config
from src.factor_library import ENHANCED_FEATURE_NAMES, FEATURE_EXPRESSIONS

DATE = "2026-10-12"
REPORT = ROOT / "reports/experiments/recent_30d_20261012"
PROVIDER = ROOT / "data/interim/recent_30d_20261012/provider"


def build_predictions():
    qlib.init(provider_uri=str(HISTORICAL_PROVIDER), region=REG_CN, kernels=1)
    last = pd.DatetimeIndex(D.calendar(freq="day")).max()
    expressions = list(FEATURE_EXPRESSIONS.values()) + ["$close / ($factor + 1e-12)"]
    features = D.features(D.instruments("csi300"), expressions, start_time="2026-08-01", end_time=str(last.date()), freq="day")
    features.columns = list(FEATURE_EXPRESSIONS) + ["raw_close"]
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.loc[~features[list(FEATURE_EXPRESSIONS)].isna().all(axis=1)].reset_index()
    financials = disclosed_financials(STATE_DIR / DATE, pd.Timestamp(DATE))
    features = merge_latest_event(features, prepare_general_report_events(financials), "fundamental_available_date")
    features = merge_latest_event(features, prepare_annual_eps_events(financials), "annual_eps_available_date")
    validate_point_in_time(features)
    features["book_to_price"] = features.book_value_per_share / features.raw_close
    features["earnings_to_price"] = features.annual_eps / features.raw_close
    features = features.set_index(["instrument", "datetime"]).replace([np.inf, -np.inf], np.nan).sort_index()
    ranked = cross_sectional_rank(features, ENHANCED_FEATURE_NAMES).astype("float32")
    model_path = MODEL_DIR / "model.txt"
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Deployed model changed")
    model = lgb.Booster(model_file=str(model_path))
    new = pd.DataFrame({"prediction": model.predict(ranked[ENHANCED_FEATURE_NAMES], num_threads=4)}, index=ranked.index)
    latest = pd.read_csv(STATE_DIR / DATE / "predictions.csv", parse_dates=["datetime"]).set_index(["instrument", "datetime"])[["prediction"]]
    old = pd.read_parquet(ROOT / "reports/experiments/main_until_2026_07/backtest_predictions.parquet")[["prediction"]]
    combined = pd.concat([old, new, latest]).sort_index()
    if combined.index.duplicated().any() or not np.isfinite(combined.prediction).all():
        raise ValueError("Extended predictions invalid")
    combined.to_parquet(REPORT / "predictions.parquet")
    codes = sorted(set(combined.index.get_level_values("instrument")))
    return combined, codes


def clone_quotes_and_append_today(codes):
    # Quote storage is isolated from both the original research and live signal data.
    PROVIDER.mkdir(parents=True, exist_ok=True)
    for directory in ["calendars", "instruments"]:
        if not (PROVIDER / directory).exists():
            shutil.copytree(HISTORICAL_PROVIDER / directory, PROVIDER / directory)
    for code in sorted(set(codes + ["SH000300"])):
        target = PROVIDER / "features" / code.lower()
        if not target.exists():
            shutil.copytree(HISTORICAL_PROVIDER / "features" / code.lower(), target)
    calendar_path = PROVIDER / "calendars/day.txt"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    date = pd.Timestamp(DATE)
    if date not in calendar:
        calendar = calendar.append(pd.DatetimeIndex([date]))
        calendar_path.write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    position = calendar.get_loc(date)
    bars = pd.read_parquet(STATE_DIR / DATE / "bars.parquet")
    snapshots = pd.read_csv(STATE_DIR / DATE / "snapshots.csv").set_index("code")
    for (instrument, _), row in bars.iterrows():
        raw = snapshots.loc[instrument[:2] + "." + instrument[2:]]
        values = {**row.to_dict(), "change": float(raw.last_price / raw.prev_close_price - 1), "amount": float(raw.turnover)}
        for field, value in values.items():
            storage = FileFeatureStorage(instrument=instrument, field=field, freq="day", provider_uri=str(PROVIDER))
            if storage.end_index is None or storage.end_index < position:
                storage.write([value], index=position)
    # Preserve units of the index as well, rather than append raw points to normalized prices.
    benchmark = pd.read_csv(REPORT / "benchmark_snapshot.csv").iloc[0]
    storage = FileFeatureStorage("SH000300", "close", "day", provider_uri=str(PROVIDER))
    _, previous_close = storage[position - 1]
    factor = float(previous_close / benchmark.prev_close_price)
    values = {"open": benchmark.open_price * factor, "close": benchmark.last_price * factor,
              "high": benchmark.high_price * factor, "low": benchmark.low_price * factor,
              "factor": factor, "volume": benchmark.volume / factor,
              "change": benchmark.last_price / benchmark.prev_close_price - 1}
    for field, value in values.items():
        storage = FileFeatureStorage("SH000300", field, "day", provider_uri=str(PROVIDER))
        if storage.end_index is None or storage.end_index < position:
            storage.write([float(value)], index=position)
    print(f"Isolated quote provider ready: {len(codes)} stocks through {DATE}", flush=True)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    predictions, codes = build_predictions()
    clone_quotes_and_append_today(codes)
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    config = {**load_research_config(), **load_main_strategy_config()}
    print("Replay the existing continuous account through the latest completed session", flush=True)
    raw = run_tiered_backtest(predictions, config, rebalance_every=10, period_end=DATE, quote_codes=codes)
    report = enrich_daily_report(raw, "main_10d_latest_30_sessions")
    report.to_csv(REPORT / "full_daily_replay.csv", encoding="utf-8-sig")
    window = report.tail(30).copy()
    if len(window) != 30 or window.index.max() != pd.Timestamp(DATE):
        raise ValueError("Backtest does not cover the requested 30 completed sessions")
    portfolio_wealth = float((1 + window.net_return).prod())
    benchmark_wealth = float((1 + window.bench).prod())
    if not np.isfinite(window[["net_return", "bench", "cost"]].to_numpy()).all():
        raise ValueError("Return series contains invalid values")
    summary = {"start": str(window.index.min().date()), "end": str(window.index.max().date()),
               "trading_days": len(window), "strategy_net_return": portfolio_wealth - 1,
               "benchmark_return": benchmark_wealth - 1,
               "geometric_excess_return": portfolio_wealth / benchmark_wealth - 1,
               "arithmetic_excess_percentage_points": (portfolio_wealth - benchmark_wealth) * 100,
               "max_drawdown": calculate_max_drawdown(window.net_return),
               "positive_days": int(window.net_return.gt(0).sum()),
               "initial_cash": config["initial_cash"], "label_horizon": 10, "rebalance_every": 10,
               "model_train_end": "2025-12-31", "selected_trees": 160,
               "buy_cost": config["buy_cost"], "sell_cost": config["sell_cost"], "minimum_cost": config["minimum_cost"],
               "continuous_account_since": str(report.index.min().date()),
               "not_live_paper_results": True,
               "not_reset_at_30_day_window_start": True}
    # Cross-check the window compound return with the reconstructed account NAV.
    before = report.loc[report.index < window.index.min(), "portfolio_nav"].iloc[-1]
    if not np.isclose(report.portfolio_nav.iloc[-1] / before, portfolio_wealth, atol=1e-12):
        raise ValueError("Window return does not agree with continuous account NAV")
    window.to_csv(REPORT / "last_30_daily.csv", encoding="utf-8-sig")
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
