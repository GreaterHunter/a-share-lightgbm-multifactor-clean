"""Combine verified Qlib history with Futu closing bars and disclosed financials."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
import requests

from paper_futu.cn_common import HISTORICAL_PROVIDER, ROOT, STATE_DIR, load_deployment, to_futu_code, validate_serving_date, china_now
from paper_futu.cn_deployment import activate_pending, model_for_signal
from src.forward_labels import label_contract, LEGACY_OPEN_TO_CLOSE
from src.build_enhanced_dataset import merge_latest_event, prepare_annual_eps_events, prepare_general_report_events, validate_point_in_time
from src.build_ranked_dataset import cross_sectional_rank
from src.factor_library import ENHANCED_FEATURE_NAMES, FEATURE_EXPRESSIONS


def calculate_price_features(prices):
    """Use the same shifts, rolling windows and sample standard deviations as Qlib."""
    def one_stock(frame):
        close, opening, high, low, volume = (frame[name] for name in ["close", "open", "high", "low", "volume"])
        returns = close / close.shift(1) - 1
        volume_returns = volume / volume.shift(1) - 1
        result = pd.DataFrame(index=frame.index)
        result["return_1d"] = returns
        for window in [5, 20, 60]:
            result[f"mom_{window}d"] = close / close.shift(window) - 1
        for window in [20, 60]:
            result[f"volatility_{window}d"] = returns.rolling(window, min_periods=1).std()
        result["intraday_return"] = close / opening - 1
        result["overnight_gap"] = opening / close.shift(1) - 1
        result["amplitude_20d"] = ((high - low) / close).rolling(20, min_periods=1).mean()
        result["volume_ratio_5_20"] = volume.rolling(5, min_periods=1).mean() / volume.rolling(20, min_periods=1).mean() - 1
        floor = low.rolling(20, min_periods=1).min()
        result["price_position_20d"] = (close - floor) / (high.rolling(20, min_periods=1).max() - floor + 1e-12)
        result["price_volume_corr_20d"] = returns.rolling(20, min_periods=2).corr(volume_returns)
        return result
    return prices.groupby(level="instrument", group_keys=False).apply(one_stock).replace([np.inf, -np.inf], np.nan)


def adjusted_bar(previous, snapshot, actions, signal_date):
    """Preserve Qlib adjusted-volume units (hands/factor), not Futu shares."""
    old_raw = previous["close"] / previous["factor"]
    theoretical_previous = old_raw
    events = actions.loc[
        actions["code"].eq(snapshot["code"])
        & pd.to_datetime(actions["ex_div_date"]).gt(previous.name[1])
        & pd.to_datetime(actions["ex_div_date"]).le(signal_date)
    ].sort_values("ex_div_date")
    for row in events.itertuples():
        theoretical_previous = theoretical_previous * row.forward_adj_factorA + row.forward_adj_factorB
    if not np.isfinite(theoretical_previous) or theoretical_previous <= 0:
        raise ValueError("Corporate action creates an invalid reference price")
    observed_reference = float(snapshot["prev_close_price"])
    if abs(theoretical_previous - observed_reference) > max(0.025, 0.001 * theoretical_previous):
        raise ValueError(f"Previous price mismatch: {snapshot['code']}, history {theoretical_previous}, Futu {observed_reference}")
    factor = float(previous["close"] / theoretical_previous)
    valid = not bool(snapshot["suspension"])
    return {
        "open": float(snapshot["open_price"]) * factor if valid else np.nan,
        "close": float(snapshot["last_price"]) * factor if valid else np.nan,
        "high": float(snapshot["high_price"]) * factor if valid else np.nan,
        "low": float(snapshot["low_price"]) * factor if valid else np.nan,
        "volume": float(snapshot["volume"]) / 100 / factor if valid else np.nan,
        "factor": factor,
    }


def read_saved_bars(path):
    """Read both pre-fix shares/factor bars and corrected hands/factor bars."""
    bars = pd.read_parquet(path)
    marker = path.with_name("bars_units.json")
    if marker.exists():
        units = json.loads(marker.read_text(encoding="utf-8"))
        if units.get("volume") != "hands_divided_by_adjustment_factor":
            raise ValueError(f"Unknown saved-bar volume unit: {path}")
    else:
        # Existing 2026-10-12/13 snapshots were written before the unit fix.
        if path.parent.name not in {"2026-10-12", "2026-10-13"}:
            raise ValueError(f"Unmarked saved-bar volume unit: {path}")
        bars = bars.copy()
        bars["volume"] /= 100
    return bars


def disclosed_financials(directory, signal_date):
    values = pd.read_parquet(ROOT / "data/external/fundamentals.parquet")
    base = pd.read_parquet(ROOT / "data/processed/main_until_2026_07/disclosures_through_2026.parquet")
    lookup = {code[-6:]: code for code in values.instrument.unique()}
    parts = [base]
    source = "https://www.cninfo.com.cn/new/information/getPrbookInfo"
    for period in ["2026-03-31", "2026-06-30"]:
        cache = directory / f"disclosures_{period}.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = requests.post(source, params={
                "sectionTime": period, "firstTime": "", "lastTime": "", "market": "szsh",
                "stockCode": "", "orderClos": "", "isDesc": "", "pagesize": "10000", "pagenum": "1",
            }, timeout=45)
            response.raise_for_status()
            payload = response.json()
            cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if payload.get("hasNextPage") or not payload.get("prbookinfos"):
            raise ValueError("Incomplete CNINFO actual disclosure response")
        raw = pd.DataFrame(payload["prbookinfos"])
        parts.append(pd.DataFrame({
            "instrument": raw.seccode.astype(str).str.zfill(6).map(lookup),
            "report_date": pd.to_datetime(raw.f001d_0102),
            "disclosure_date": pd.to_datetime(raw.f006d_0102, errors="coerce"),
        }).dropna(subset=["instrument", "report_date"]))
    dates = pd.concat(parts, ignore_index=True).drop_duplicates(["instrument", "report_date"], keep="last")
    financials = values.merge(dates[["instrument", "report_date", "disclosure_date"]],
                              on=["instrument", "report_date"], how="inner", validate="one_to_one")
    financials["available_date"] = pd.to_datetime(financials.disclosure_date) + pd.Timedelta(days=1)
    return financials.loc[financials.available_date.le(signal_date)].copy()


def model_inputs(panel, metadata):
    """Feature-name equality alone does not distinguish raw from ranked models."""
    kind = metadata.get("input_kind", "rank")
    if kind == "raw":
        return panel[ENHANCED_FEATURE_NAMES].astype("float32")
    if kind == "rank":
        return cross_sectional_rank(panel, ENHANCED_FEATURE_NAMES).astype("float32")
    raise ValueError(f"Unsupported deployed input transformation: {kind}")


def model_financials(directory, date, metadata):
    source = metadata.get("financial_events_path")
    if not source:
        return disclosed_financials(directory, date)
    path = (ROOT / source).resolve()
    if not path.is_relative_to(ROOT.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["financial_events_sha256"]:
        raise ValueError("Verified deployment financial snapshot changed")
    events = pd.read_parquet(path)
    return events.loc[events.available_date.le(date)].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--daily-only", action="store_true", help="Update daily features without replacing the scheduled rebalance plan")
    parser.add_argument("--model-dir", help="Explicit model for an isolated, no-order readiness check")
    parser.add_argument("--output-dir", help="Separate readiness output; required with --model-dir")
    parser.add_argument("--prebuilt-bars", type=Path, help="Validated independent market package; never uses OpenD")
    args = parser.parse_args()
    date = pd.Timestamp(args.date)
    directory = STATE_DIR / args.date
    if args.prebuilt_bars:
        from paper_futu.independent_cn_market import sha
        directory = args.prebuilt_bars.resolve().parent
        manifest = json.loads((directory / "market_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status") != "validated" or manifest.get("validation_schema") != 2 or manifest.get("date") != args.date
                or args.prebuilt_bars.name != "bars.parquet"):
            raise ValueError("Independent daily market package is not validated")
        for name in ["bars.parquet", "bars_units.json", "members.csv", "snapshots.csv"]:
            if sha(directory / name) != manifest["artifacts"][name]:
                raise ValueError(f"Independent market artifact changed: {name}")
    output = directory
    if bool(args.model_dir) != bool(args.output_dir) or (args.model_dir and not args.daily_only):
        raise ValueError("Explicit model checks require --daily-only and a separate --output-dir")
    members = pd.read_csv(directory / "members.csv")
    snapshots = pd.read_csv(directory / "snapshots.csv").set_index("code", drop=False)
    actions = (pd.DataFrame() if args.prebuilt_bars else pd.read_csv(directory / "corporate_actions.csv"))
    if args.model_dir:
        model_dir = (ROOT / args.model_dir).resolve()
        output = (ROOT / args.output_dir).resolve()
        if (not model_dir.is_relative_to((ROOT / "models").resolve())
                or not output.is_relative_to((ROOT / "reports/paper_futu").resolve())):
            raise ValueError("Isolated readiness paths must stay under models and reports/paper_futu")
        output.mkdir(parents=True, exist_ok=True)
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    else:
        model_dir, metadata = model_for_signal(args.date)
    validate_serving_date(metadata, args.date)
    portfolio = metadata["portfolio"]
    if date.year != metadata["model_year"]:
        raise ValueError("Annual model is not valid for the signal year")
    path = model_dir / "model.txt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model hash changed")
    qlib.init(provider_uri=str(HISTORICAL_PROVIDER), region=REG_CN, kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    historical_end = calendar.max()
    start = date - pd.Timedelta(days=240)
    prices = D.features(members.instrument.tolist(),
                        fields=["$open", "$close", "$high", "$low", "$volume", "$factor"],
                        start_time=str(start.date()), end_time=str(historical_end.date()), freq="day")
    prices.columns = ["open", "close", "high", "low", "volume", "factor"]
    prices = prices.sort_index()
    from paper_futu.cn_daily_history import prior_saved_parts,merge_history
    prices=merge_history([prices,*prior_saved_parts(args.date)])
    if args.prebuilt_bars:
        bars = read_saved_bars(args.prebuilt_bars)
        if (set(bars.index.get_level_values("datetime")) != {date}
                or set(bars.index.get_level_values("instrument")) != set(members.instrument)
                or bars.index.has_duplicates):
            raise ValueError("Independent bar dates or stock universe differ")
        prices = pd.concat([prices, bars]).sort_index()
    if prices.index.duplicated().any():
        raise ValueError("Historical/live bars overlap")
    # Some community-provider symbols stop early; fill only verified missing dates
    # from Futu raw daily bars, preserving the last available Qlib price units.
    gaps = {}
    for instrument in members.instrument:
        stock = prices.xs(instrument, level="instrument", drop_level=False)
        if stock.index.get_level_values("datetime").max() < historical_end and not bool(snapshots.loc[to_futu_code(instrument)].suspension):
            gaps[to_futu_code(instrument)] = str(stock.index.get_level_values("datetime").max().date())
    if gaps:
        if args.prebuilt_bars:
            raise ValueError("Independent market package has unresolved historical gaps; no OpenD fallback")
        (output / "history_gaps.json").write_text(json.dumps(gaps, indent=2), encoding="utf-8")
        if any(not (directory / "history" / f"{code}.csv").exists() for code in gaps):
            subprocess.run([str(ROOT / "paper_futu/.venv/Scripts/python.exe"), "-m", "paper_futu.fetch_cn_market",
                            "--date", args.date, "--repair-history"], cwd=ROOT, check=True, timeout=300)
        repairs = []
        for code, last in gaps.items():
            instrument = code.replace(".", "")
            previous = prices.xs(instrument, level="instrument", drop_level=False).iloc[-1]
            history = pd.read_csv(directory / "history" / f"{code}.csv")
            history["date"] = pd.to_datetime(history.time_key).dt.normalize()
            history = history.loc[history.date.gt(pd.Timestamp(last)) & history.date.le(historical_end)].sort_values("date")
            for item in history.itertuples():
                snapshot = {"code": code, "suspension": False, "prev_close_price": item.last_close,
                            "open_price": item.open, "last_price": item.close, "high_price": item.high,
                            "low_price": item.low, "volume": item.volume}
                values = adjusted_bar(previous, snapshot, actions, item.date)
                repairs.append({"instrument": instrument, "datetime": item.date, **values})
                previous = pd.Series(values, name=(instrument, item.date))
        repair_frame = pd.DataFrame(repairs).set_index(["instrument", "datetime"])
        prices = pd.concat([prices, repair_frame]).sort_index()
        repair_frame.to_parquet(output / "history_repairs.parquet")
    if date > prices.index.get_level_values("datetime").max():
        future_calendar = pd.DatetimeIndex(pd.to_datetime((HISTORICAL_PROVIDER / "calendars/day_future.txt").read_text().splitlines()))
        all_dates = calendar.union(future_calendar).sort_values()
        previous_day = all_dates[all_dates < date].max()
        if prices.index.get_level_values("datetime").max() != previous_day:
            raise ValueError("Live price history has missing trading days; refresh the provider first")
        new_rows = []
        for instrument in members.instrument:
            stock = prices.xs(instrument, level="instrument", drop_level=False)
            available = stock.loc[stock.index.get_level_values("datetime") == previous_day]
            snapshot = snapshots.loc[to_futu_code(instrument)]
            previous = stock.iloc[-1] if available.empty and bool(snapshot.suspension) else available.iloc[0]
            if not bool(snapshot.suspension) and not np.isfinite(previous["close"]):
                previous = stock.loc[stock.close.notna() & stock.factor.notna()].iloc[-1]
            if not np.isfinite(previous["factor"]) or previous["factor"] <= 0:
                raise ValueError(f"Missing previous factor: {instrument}")
            stamp = pd.Timestamp(snapshot.update_time)
            if not bool(snapshot.suspension) and (stamp.date() != date.date() or stamp.hour < 15):
                raise ValueError(f"Closing quote is stale or incomplete: {instrument}, {stamp}")
            values = ({name: np.nan for name in ["open", "close", "high", "low", "volume"]}
                      | {"factor": float(previous["factor"])}) if bool(snapshot.suspension) else adjusted_bar(previous, snapshot, actions, date)
            new_rows.append({"instrument": instrument, "datetime": date, **values})
        bars = pd.DataFrame(new_rows).set_index(["instrument", "datetime"]).sort_index()
        (output / "bars_units.json").write_text(
            json.dumps({"volume": "hands_divided_by_adjustment_factor"}, indent=2), encoding="utf-8"
        )
        bars.to_parquet(output / "bars.parquet")
        prices = pd.concat([prices, bars]).sort_index()
    else:
        all_dates = calendar.union(pd.DatetimeIndex(pd.to_datetime((HISTORICAL_PROVIDER / "calendars/day_future.txt").read_text().splitlines()))).sort_values()
    features = calculate_price_features(prices)
    # Check the live implementation against the project's Qlib expressions on a known day.
    check_date = historical_end
    reference = D.features(members.instrument.tolist(), list(FEATURE_EXPRESSIONS.values()),
                           start_time=str(check_date.date()), end_time=str(check_date.date()), freq="day")
    reference.columns = list(FEATURE_EXPRESSIONS)
    actual = features.reindex(reference.index)[list(FEATURE_EXPRESSIONS)]
    differences = (actual - reference).abs().max()
    if (differences.dropna() > 1e-5).any():
        raise ValueError(f"Live feature formulas differ from Qlib: {differences.to_dict()}")
    panel = features.loc[features.index.get_level_values("datetime") == date].copy()
    panel["raw_close"] = prices.loc[panel.index, "close"] / prices.loc[panel.index, "factor"]
    panel = panel.reset_index()
    financials = model_financials(directory, date, metadata)
    panel = merge_latest_event(panel, prepare_general_report_events(financials), "fundamental_available_date")
    panel = merge_latest_event(panel, prepare_annual_eps_events(financials), "annual_eps_available_date")
    validate_point_in_time(panel)
    panel["book_to_price"] = panel.book_value_per_share / panel.raw_close
    panel["earnings_to_price"] = panel.annual_eps / panel.raw_close
    panel = panel.set_index(["instrument", "datetime"]).replace([np.inf, -np.inf], np.nan).sort_index()
    suspended = [code.replace(".", "") for code in snapshots.loc[snapshots.suspension, "code"]]
    panel = panel.loc[~panel.index.get_level_values("instrument").isin(suspended)].copy()
    coverage = panel[ENHANCED_FEATURE_NAMES].notna().mean()
    if (coverage < 0.90).any():
        raise ValueError(f"Feature coverage is too low: {coverage.to_dict()}")
    ranked = cross_sectional_rank(panel, ENHANCED_FEATURE_NAMES).astype("float32")
    inputs = model_inputs(panel, metadata)
    booster = lgb.Booster(model_file=str(path))
    if booster.feature_name() != ENHANCED_FEATURE_NAMES:
        raise ValueError("Model feature order mismatch")
    result = pd.DataFrame({"prediction": booster.predict(inputs, num_threads=4)}, index=inputs.index)
    result["code"] = result.index.get_level_values("instrument").map(to_futu_code)
    result["suspension"] = result.code.map(snapshots.suspension)
    result["name"] = result.code.map(snapshots["name"])
    result["raw_close"] = panel.raw_close
    result["eligible"] = ~result.suspension & ~result.name.str.contains("ST", na=False)
    result = result.sort_values(["prediction", "code"], ascending=[False, True])
    panel.to_parquet(output / "features_raw.parquet")
    ranked.to_parquet(output / "features_ranked.parquet")
    result.to_csv(output / "predictions.csv", encoding="utf-8-sig")
    candidates = result.loc[result.eligible]
    if len(candidates) < portfolio["top_k"] or not np.isfinite(result.prediction).all():
        raise ValueError("Not enough valid prediction candidates")
    plan = {
        **{name: metadata.get(name, label_contract(metadata["label_horizon"], LEGACY_OPEN_TO_CLOSE).get(name))
           for name in ["label_price_basis", "label_expression", "target"]},
        "execution_price_basis": "next_session_open",
        "signal_date": args.date,
        "execution_date": str(all_dates[all_dates > date].min().date()),
        "next_rebalance_signal_date": str(all_dates[all_dates >= date][portfolio["rebalance_every"]].date()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": metadata["model_sha256"], "model_year": metadata["model_year"],
        "model_dir": model_dir.relative_to(ROOT.resolve()).as_posix(),
        "ordered_candidates": candidates.code.tolist(),
        "predictions": {row.code: float(row.prediction) for row in candidates.itertuples()},
        **{name: portfolio[name] for name in ["top_k", "n_drop", "top_tier_size", "top_tier_weight", "risk_degree", "rebalance_every"]},
        "label_horizon": metadata["label_horizon"],
        "features": ENHANCED_FEATURE_NAMES, "feature_coverage": coverage.to_dict(),
        "reference_prices": {row.code: float(row.raw_close) for row in result.itertuples()},
        **{name: metadata[name] for name in ["input_kind", "retrain_months", "information_cutoff", "selection_rule"] if name in metadata},
        "qlib_formula_max_differences": differences.to_dict(), "stage": "ready_for_next_session",
    }
    if args.daily_only:
        observed_now=china_now()
        plan.update(data_only_prediction=True,not_an_execution_authorization=True,
                    retrospectively_generated=observed_now.date()>date.date(),
                    stage='daily_inference_only')
    if args.prebuilt_bars:
        plan.update(market_data_source="independent_public_daily",
                    market_manifest_sha256=sha(directory / "market_manifest.json"),
                    market_validation_schema=manifest["validation_schema"])
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if not args.daily_only:
        if not activate_pending(plan):
            (STATE_DIR / "active_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Signals ready for {plan['execution_date']}: {len(result)} active-stock predictions, {plan['top_k']} initial holdings, n_drop={plan['n_drop']}", flush=True)
    print(result.head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
