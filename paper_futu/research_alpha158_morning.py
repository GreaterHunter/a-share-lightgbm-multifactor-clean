"""Independent Alpha158 morning forecast research; never submits orders/deploys.

Today means the observation session S. Complete Alpha158 inputs end at S-1,
only S's confirmed auction open is added. Daily opening returns are prediction
references; executable strategy returns require post-observation intraday fills.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.storage.file_storage import FileFeatureStorage
import requests

from paper_futu.cn_common import ROOT, STATE_DIR, load_deployment
from src.alpha158_morning import feature_names, morning_end_dates, morning_registry, normalize_local_bar, specification
from src.build_enhanced_dataset import merge_latest_event, prepare_annual_eps_events, prepare_general_report_events, validate_point_in_time
from src.build_ranked_dataset import cross_sectional_rank
from src.config import load_model_config
from src.forward_labels import select_mature_period
from src.morning_observations import build_morning_observation
from src.runtime_budget import require_memory

SOURCE = ROOT / "data/processed/cn_latest_mature_open10_20261012"
HORIZON = 10
CANDIDATES = [10, 20, 40, 80, 160, 320, 640]


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def paths_for(as_of):
    name = f"morning_alpha158_{as_of:%Y%m%d}"
    return {"data": ROOT / "data/processed" / name, "report": ROOT / "reports/experiments" / name,
            "model": ROOT / "models/experiments" / name}


def protected_state():
    directory, _ = load_deployment()
    config_path = ROOT / "configs/futu_cn_paper.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = [config_path, STATE_DIR / "active_plan.json", STATE_DIR / "catchup_plan.json",
             directory / "model.txt", directory / "metadata.json"]
    if config.get("pending_deployment"):
        candidate = ROOT / config["pending_deployment"]["model_dir"]
        paths += [candidate / "model.txt", candidate / "metadata.json"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths if p.exists()}


def verify_historical_units(directory, historical_end):
    """Spot-check three stocks with Tencent's overlapping volume-in-hands data."""
    checks = []
    for code in ["SH600000", "SH600741", "SZ300316"]:
        cache = directory / f"unit_source_{code}.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = requests.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                params={"param": f"{code.lower()},day,{historical_end},{historical_end},2,qfq"}, timeout=15)
            response.raise_for_status()
            payload = response.json()
            save_json(cache, payload)
        data = payload["data"][code.lower()]
        bars = data.get("qfqday", data.get("day", []))
        row = next(bar for bar in bars if bar[0] == historical_end)
        history = D.features([code], ["$volume", "$factor", "$amount", "$vwap"],
                             start_time=historical_end, end_time=historical_end).iloc[0]
        hands = float(row[5])
        observed_hands = float(history["$volume"] * history["$factor"])
        vwap_multiplier = float(history["$vwap"] * history["$volume"] / history["$amount"])
        if not np.isclose(hands, observed_hands, rtol=2e-5) or not np.isclose(vwap_multiplier, 10, rtol=2e-5):
            raise ValueError(f"Historical units are not the verified hands/1000-CNY convention: {code}")
        checks.append({"instrument": code, "date": historical_end, "tencent_volume_hands": hands,
                       "qlib_volume_times_factor": observed_hands, "vwap_amount_volume_multiplier": vwap_multiplier,
                       "source_sha256": digest(cache)})
    return checks


def write_feature_value(provider, instrument, name, position, value):
    """Update only the independent provider's float32 Qlib file at one session.

    Qlib 0.9.7's FileFeatureStorage rewrite branch passes a float32 header into
    range(), which raises TypeError. Its append branch is usable unchanged.
    """
    storage = FileFeatureStorage(instrument, name, "day", provider_uri=str(provider))
    if storage.start_index is None or position > storage.end_index:
        storage.write([float(value)], index=position)
        return
    if position < storage.start_index:
        raise ValueError("Cannot insert before the independent provider's first session")
    with storage.uri.open("rb+") as stream:
        stream.seek(4 * (1 + position - storage.start_index))
        np.asarray([value], dtype="<f4").tofile(stream)


def prepare_provider(paths, as_of):
    provider = paths["data"] / "provider"
    ready_path = provider / "morning_snapshot_ready.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready["as_of"] != str(as_of.date()):
            raise ValueError("Wrong cached snapshot cutoff")
        for row in ready["local_inputs"]:
            if digest(ROOT / row["bar_path"]) != row["bar_sha256"] or digest(ROOT / row["snapshot_path"]) != row["snapshot_sha256"]:
                raise ValueError("Local source changed since snapshot creation")
        return provider
    # Copies, NOT hard links: repair only this experiment, never frozen/deployed inputs.
    shutil.copytree(SOURCE / "provider", provider, dirs_exist_ok=True)
    source_ready = json.loads((provider / "snapshot_ready.json").read_text(encoding="utf-8"))
    historical_end = source_ready["historical_end"][:10]
    observed = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines()))
    future = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day_future.txt").read_text().splitlines()))
    calendar = observed.union(future)
    calendar = calendar[calendar <= as_of].sort_values()
    if calendar[-1] != as_of:
        raise ValueError("Cutoff is not an exchange session")
    local_dates = calendar[calendar > pd.Timestamp(historical_end)]
    for date in local_dates:
        directory = STATE_DIR / str(date.date())
        if not (directory / "bars.parquet").exists() or not (directory / "snapshots.csv").exists():
            raise ValueError(f"Verified local source is missing for {date.date()}")
    (provider / "calendars/day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1)
    units = verify_historical_units(paths["data"], historical_end)
    local_inputs = []
    membership_path = provider / "instruments/csi300.txt"
    membership = pd.read_csv(membership_path, sep="\t", header=None, names=["instrument", "start", "end"])
    for date in local_dates:
        directory = STATE_DIR / str(date.date())
        bars = pd.read_parquet(directory / "bars.parquet")
        quotes = pd.read_csv(directory / "snapshots.csv").set_index("code")
        if bars.index.has_duplicates or quotes.index.has_duplicates:
            raise ValueError("Duplicate local quote or bar")
        for (instrument, stamp), bar in bars.iterrows():
            if pd.Timestamp(stamp) != date:
                raise ValueError("Local bar date mismatch")
            quote = quotes.loc[instrument[:2] + "." + instrument[2:]].to_dict()
            values = normalize_local_bar(quote, float(bar.factor))
            if not bool(quote["suspension"]) and not np.isclose(values["open"], bar.open, rtol=1e-5):
                raise ValueError("Adjusted opening price unit mismatch")
            for name, value in values.items():
                write_feature_value(provider, instrument, name, int(calendar.get_loc(date)), value)
        members = pd.read_csv(directory / "members.csv")
        if members.instrument.nunique() != 300:
            raise ValueError("Incomplete local CSI300 membership")
        previous = calendar[calendar < date][-1]
        for instrument in members.instrument:
            rows = membership.index[membership.instrument.eq(instrument)]
            if len(rows) and pd.Timestamp(membership.at[rows[-1], "end"]) == previous:
                membership.at[rows[-1], "end"] = str(date.date())
            elif not len(rows) or pd.Timestamp(membership.at[rows[-1], "end"]) < date:
                membership.loc[len(membership)] = [instrument, str(date.date()), str(date.date())]
        local_inputs.append({"date": str(date.date()), "bar_path": (directory / "bars.parquet").relative_to(ROOT).as_posix(),
                             "bar_sha256": digest(directory / "bars.parquet"),
                             "snapshot_path": (directory / "snapshots.csv").relative_to(ROOT).as_posix(),
                             "snapshot_sha256": digest(directory / "snapshots.csv")})
    membership.to_csv(membership_path, sep="\t", index=False, header=False)
    save_json(ready_path, {"as_of": str(as_of.date()), "historical_end": historical_end,
                          "source_provider": (SOURCE / "provider").relative_to(ROOT).as_posix(),
                          "local_inputs": local_inputs, "historical_unit_spot_checks": units,
                          "volume_conversion": "Futu shares / 100 / adjusted factor", "amount_conversion": "Futu CNY / 1000",
                          "local_vwap": "Futu turnover / shares * adjusted factor"})
    return provider


def build_chunk(paths, as_of, start, end, sample_symbols=None, sample=False):
    provider = prepare_provider(paths, as_of)
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    market = D.instruments("csi300")
    if sample_symbols:
        market = D.list_instruments(market, start_time=start, end_time=end, as_list=True)[:sample_symbols]
    registry = morning_registry()
    helper = ["Ref($close / ($factor + 1e-12), 1)", "$open"]
    spec = specification(HORIZON)
    frame = D.features(market, list(registry.values()) + helper + [spec["label_expression"]],
                       start_time=str(start.date()), end_time=str(end.date()), freq="day")
    frame.columns = list(registry) + ["previous_raw_close", "observed_adjusted_open", spec["raw_label"]]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.loc[frame.observed_adjusted_open.gt(0)].copy()
    financials = pd.read_parquet(SOURCE / "financial_events.parquet")
    panel = merge_latest_event(frame.reset_index(), prepare_general_report_events(financials), "fundamental_available_date")
    panel = merge_latest_event(panel, prepare_annual_eps_events(financials), "annual_eps_available_date")
    validate_point_in_time(panel)
    price = panel.previous_raw_close.where(panel.previous_raw_close > 0)
    panel["book_to_price"] = panel.book_value_per_share / price
    panel["earnings_to_price"] = panel.annual_eps / price
    panel = panel.set_index(["instrument", "datetime"]).sort_index().replace([np.inf, -np.inf], np.nan)
    names = feature_names()
    raw = panel[names].copy()
    if raw.index.has_duplicates or raw[names].isna().all(axis=1).any():
        raise ValueError("Invalid stock-session feature rows")
    ranked = cross_sectional_rank(raw, names).astype("float32")
    ranked["label_end_date"] = morning_end_dates(ranked.index, calendar, HORIZON)
    ranked[spec["raw_label"]] = panel[spec["raw_label"]].where(ranked.label_end_date.le(as_of)).astype("float32")
    ranked[spec["target"]] = cross_sectional_rank(ranked, [spec["raw_label"]])[spec["raw_label"]].astype("float32")
    if not np.isfinite(ranked[names].stack().to_numpy()).all():
        raise ValueError("Infinite ranked feature")
    output = paths["data"] / ("sample.parquet" if sample else
                              f"annual_shards/{start:%Y%m%d}_{end:%Y%m%d}.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(output)
    if sample:
        raw.to_parquet(paths["data"] / "sample_raw_features.parquet")
    audit = {"file": output.relative_to(ROOT).as_posix(), "sha256": digest(output), "rows": len(ranked),
             "start": str(start.date()), "end": str(end.date()), "feature_count": len(names),
             "symbols": ranked.index.get_level_values("instrument").nunique(),
             "sample_only_not_for_selection": sample,
             "feature_missing_ratios": raw.isna().mean().to_dict(),
             "latest_label_end_date": (str(ranked.label_end_date.dropna().max().date())
                                       if ranked.label_end_date.notna().any() else None),
             "unmatured_rows_preserved_for_inference": int(
                 ranked.label_end_date.isna().sum() + ranked.label_end_date.gt(as_of).sum()),
             "mature_rows_with_missing_return": int(
                 (ranked.label_end_date.le(as_of) & ranked[spec["raw_label"]].isna()).sum())}
    save_json(output.with_suffix(".json"), audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "feature_missing_ratios"}, ensure_ascii=False), flush=True)
    return audit


def audit_morning_reconstruction(paths, as_of):
    """Historical replay check: no S-day full-bar field is needed for raw X."""
    provider = paths["data"] / "provider"
    calendar = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines()))
    previous = calendar[calendar < as_of][-1]
    expected = pd.read_parquet(paths["data"] / "sample_raw_features.parquet")
    expected = expected.loc[expected.index.get_level_values("datetime") == as_of]
    members = expected.index.get_level_values("instrument").tolist()
    quotes = pd.read_csv(STATE_DIR / str(as_of.date()) / "snapshots.csv")
    observed, _ = build_morning_observation(
        str(provider), previous, as_of, members, quotes,
        as_of + pd.Timedelta(hours=23, minutes=59),
        pd.read_parquet(SOURCE / "financial_events.parquet"),
    )
    left, right = observed.sort_index(), expected.sort_index()
    if not left.index.equals(right.index) or left.columns.tolist() != right.columns.tolist():
        raise ValueError("Morning replay and historical factors have different rows or columns")
    difference = (left - right).abs().to_numpy(dtype="float64")
    maximum = float(np.nanmax(difference))
    if not np.allclose(left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64"),
                       atol=1e-6, rtol=1e-6, equal_nan=True):
        raise ValueError(f"Yesterday + confirmed open did not reproduce historical factors: {maximum}")
    result = {"historical_replay_only": True, "session": str(as_of.date()),
              "stock_count": len(observed), "feature_count": len(observed.columns),
              "maximum_absolute_difference": maximum,
              "no_current_session_close_high_low_or_daily_volume_input": True}
    save_json(paths["report"] / "morning_reconstruction_audit.json", result)
    return result


def build_all_data(paths, as_of):
    require_memory(2.0, "完整Alpha158数据构建")
    prepare_provider(paths, as_of)
    records = []
    for year in range(2010, as_of.year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        end = min(pd.Timestamp(year=year, month=12, day=31), as_of)
        output = paths["data"] / f"annual_shards/{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        if output.exists() and output.with_suffix(".json").exists():
            record = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            if digest(output) != record["sha256"]:
                raise ValueError("Cached partition hash mismatch")
        else:
            command = [sys.executable, "-m", "paper_futu.research_alpha158_morning", "--as-of", str(as_of.date()),
                       "--stage", "chunk", "--start", str(start.date()), "--end", str(end.date())]
            subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            record = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        records.append(record)
    save_json(paths["data"] / "dataset_manifest.json", {**specification(HORIZON), "as_of": str(as_of.date()),
              "partitions": records, "rows": sum(r["rows"] for r in records),
              "financial_snapshot_sha256": digest(SOURCE / "financial_events.parquet")})


def read_period(paths, start, cutoff, names):
    spec = specification(HORIZON)
    parts = []
    for file in sorted((paths["data"] / "annual_shards").glob("*.parquet")):
        if file.stem[:4] < str(pd.Timestamp(start).year) or file.stem[:4] > str(pd.Timestamp(cutoff).year):
            continue
        frame = pd.read_parquet(file, columns=names + [spec["raw_label"], spec["target"], "label_end_date"])
        dates = frame.index.get_level_values("datetime")
        parts.append(frame.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(cutoff))
                               & frame.label_end_date.le(pd.Timestamp(cutoff))].dropna(subset=[spec["raw_label"], spec["target"]]))
    result = pd.concat(parts).sort_index()
    if result.empty or result.index.has_duplicates:
        raise ValueError("No valid mature data, or duplicate index")
    return select_mature_period(result, start, cutoff)


def train_models(paths, as_of):
    require_memory(4.0, "162因子全量LightGBM训练")
    import lightgbm as lgb
    from paper_futu.backtest_label_grid_2022_2025 import daily_ic
    from src.train_walk_forward_cv import build_validation_folds, summarize_iteration_candidates

    manifest = json.loads((paths["data"] / "dataset_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["partitions"]:
        if digest(ROOT / record["file"]) != record["sha256"]:
            raise ValueError("Training partition changed")
    spec = specification(HORIZON)
    if manifest["feature_recipe_sha256"] != spec["feature_recipe_sha256"]:
        raise ValueError("Feature specification changed; rebuild the dataset")
    parameters = {**deepcopy(load_model_config()["parameters"]), "n_jobs": 4, "random_state": 42}
    summary = []
    # Same 161 historical/PIT features, with/without the one observed-opening factor.
    for variant, opening in [("history_only", False), ("with_open", True)]:
        names = feature_names(opening)
        directory = paths["model"] / variant
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for fold in build_validation_folds(2026, 3, 2010):
            require_memory(4.0, "时间序列验证折")
            train = read_period(paths, fold["train_start"], fold["train_end"], names)
            valid = read_period(paths, fold["valid_start"], fold["valid_end"], names)
            print(f"{variant}: validation {fold['validation_year']}, train {len(train):,}, features {len(names)}", flush=True)
            model = lgb.LGBMRegressor(**{**parameters, "n_estimators": max(CANDIDATES)})
            model.fit(train[names], train[spec["target"]])
            for count in CANDIDATES:
                prediction = pd.Series(model.predict(valid[names], num_iteration=count), index=valid.index)
                ic = daily_ic(prediction, valid[spec["raw_label"]])
                records.append({**fold, "candidate_iteration": count, "valid_mean_ic": float(ic.ic.mean()),
                                "valid_mean_rank_ic": float(ic.rank_ic.mean()),
                                "valid_mse": float((prediction-valid[spec["target"]]).pow(2).mean()),
                                "train_label_end_max": str(train.label_end_date.max().date()),
                                "valid_label_end_max": str(valid.label_end_date.max().date())})
            pd.DataFrame(records).to_csv(paths["report"] / f"{variant}_validation_folds.csv", index=False)
            del train, valid, model
            gc.collect()
        selections = summarize_iteration_candidates(records)
        chosen = int(selections.iloc[0].candidate_iteration)
        selections.to_csv(paths["report"] / f"{variant}_candidates.csv", index=False)
        train = read_period(paths, "2010-01-01", as_of, names)
        model = lgb.LGBMRegressor(**{**parameters, "n_estimators": chosen})
        model.fit(train[names], train[spec["target"]])
        path = directory / "model.txt"
        model.booster_.save_model(str(path))
        check = train[names].iloc[::max(1, len(train)//1000)]
        reloaded = lgb.Booster(model_file=str(path))
        if not np.allclose(model.predict(check), reloaded.predict(check, num_threads=4), atol=1e-12, rtol=0):
            raise ValueError("Saved/reloaded predictions differ")
        latest_path = next(r["file"] for r in manifest["partitions"]
                           if pd.Timestamp(r["start"]) <= as_of <= pd.Timestamp(r["end"]))
        latest = pd.read_parquet(ROOT / latest_path, columns=names)
        latest = latest.loc[latest.index.get_level_values("datetime") == as_of]
        prediction = pd.DataFrame({"prediction": reloaded.predict(latest[names], num_threads=4)}, index=latest.index)
        if not np.isfinite(prediction.prediction).all():
            raise ValueError("Invalid latest scores")
        prediction.sort_values("prediction", ascending=False).to_csv(paths["report"] / f"{variant}_latest_ranking.csv")
        importance = pd.DataFrame({"factor": names, "gain": reloaded.feature_importance("gain")}).sort_values("gain", ascending=False)
        importance.to_csv(paths["report"] / f"{variant}_feature_importance.csv", index=False)
        metadata = {**spec, "variant": variant, "features": names, "feature_count": len(names),
                    "information_cutoff": str(as_of.date()), "last_training_signal": str(train.index.get_level_values("datetime").max().date()),
                    "last_label_end_date": str(train.label_end_date.max().date()), "train_rows": len(train),
                    "chosen_iteration": chosen, "parameters": {**parameters, "n_estimators": chosen},
                    "selection_mean_rank_ic": float(selections.iloc[0].mean_validation_rank_ic),
                    "model_sha256": digest(path), "dataset_manifest_sha256": digest(paths["data"] / "dataset_manifest.json"),
                    "deployment_status": "offline_only_requires_post_observation_execution_validation",
                    "latest_ranking_retrospective": True,
                    "daily_open_price_fill_backtest_allowed": False}
        save_json(directory / "metadata.json", metadata)
        summary.append({name: metadata[name] for name in ["variant", "feature_count", "chosen_iteration", "selection_mean_rank_ic", "train_rows"]})
        pd.DataFrame(summary).to_csv(paths["report"] / "opening_information_ablation.csv", index=False)
        del model, reloaded, train, check, latest
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--stage", choices=["sample", "data", "train", "all", "chunk"], default="sample")
    parser.add_argument("--sample-symbols", type=int, default=30)
    parser.add_argument("--sample-days", type=int, default=5)
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of).normalize()
    paths = paths_for(as_of)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if args.stage == "chunk":
        if not args.start or not args.end:
            raise ValueError("The chunk stage requires --start and --end")
        require_memory(2.0, "Alpha158年度数据分块")
        build_chunk(paths, as_of, pd.Timestamp(args.start), pd.Timestamp(args.end))
        return
    before = protected_state()
    save_json(paths["report"] / "feature_specification.json", specification(HORIZON))
    try:
        if args.stage == "sample":
            require_memory(.35, "小样本因子校验")
            provider = prepare_provider(paths, as_of)
            calendar = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines()))
            audit = build_chunk(paths, as_of, calendar[-args.sample_days], as_of, args.sample_symbols, True)
            save_json(paths["report"] / "sample_audit.json", audit)
            print(f"Morning information boundary: {audit_morning_reconstruction(paths, as_of)}", flush=True)
        else:
            if args.stage in ["data", "all"]:
                build_all_data(paths, as_of)
            if args.stage in ["train", "all"]:
                train_models(paths, as_of)
    finally:
        after = protected_state()
        save_json(paths["report"] / "deployment_unchanged.json", {"unchanged": before == after, "before": before, "after": after})
        if before != after:
            raise RuntimeError("Existing deployment or confirmed plans changed; manual review required")


if __name__ == "__main__":
    main()
