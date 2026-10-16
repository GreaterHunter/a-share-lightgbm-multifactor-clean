"""Build raw/ranked features and exact forward labels on one frozen provider."""
from __future__ import annotations

import shutil
import numpy as np
import pandas as pd

from paper_futu.final_ablation_core import (
    AS_OF, DATA, FINANCIAL_SOURCE, PROVIDER, REPORT, SOURCE_PROVIDER, SCHEDULES,
    digest, identity, init_qlib, read_json, save_json,
)
from src.config import PROJECT_ROOT
from src.factor_library import ENHANCED_FEATURE_NAMES, FEATURE_EXPRESSIONS
from src.build_enhanced_dataset import (
    merge_latest_event, prepare_annual_eps_events, prepare_general_report_events,
    validate_point_in_time,
)


def prepare_data():
    manifest_path = DATA / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for path, expected in manifest["artifacts"].items():
            if not (DATA / path).exists() or digest(DATA / path) != expected:
                raise ValueError(f"Frozen data artifact changed: {path}")
        print(f"Verified frozen snapshot {manifest['snapshot_id']}", flush=True)
        return manifest
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    print("Copying a private quote snapshot", flush=True)
    for folder in ["calendars", "instruments", "features"]:
        shutil.copytree(SOURCE_PROVIDER / folder, PROVIDER / folder, dirs_exist_ok=True)
    calendar_path = PROVIDER / "calendars/day.txt"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    calendar = calendar[calendar <= AS_OF]
    if calendar[-1] != AS_OF or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar must reach the exact cutoff and be unique/sorted")
    calendar_path.write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    # Delete no source data. Truncate copied binary arrays at the information cutoff.
    for path in (PROVIDER / "features").rglob("*.bin"):
        values = np.fromfile(path, dtype="<f4")
        if len(values):
            length = max(1, 1 + len(calendar) - int(values[0]))
            values[:length].tofile(path)
    init_qlib()
    from qlib.data import D
    expressions = list(FEATURE_EXPRESSIONS.values()) + ["$close / ($factor + 1e-12)"]
    daily = D.features(D.instruments("csi300"), expressions, start_time="2010-01-01",
                       end_time=str(AS_OF.date()), freq="day")
    daily.columns = list(FEATURE_EXPRESSIONS) + ["raw_close"]
    events = pd.read_parquet(FINANCIAL_SOURCE)
    events = events.loc[events.available_date.le(AS_OF)].copy()
    if events.available_date.le(events.report_date).any():
        raise ValueError("Financial availability precedes report end")
    events.to_parquet(DATA / "financial_events.parquet", index=False)
    panel = merge_latest_event(daily.reset_index(), prepare_general_report_events(events),
                               "fundamental_available_date")
    panel = merge_latest_event(panel, prepare_annual_eps_events(events), "annual_eps_available_date")
    validate_point_in_time(panel)
    price = panel.raw_close.where(panel.raw_close.gt(0))
    panel["book_to_price"] = panel.book_value_per_share / price
    panel["earnings_to_price"] = panel.annual_eps / price
    alignment = panel[["instrument", "datetime", "fundamental_report_date", "fundamental_available_date",
                       "annual_eps_report_date", "annual_eps_available_date"]]
    alignment.to_parquet(DATA / "financial_alignment.parquet", index=False)
    raw = panel.set_index(["instrument", "datetime"])[ENHANCED_FEATURE_NAMES]
    raw = raw.replace([np.inf, -np.inf], np.nan).sort_index()
    raw = raw.loc[~raw.isna().all(axis=1)].astype("float32")
    if raw.index.has_duplicates or np.isinf(raw.to_numpy()).any():
        raise ValueError("Duplicate or infinite raw factors")
    ranked = (raw.groupby(level="datetime").rank(method="average", pct=True) - .5).astype("float32")
    raw.to_parquet(DATA / "raw_features.parquet")
    ranked.to_parquet(DATA / "rank_features.parquet")
    raw.isna().mean().rename("missing_ratio").to_csv(REPORT / "feature_missing.csv", encoding="utf-8-sig")
    overlap_path = PROJECT_ROOT / "data/processed/correct_open_grid_top10_2022_2026_asof_20261013/ranked_features_source.parquet"
    if overlap_path.exists():
        old = pd.read_parquet(overlap_path)
        common = ranked.index.intersection(old.index)
        delta = (ranked.loc[common] - old.loc[common, ENHANCED_FEATURE_NAMES]).abs()
        save_json(REPORT / "old_rank_overlap.json", dict(rows=len(common),
                  index_identical=ranked.index.equals(old.index), maximum_rank_difference=float(delta.max().max()),
                  source_sha256=digest(overlap_path), old_results_reusable=False,
                  added_rows=len(ranked.index.difference(old.index)),
                  maximum_difference_by_factor=delta.max().to_dict(),
                  note="Old build_dataset dropped missing future labels before ranking; new factors rank all signal-date rows without future-label filtering."))
    dates = raw.index.get_level_values("datetime")
    positions = calendar.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Unknown feature date")
    for horizon in SCHEDULES:
        label = D.features(D.instruments("csi300"),
                           [f"Ref($open, -{horizon+1}) / Ref($open, -1) - 1"],
                           start_time="2010-01-01", end_time=str(AS_OF.date()), freq="day")
        label.columns = ["label"]
        label = label.reindex(raw.index).replace([np.inf, -np.inf], np.nan)
        ends = positions + horizon + 1
        end_dates = np.full(len(ends), np.datetime64("NaT"), dtype="datetime64[ns]")
        known = ends < len(calendar)
        end_dates[known] = calendar.to_numpy()[ends[known]]
        label["label_end_date"] = end_dates
        label.loc[~known, "label"] = np.nan
        label["rank_target"] = label.label.groupby(level="datetime").rank(method="average", pct=True) - .5
        label.to_parquet(DATA / f"labels_h{horizon}.parquet")
    # Independent, already crosschecked terminal benchmark patch (provider lacks its last index bar).
    benchmark_source = PROJECT_ROOT / "data/processed/correct_open_grid_top10_2022_2026_asof_20261013/csi300_terminal_benchmark_audit.json"
    benchmark = read_json(benchmark_source)
    previous = D.features(["SH000300"], ["$close / $factor"], start_time="2026-10-12", end_time="2026-10-12").iloc[0, 0]
    if not np.isclose(previous, benchmark["previous_close"], atol=.01):
        raise ValueError("Terminal benchmark patch failed its quote crosscheck")
    save_json(DATA / "benchmark_audit.json", benchmark)
    hashes = {str(p.relative_to(DATA)): digest(p) for p in sorted(DATA.rglob("*")) if p.is_file()}
    manifest = dict(as_of=str(AS_OF.date()), start="2010-01-01", rows=len(raw),
                    feature_order=ENHANCED_FEATURE_NAMES, source_provider=str(SOURCE_PROVIDER),
                    financial_source_sha256=digest(FINANCIAL_SOURCE), artifacts=hashes,
                    coverage=raw.notna().mean().to_dict(),
                    actual_latest_feature_date=str(dates.max().date()),
                    label="Open(T+H+1)/Open(T+1)-1", universe="source historical CSI300 intervals",
                    financial_revision_archive_complete=False)
    manifest["snapshot_id"] = identity(manifest)
    save_json(manifest_path, manifest)
    print(f"Built {len(raw):,} raw/ranked stock-date rows", flush=True)
    return manifest
