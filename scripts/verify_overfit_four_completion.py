"""Independently reconcile the completed four-strategy study, without refitting."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from paper_futu import overfit_four_core as core


def main():
    started = time.time()
    store = core.Store()
    protocol = store.protocol
    completion = core.read(core.OUT / "completion.json")
    manifest = core.read(core.OUT / "report_manifest.json")
    assert completion["status"] == manifest["status"] == "complete"
    assert completion["implementation_sha256"] == core.scientific_hash()
    assert manifest["snapshot_id"] == protocol["snapshot_id"]
    assert not completion["selected_champion"]
    for name, digest in manifest["results"].items():
        assert core.frozen.digest(name) == digest, name
    scheduler = manifest["scheduler"]
    assert core.frozen.digest(scheduler["path"]) == scheduler["sha256"]
    folds = core.fold_jobs(protocol)
    fits = core.read(core.OUT / "fit_jobs.json")
    accounts = core.read(core.OUT / "account_jobs.json")
    diagnostics = core.read(core.OUT / "diagnostic_jobs.json")
    assert (len(folds), len(fits), len(accounts), len(diagnostics)) == (136, 140, 340, 370)
    assert core.frozen.identity(fits) == core.frozen.identity(
        core.final_jobs(protocol, core.read(core.OUT / "selected_trees.json")))
    all_jobs = folds + fits + accounts + diagnostics
    keys = [store.key(job) for job in all_jobs]
    assert len(set(keys)) == len(keys)
    for job in all_jobs:
        assert store.record(job), store.key(job)
    print("Verified all task keys and cached artifact hashes", flush=True)

    for job in folds:
        year = job["validation_year"]
        for row in store.record(job)["result"]["records"]:
            assert pd.Timestamp(row["train_label_end_max"]) <= pd.Timestamp(year - 1, 12, 31)
            assert pd.Timestamp(row["valid_label_end_max"]) <= pd.Timestamp(year, 12, 31)

    reloaded = 0
    for family in core.FAMILIES:
        frame = core.load_frame(family)
        dates = frame.index.get_level_values("datetime")
        for job in (job for job in fits if job["family"] == family):
            record = store.record(job)["result"]
            parent = next(row for row in protocol["families"][family]["periods"]
                          if row["selection_year"] == job["execution_year"])
            cutoff = pd.Timestamp(parent["cutoff"])
            first = core.start_date(pd.Timestamp(job["execution_year"], 1, 1), job["variant"])
            mask = ((dates >= first) & (dates <= cutoff) & frame.label_end_date.le(cutoff)
                    & frame.label.notna() & frame.target.notna())
            assert int(mask.sum()) == record["train_rows"]
            assert pd.Timestamp(record["label_end_max"]) == frame.loc[mask, "label_end_date"].max()
            assert pd.Timestamp(record["label_end_max"]) <= cutoff
            part = pd.read_parquet(record["path"])
            assert part.index.is_unique and np.isfinite(part.prediction).all()
            assert part.execution_date.between(parent["serving_start"], parent["serving_end"]).all()
            subset = part.iloc[np.linspace(0, len(part) - 1, min(128, len(part)), dtype=int)]
            loaded = lgb.Booster(model_file=record["model_path"])
            predicted = loaded.predict(frame.reindex(subset.index)[list(core.frozen.ENHANCED_FEATURE_NAMES)], num_threads=4)
            np.testing.assert_allclose(predicted, subset.prediction, atol=1e-12, rtol=0)
            reloaded += 1
        del frame
    print(f"Verified mature boundaries and reloaded {reloaded} final models", flush=True)

    reference_index = reference_benchmark = None
    sessions = core.calendar()
    expected_days = sessions[(sessions >= core.frozen.START) & (sessions <= core.frozen.AS_OF)]
    for index, job in enumerate(accounts):
        result = store.record(job)["result"]
        directory = Path(result["directory"])
        core.verify_account(directory)
        daily = pd.read_parquet(directory / "daily.parquet")
        assert daily.index.equals(expected_days)
        if reference_index is None:
            reference_index, reference_benchmark = daily.index, daily.bench
        else:
            assert daily.index.equals(reference_index)
            np.testing.assert_allclose(daily.bench, reference_benchmark, rtol=0, atol=1e-15)
        if (index + 1) % 50 == 0:
            print(f"Independently reconciled {index + 1}/340 accounts", flush=True)
    streams = core.read(core.OUT / "streams.json")
    assert len(streams) == 30
    for family in core.FAMILIES:
        source = pd.read_parquet(protocol["families"][family]["prediction_path"])
        for stream in (stream for stream in streams if stream["family"] == family):
            values = pd.read_parquet(stream["prediction_path"])
            assert values.index.is_unique and values.index.equals(source.index)
            assert np.isfinite(values.prediction).all()
            assert values.execution_date.equals(source.execution_date)
    intervals = pd.read_csv(core.OUT / "paired_block_bootstrap.csv")
    assert len(intervals) == 96 and set(intervals.block) == {60, 120}
    assert intervals.repeats.eq(2000).all()
    assert np.isfinite(intervals[["point_delta", "ci_2_5", "ci_97_5"]]).all().all()
    core.verify_protected(protocol, full_snapshot=True)
    result = dict(status="passed", verified_at=time.time(), elapsed_seconds=time.time() - started,
        job_counts=dict(Counter(job["kind"] for job in all_jobs)),
        final_models_reloaded=reloaded, sampled_predictions_per_model=128,
        accounts_reconciled=340, streams_verified=30, paired_interval_rows=96,
        task_and_artifact_hashes=True, mature_training_and_validation_boundaries=True,
        account_nav_cash_fees_and_shared_benchmark=True, snapshot_and_protected_state=True,
        scientific_implementation_sha256=core.scientific_hash(), runtime_scheduler_sha256=scheduler["sha256"],
        original_report_manifest_sha256=core.frozen.digest(core.OUT / "report_manifest.json"))
    core.save(core.OUT / "final_verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
