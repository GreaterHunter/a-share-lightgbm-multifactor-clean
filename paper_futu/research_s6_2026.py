"""Isolated S6 2026 frozen-year and causal quarterly/half-year comparisons."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
NAME = "s6_head6_40_2026_holdout_asof_20261013"
OUT = ROOT / "reports/experiments" / NAME
MODELS = ROOT / "models/experiments" / NAME
DATA = ROOT / "data/processed/final_ablation_asof_20261013"
STUDY = ROOT / "reports/experiments/overfit_four_asof_20261013"
SOURCE = ROOT / "reports/experiments/final_ablation_asof_20261013"
VARIANTS = ("V0", "V2", "V3", "V4", "V7", "V8")
CASES = (*VARIANTS, "M6", "M3")
START, END = "2026-01-05", "2026-10-13"
HISTORY_START = "2022-01-04"
PORTFOLIO = dict(top_k=20, top_tier_size=6, top_tier_weight=.4, n_drop=6)
ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
       "NUMEXPR_NUM_THREADS": "4", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as source:
        h = hashlib.file_digest(source, "sha256")
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def implementation():
    files = [ROOT / "paper_futu" / name for name in
             ("research_s6_2026.py", "s6_2026_account.py", "s6_2026_report.py")]
    return identity({p.name: digest(p) for p in files})


def protocol():
    value = read(OUT / "protocol.json")
    if value["implementation_sha256"] != implementation():
        raise ValueError("Implementation changed: review cache compatibility before resuming")
    return value


def verify_artifacts(record):
    for name, expected in record["artifacts"].items():
        if digest(name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")


def cached(kind, name):
    path = OUT / "tasks" / f"{kind}_{name}.json"
    if not path.exists():
        return None
    record = read(path)
    if record["status"] != "complete":
        return None
    if record["implementation_sha256"] != implementation():
        raise ValueError("Cached task belongs to another implementation")
    if record["protocol_sha256"] != digest(OUT / "protocol.json"):
        raise ValueError("Cached task belongs to another protocol")
    verify_artifacts(record)
    return record["result"]


def run_task(kind, name, function):
    existing = cached(kind, name)
    if existing is not None:
        return existing
    import psutil
    path = OUT / "tasks" / f"{kind}_{name}.json"
    record = dict(status="running", kind=kind, name=name, pid=os.getpid(),
                  created=psutil.Process().create_time(), started=time.time(),
                  implementation_sha256=implementation(), snapshot_id=protocol()["snapshot_id"])
    record["protocol_sha256"] = digest(OUT / "protocol.json")
    save(path, record)
    try:
        result, artifacts = function()
        record.update(status="complete", result=result, elapsed=time.time()-record["started"],
                      artifacts={str(p): digest(p) for p in artifacts})
        save(path, record)
        print(f"COMPLETE {kind} {name} seconds={record['elapsed']:.1f}", flush=True)
        return result
    except BaseException as error:
        record.update(status="failed", error=repr(error), elapsed=time.time()-record["started"])
        save(path, record)
        raise


def calendar():
    import pandas as pd
    return pd.DatetimeIndex(pd.to_datetime((DATA / "provider/calendars/day.txt").read_text().splitlines()))


def serving_periods(months, start=HISTORY_START, end=END):
    """All model cutoffs precede their execution dates; holiday dates come from the snapshot."""
    import pandas as pd
    sessions = calendar()
    result = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year+1):
        for month in range(1, 13, months):
            boundary = pd.Timestamp(year, month, 1)
            span = sessions[(sessions >= boundary) & (sessions < boundary+pd.DateOffset(months=months))
                            & (sessions >= start) & (sessions <= end)]
            if len(span):
                result.append(dict(start=str(span[0].date()), end=str(span[-1].date()),
                                   cutoff=str(sessions[sessions < span[0]][-1].date()), year=year))
    return result


def fit_cutoffs():
    return sorted({p["cutoff"] for p in serving_periods(3)
                   if not p["start"][5:7] == "01"})


def load_frame():
    import pandas as pd
    from paper_futu import final_ablation_core as frozen
    result = pd.read_parquet(DATA / "raw_features.parquet", columns=list(frozen.ENHANCED_FEATURE_NAMES))
    result = result.join(pd.read_parquet(DATA / "labels_h10.parquet"))
    result["target"] = result.rank_target
    return result


def source_bindings():
    """Bind existing models and validation tasks without changing either original experiment."""
    import numpy as np
    import pandas as pd
    from paper_futu import overfit_four_core as core
    from paper_futu.overfit_four_diagnostics import source_fold
    store = core.Store()
    source = read(SOURCE / "predictions/lgb_15_raw_rank_h10_fixed_m12_s42.json")
    streams = read(STUDY / "streams.json")
    choices = read(STUDY / "selected_trees.json")
    bindings, files = {}, {}
    for variant in VARIANTS:
        if variant == "V0":
            parent = next(p for p in source["periods"] if p["selection_year"] == 2026)
            prediction = source["prediction_path"]
            task = read(SOURCE / "tasks" / (Path(parent["model_path"]).parent.name+".json"))
        else:
            stream = next(s for s in streams if s["family"] == "raw10" and s["variant"] == variant and s["seed"] == 42)
            parent = next(p for p in stream["periods"] if p["execution_year"] == 2026)
            prediction = stream["prediction_path"]
            task = read(STUDY / "tasks" / (Path(parent["model_path"]).parent.name+".json"))
        if task["status"] != "complete":
            raise ValueError("Source final fit is incomplete")
        verify_artifacts(task)
        if digest(parent["model_path"]) != parent["model_sha256"]:
            raise ValueError("Source model hash mismatch")
        choice = next(c for c in choices if c["family"] == "raw10" and c["variant"] == variant and c["execution_year"] == 2026)
        folds = []
        for year in (2023, 2024, 2025):
            if variant == "V0":
                rows = source_fold(store.protocol, "raw10", parent, year)
            else:
                job = dict(kind="fold", family="raw10", variant=variant,
                           parameters=choice["parameters"], validation_year=year)
                fold = store.record(job)
                if not fold:
                    raise ValueError("Source validation fit is incomplete")
                rows = fold["result"]["records"]
            for row in rows:
                if (pd.Timestamp(row["train_label_end_max"]) >= pd.Timestamp(year, 1, 1)
                        or pd.Timestamp(row["valid_label_end_max"]) > pd.Timestamp(year, 12, 31)):
                    raise ValueError("Source validation labels cross a boundary")
            folds.extend(rows)
        selection = core.frozen.choose_candidates(pd.DataFrame(folds))
        if int(selection.iloc[0].trees) != choice["trees"] or parent["trees"] != choice["trees"]:
            raise ValueError("Recomputed tree selection differs from cached model")
        if parent["cutoff"] != "2025-12-31" or parent["label_end_max"] > "2025-12-31":
            raise ValueError("A frozen-year model used 2026 training labels")
        bindings[variant] = dict(parent=parent, parameters=choice["parameters"], trees=choice["trees"],
                                 prediction_path=prediction, fold_records=folds,
                                 selected_mean_rank_ic=float(selection.iloc[0].mean_rank_ic))
        for path in (prediction, parent["model_path"]):
            files[str(path)] = digest(path)
    for path in (SOURCE / "predictions/lgb_15_raw_rank_h10_fixed_m12_s42.json",
                 STUDY / "streams.json", STUDY / "selected_trees.json"):
        files[str(path)] = digest(path)
    annual = {}
    for parent in source["periods"]:
        year = parent["selection_year"]
        rows = []
        for validation_year in range(year-3, year):
            fold = source_fold(store.protocol, "raw10", parent, validation_year)
            for row in fold:
                if (pd.Timestamp(row["train_label_end_max"]) >= pd.Timestamp(validation_year,1,1)
                        or pd.Timestamp(row["valid_label_end_max"]) > pd.Timestamp(validation_year,12,31)):
                    raise ValueError("Historical annual validation boundary crossed")
            rows.extend(fold)
        chosen = core.frozen.choose_candidates(pd.DataFrame(rows)).iloc[0]
        if int(chosen.trees) != parent["trees"] or parent["label_end_max"] > parent["cutoff"]:
            raise ValueError("Historical annual tree/boundary audit failed")
        task = read(SOURCE / "tasks" / (Path(parent["model_path"]).parent.name+".json"))
        verify_artifacts(task)
        files[parent["model_path"]] = digest(parent["model_path"])
        annual[str(year)] = dict(parent=parent, trees=parent["trees"], parameters=parent["config"]["parameters"],
                                validation_years=list(range(year-3,year)), fold_records=rows)
    return bindings, files, annual


def frame_fingerprint(frame):
    import pandas as pd
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def audit():
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from paper_futu import overfit_four_core as core
    from src.runtime_budget import require_memory
    require_memory(3., "S6 2026 source audit")
    store = core.Store()
    print("AUDIT protected sources and full snapshot", flush=True)
    core.verify_protected(store.protocol, full_snapshot=True)
    bindings, sources, annual = source_bindings()
    label_audit = core.verify_forward_labels(10)
    frame = load_frame()
    if frame.index.has_duplicates or np.isinf(frame[list(core.frozen.ENHANCED_FEATURE_NAMES)]).any().any():
        raise ValueError("Duplicate stock/date samples or infinite features")
    common, audits = None, {}
    for variant, binding in bindings.items():
        parent = binding["parent"]
        train = core.mature(frame, parent["train_start"], parent["cutoff"])
        if len(train) != parent["train_rows"]:
            raise ValueError("Reconstructed training sample differs from saved model")
        old = pd.read_parquet(binding["prediction_path"])
        pred = old.loc[old.execution_date.between(START, END), ["prediction", "execution_date"]].sort_index()
        if common is None:
            common = pred.index
        if not pred.index.equals(common) or not np.isfinite(pred.prediction).all():
            raise ValueError("Frozen comparison prediction samples differ")
        loaded = lgb.Booster(model_file=parent["model_path"])
        values = loaded.predict(frame.reindex(pred.index)[list(core.frozen.ENHANCED_FEATURE_NAMES)], num_threads=4)
        np.testing.assert_allclose(values, pred.prediction, atol=1e-12, rtol=0)
        if loaded.num_trees() != binding["trees"]:
            raise ValueError("Model has unexpected tree count")
        pred["model_cutoff"] = pd.Timestamp(parent["cutoff"])
        path = OUT / "predictions" / f"{variant}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(path)
        audits[variant] = dict(train_rows=len(train), train_start=parent["train_start"],
            train_signal_end=str(train.index.get_level_values("datetime").max().date()),
            label_end_max=str(train.label_end_date.max().date()), trees=binding["trees"],
            training_fingerprint=frame_fingerprint(train), model_reload_prediction_equal=True)
        del train, loaded, old
        print(f"AUDIT {variant} trees={binding['trees']} passed", flush=True)
    source_prediction = pd.read_parquet(bindings["V0"]["prediction_path"])
    for year,binding in annual.items():
        parent = binding["parent"]
        source_part = source_prediction.loc[source_prediction.execution_date.dt.year.eq(int(year))]
        model = lgb.Booster(model_file=parent["model_path"])
        values = model.predict(frame.reindex(source_part.index)[list(core.frozen.ENHANCED_FEATURE_NAMES)],num_threads=4)
        np.testing.assert_allclose(values,source_part.prediction,atol=1e-12,rtol=0)
        if model.num_trees() != binding["trees"]:
            raise ValueError("Historical annual model tree count changed")
    del source_prediction,model
    # Perturb every post-cutoff target, including labels attached to 2025 signals
    # whose outcome matures in 2026; the exact mature sample must remain identical.
    original = core.mature(frame, "2010-01-01", "2025-12-31")
    before = frame_fingerprint(original)
    future = frame.label_end_date.gt(pd.Timestamp("2025-12-31")) | (frame.index.get_level_values("datetime") > "2025-12-31")
    frame.loc[future, ["label", "target", "rank_target"]] = 999.
    assert frame_fingerprint(core.mature(frame, "2010-01-01", "2025-12-31")) == before
    historical_baselines = read(ROOT / "reports/experiments/s6_head6_40_asof_20261013/verified_accounts.json")["new_accounts"]
    for baseline in historical_baselines:
        for name, expected in baseline["artifacts"].items():
            if digest(name) != expected:
                raise ValueError("Original S6 head-six continuous account changed")
        core.verify_account(Path(baseline["directory"]))
        sources.update(baseline["artifacts"])
    settings = dict(snapshot_id=store.protocol["snapshot_id"], implementation_sha256=implementation(),
        source_science_sha256=core.frozen.code_hash(), protected=core.frozen.protected_state(),
        bindings=bindings, annual_bindings=annual, sources=sources, banks=sorted(store.protocol["bank_names"]),
        historical_baseline_accounts=historical_baselines, history_start=HISTORY_START,
        portfolio=PORTFOLIO, start=START, end=END, seed=42, rebalance=5, offsets=list(range(5)),
        initial_cash=1_000_000, risk_degree=.95, buy_cost=.0005, sell_cost=.0015, min_cost=5.,
        trade_unit=100, min_start_gib=3., min_task_gib=2., max_workers=2, threads=4,
        new_fits=15, new_accounts=50, accounts=55, reused_accounts=5,
        frozen_variants=list(VARIANTS), cadence_cases=["M6", "M3"],
        cadence_periods={str(m): serving_periods(m) for m in (12, 6, 3)},
        limitations=[*core.frozen.LIMITATIONS,
            "2026 frozen-year and rolling-year evaluations have different training information sets.",
            "Fresh-mode accounts start from cash in 2026; historical cadence accounts continue from 2022."])
    if (OUT / "protocol.json").exists() and read(OUT / "protocol.json") != settings:
        raise ValueError("Existing experiment binding changed")
    save(OUT / "protocol.json", settings)
    save(OUT / "audit.json", dict(status="passed", models=audits, labels=label_audit,
        future_target_perturbation_invariant=True, common_prediction_rows=len(common),
        protected_unchanged=True, protocol_sha256=digest(OUT / "protocol.json")))
    main = dict(strategy_id="S6_head6_40", research_main=True, variant="V0", retrain_months=12,
        model_path=bindings["V0"]["parent"]["model_path"],
        model_sha256=bindings["V0"]["parent"]["model_sha256"], **{k: settings[k] for k in
        ("snapshot_id", "portfolio", "rebalance", "risk_degree", "seed")},
        horizon=10, input_kind="raw", target_kind="rank", label="Open(T+11)/Open(T+1)-1",
        test_information_cutoff="2025-12-31", deployment_changed=False)
    save(OUT / "research_main.json", main)
    save(ROOT / "configs/experiments/s6_head6_40_research_main.json", main)
    print("AUDIT complete: six reusable models, no 2026 frozen training labels", flush=True)


def check_sources():
    p = protocol()
    from paper_futu import final_ablation_core as frozen
    if frozen.code_hash() != p["source_science_sha256"] or frozen.protected_state() != p["protected"]:
        raise ValueError("Protected scientific or deployment state changed")
    for path, expected in p["sources"].items():
        if digest(path) != expected:
            raise ValueError(f"Source changed: {path}")
    return p


def fit(cutoff):
    def execute():
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
        from paper_futu import overfit_four_core as core
        from src.runtime_budget import require_memory
        require_memory(2., "S6 mature-label update fit")
        p = protocol()
        annual = p["annual_bindings"][str(pd.Timestamp(cutoff).year)]
        training = core.mature(load_frame(), "2010-01-01", cutoff)
        model = lgb.LGBMRegressor(**{**annual["parameters"], "n_estimators":annual["trees"], "random_state":42})
        model.fit(training[list(core.frozen.ENHANCED_FEATURE_NAMES)], training.target)
        directory = MODELS / cutoff
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.txt"
        model.booster_.save_model(str(path))
        loaded = lgb.Booster(model_file=str(path))
        check = training.iloc[::max(1, len(training)//2000)][list(core.frozen.ENHANCED_FEATURE_NAMES)]
        np.testing.assert_allclose(model.predict(check), loaded.predict(check, num_threads=4), atol=1e-12, rtol=0)
        importance = directory / "importance.csv"
        pd.DataFrame(dict(factor=list(core.frozen.ENHANCED_FEATURE_NAMES), gain=loaded.feature_importance("gain"))).to_csv(importance,index=False)
        result = dict(cutoff=cutoff, train_start="2010-01-01", train_rows=len(training),
            train_signal_end=str(training.index.get_level_values("datetime").max().date()),
            label_end_max=str(training.label_end_date.max().date()), model_path=str(path), model_sha256=digest(path),
            parameters=annual["parameters"], trees=annual["trees"], seed=42,
            training_fingerprint=frame_fingerprint(training), reload_equal=True)
        if result["label_end_max"] > cutoff:
            raise ValueError("Update training labels cross cutoff")
        save(directory / "metadata.json", result)
        return result, [path, importance, directory / "metadata.json"]
    return run_task("fit", cutoff, execute)


def assemble():
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from paper_futu import final_ablation_core as frozen
    p = protocol()
    features = pd.read_parquet(DATA / "raw_features.parquet", columns=list(frozen.ENHANCED_FEATURE_NAMES))
    base = pd.read_parquet(p["bindings"]["V0"]["prediction_path"])[["prediction","execution_date"]].sort_index()
    base["model_cutoff"] = pd.to_datetime(base.execution_date.dt.year.astype(str).map(
        {y:b["parent"]["cutoff"] for y,b in p["annual_bindings"].items()}))
    base.to_parquet(OUT / "predictions/history_V0.parquet")
    for months, case in ((6, "M6"), (3, "M3")):
        pieces = []
        for period in serving_periods(months):
            selected = base.loc[base.execution_date.between(period["start"], period["end"])].copy()
            if period["start"][5:7] != "01":
                result = cached("fit", period["cutoff"])
                if result is None:
                    raise ValueError("Missing cadence fit")
                model = lgb.Booster(model_file=result["model_path"])
                selected["prediction"] = model.predict(features.reindex(selected.index), num_threads=4)
                selected["model_cutoff"] = pd.Timestamp(period["cutoff"])
            if (selected.model_cutoff >= selected.execution_date).any():
                raise ValueError("Model information cutoff reaches serving date")
            pieces.append(selected)
        result = pd.concat(pieces).sort_index()
        if not result.index.equals(base.index) or not np.isfinite(result.prediction).all():
            raise ValueError("Cadence changed serving sample")
        result.to_parquet(OUT / "predictions" / f"history_{case}.parquet")
        result.loc[result.execution_date.between(START,END)].to_parquet(OUT / "predictions" / f"{case}.parquet")
    half = pd.read_parquet(OUT / "predictions/history_M6.parquet")
    quarter = pd.read_parquet(OUT / "predictions/history_M3.parquet")
    for year in range(2022,2027):
        for other, boundary in ((half,f"{year}-07-01"),(quarter,f"{year}-04-01")):
            mask = base.execution_date.between(f"{year}-01-01", pd.Timestamp(boundary)-pd.Timedelta(days=1))
            np.testing.assert_array_equal(base.loc[mask,"prediction"],other.loc[mask,"prediction"])
        mask = half.execution_date.between(f"{year}-07-01",f"{year}-09-30")
        np.testing.assert_array_equal(half.loc[mask,"prediction"],quarter.loc[mask,"prediction"])
    names = (*CASES,"history_V0","history_M6","history_M3")
    save(OUT / "streams.json", dict(streams={case:dict(path=str(OUT / "predictions" / f"{case}.parquet"),
        sha256=digest(OUT / "predictions" / f"{case}.parquet")) for case in names},
        model_updates_change_no_rebalance_dates=True, initial_predictions_equal=True, shared_june_fit_equal=True))


def account_jobs():
    return ([dict(mode="fresh",case=case,offset=offset) for case in CASES for offset in range(5)]
            + [dict(mode="history",case=case,offset=offset) for case in ("M6","M3") for offset in range(5)])


def worker(kind, names):
    p = protocol()
    from src.runtime_budget import require_memory
    require_memory(2., "S6 worker")
    if kind == "fit":
        for name in names:
            if name not in fit_cutoffs():
                raise ValueError("Unexpected fit date")
            fit(name)
    else:
        from paper_futu.s6_2026_account import init_qlib, replay
        init_qlib()
        for name in names:
            mode,case,offset = name.split("_")
            if mode not in ("fresh","history") or case not in CASES or int(offset) not in range(5):
                raise ValueError("Unexpected account")
            run_task("account",name,lambda case=case,offset=offset,mode=mode:replay(case,int(offset),mode))


def execute_queue(kind, names):
    """Exclusive short batches, RAM-aware launch, durable one-shot memory retry."""
    import psutil
    pending = [n for n in names if cached(kind, n) is None]
    active, logs = {}, []
    retries_path = OUT / "memory_retries.json"
    retries = read(retries_path) if retries_path.exists() else {}
    max_workers,wait_since,failure = 2,None,None
    started,peak_parallel,min_free = time.time(),0,psutil.virtual_memory().available/2**30

    def active_snapshot():
        result = []
        for pid,(child,batch,_,_) in active.items():
            if child.poll() is not None:
                continue
            try:
                proc = psutil.Process(pid)
                result.append(dict(pid=pid,created=proc.create_time(),names=batch,command=proc.cmdline(),
                    rss_gib=proc.memory_info().rss/2**30,cpu_seconds=sum(proc.cpu_times()[:2]),
                    io=proc.io_counters()._asdict()))
            except psutil.Error:
                result.append(dict(pid=pid,names=batch,state="exiting"))
        return result
    for name in pending:
        task = OUT / "tasks" / f"{kind}_{name}.json"
        if task.exists() and read(task).get("status") == "failed":
            previous = read(task)
            key = f"{kind}_{name}"
            message = previous.get("error", "")
            if "GiB" not in message and "MemoryError" not in message:
                raise RuntimeError(f"Previous failure requires review: {task}")
            if key in retries:
                raise RuntimeError(f"Memory retry already exhausted: {name}")
            retries[key] = dict(used=True, original_error=message)
            save(retries_path, retries)
            max_workers = 1
    try:
        while pending or active:
            for pid, (child, batch, stdout, stderr) in list(active.items()):
                if child.poll() is None:
                    continue
                stdout.close(); stderr.close()
                del active[pid]
                missing = [n for n in batch if cached(kind, n) is None]
                if child.returncode or missing:
                    errors = [read(OUT / "tasks" / f"{kind}_{n}.json").get("error", "")
                              for n in missing if (OUT / "tasks" / f"{kind}_{n}.json").exists()]
                    memory = bool(errors) and all("GiB" in e or "MemoryError" in e for e in errors)
                    if memory and all(f"{kind}_{n}" not in retries for n in missing):
                        for n in missing:
                            retries[f"{kind}_{n}"] = dict(used=True, original_error=errors)
                        save(retries_path, retries)
                        pending = missing+pending
                        max_workers = 1
                    else:
                        failure = f"Worker {pid} failed: {errors}; inspect logs"
            if failure and not active:
                raise RuntimeError(failure)
            free = psutil.virtual_memory().available/2**30
            if pending and not active and free < 3.:
                wait_since = wait_since or time.time()
                if time.time()-wait_since >= 1800:
                    raise MemoryError("Available RAM remained below 3 GiB for 30 minutes; manual resume required")
            else:
                wait_since = None
            while not failure and pending and len(active) < max_workers and psutil.virtual_memory().available/2**30 >= 3.:
                count = 1 if kind == "fit" else min(5, len(pending))
                batch, pending = pending[:count], pending[count:]
                stamp = f"{kind}_{time.time_ns()}"
                logdir = OUT / "logs"; logdir.mkdir(exist_ok=True)
                stdout = (logdir / f"{stamp}.stdout.log").open("w", encoding="utf-8")
                stderr = (logdir / f"{stamp}.stderr.log").open("w", encoding="utf-8")
                logs.extend([stdout, stderr])
                command = [sys.executable, "-u", "-m", "paper_futu.research_s6_2026", "--stage", "worker",
                           "--kind", kind, "--names", ",".join(batch)]
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                    env={**os.environ, **ENV}, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                active[child.pid] = (child, batch, stdout, stderr)
                peak_parallel = max(peak_parallel,len(active))
            min_free = min(min_free,psutil.virtual_memory().available/2**30)
            completed = sum(cached(kind, n) is not None for n in names)
            save(OUT / "status.json", dict(status="draining_error" if failure else "running", phase=kind,
                controller_pid=os.getpid(), controller_created=psutil.Process().create_time(), updated=time.time(),
                completed=completed,total=len(names),free_gib=free,disk_free_gib=psutil.disk_usage(str(ROOT)).free/2**30,
                active=active_snapshot(),
                max_workers=max_workers,peak_parallel=peak_parallel,min_free_gib=min_free,error=failure))
            if pending or active:
                time.sleep(2)
    finally:
        for child, _, _, _ in active.values():
            child.wait()
        for log in logs:
            log.close()
        save(OUT / "runtime" / f"{kind}_{time.time_ns()}.json",dict(kind=kind,started=started,
            elapsed=time.time()-started,peak_parallel=peak_parallel,min_free_gib=min_free,error=failure))


@contextmanager
def controller_lock():
    import psutil
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "controller.lock"
    # An old controller lock never authorizes overlap with a live orphan worker.
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        if process.pid != os.getpid() and "paper_futu.research_s6_2026" in (process.info["cmdline"] or []):
            raise RuntimeError(f"Existing experiment process: {process.info}")
    if path.exists():
        old = read(path)
        try:
            proc = psutil.Process(old["pid"])
            if abs(proc.create_time()-old["created"]) < .01:
                raise RuntimeError("Existing controller owns experiment lock")
        except psutil.NoSuchProcess:
            pass
        path.unlink()
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["audit", "train", "backtest", "report", "all", "status", "worker"], default="all")
    parser.add_argument("--kind", choices=["fit", "account"])
    parser.add_argument("--names", default="")
    args = parser.parse_args()
    if args.stage == "status":
        print(json.dumps(read(OUT / "status.json") if (OUT / "status.json").exists() else dict(status="not_started")))
        return
    if args.stage == "worker":
        worker(args.kind, args.names.split(","))
        return
    with controller_lock():
        try:
            if args.stage in ("audit", "all"):
                if not (OUT / "audit.json").exists():
                    audit()
                else:
                    check_sources()
            else:
                check_sources()
            if args.stage in ("train", "all"):
                execute_queue("fit",fit_cutoffs())
                subprocess.run([sys.executable,"-u","-c", "from paper_futu.research_s6_2026 import assemble; assemble()"],
                               cwd=ROOT, env={**os.environ,**ENV}, check=True)
            if args.stage in ("backtest", "all"):
                if not (OUT / "streams.json").exists():
                    raise ValueError("Complete train stage first")
                for stream in read(OUT / "streams.json")["streams"].values():
                    if digest(stream["path"]) != stream["sha256"]:
                        raise ValueError("Serving stream hash changed")
                execute_queue("account",[f"{j['mode']}_{j['case']}_{j['offset']}" for j in account_jobs()])
            if args.stage in ("report", "all"):
                from paper_futu.s6_2026_report import build_report
                build_report()
            if args.stage != "all":
                print(f"STAGE COMPLETE {args.stage}", flush=True)
        except BaseException as error:
            previous = read(OUT / "status.json") if (OUT / "status.json").exists() else {}
            save(OUT / "status.json", {**previous, "status":"failed", "error":repr(error), "updated":time.time()})
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
