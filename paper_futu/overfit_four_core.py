"""Isolated, resumable protocol and training jobs for the four-strategy study."""
from __future__ import annotations

import copy
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paper_futu import final_ablation_core as frozen
from src.config import PROJECT_ROOT
from src.runtime_budget import require_memory

ROOT = PROJECT_ROOT
NAME = "overfit_four_asof_20261013"
OUT = ROOT / "reports/experiments" / NAME
MODELS = ROOT / "models/experiments" / NAME
SOURCE = frozen.REPORT
ALL = ROOT / "reports/experiments/all_full15_offsets_asof_20261013"
DATA = frozen.DATA
YEARS = tuple(range(2022, 2027))
# User explicitly reduced the launch gate on 2026-10-15; the per-task 2 GiB check remains.
MIN_START_GIB = 3.0
VARIANTS = {"V0": "original", "V1": "rolling_3y", "V2": "rolling_5y",
            "V3": "rolling_10y", "V4": "decay_756_sessions", "V5": "strong_l1",
            "V6": "strong_l2", "V7": "strong_l1_l2", "V8": "smaller_trees"}
FAMILIES = {
    "raw10": dict(spec="lgb_15_raw_rank", horizon=10, policy="fixed", input_kind="raw"),
    "rank20": dict(spec="lgb_15_rank_rank", horizon=20, policy="tuned", input_kind="rank"),
}
STRATEGIES = {
    "S2": dict(family="raw10", rebalance=5, portfolio=dict(top_k=20, top_tier_size=5, top_tier_weight=.25, n_drop=3)),
    "S4": dict(family="rank20", rebalance=10, portfolio=dict(top_k=10, top_tier_size=2, top_tier_weight=.5, n_drop=6)),
    "S5": dict(family="rank20", rebalance=10, portfolio=dict(top_k=10, top_tier_size=2, top_tier_weight=.6, n_drop=6)),
    "S6": dict(family="raw10", rebalance=5, portfolio=dict(top_k=20, top_tier_size=5, top_tier_weight=.25, n_drop=6)),
}
STYLE = ["book_to_price", "earnings_to_price", "volatility_60d", "mom_20d", "mom_60d"]


def save(path, obj):
    frozen.save_json(path, obj)


def read(path):
    return frozen.read_json(path)


def scientific_hash():
    return frozen.identity({str(p.relative_to(ROOT)): frozen.digest(p) for p in
        sorted((ROOT / "paper_futu").glob("overfit_four_*.py")) + [ROOT / "paper_futu/research_overfit_four.py"]})


def parameters(parent, variant):
    p = copy.deepcopy(parent)
    if variant in ("V5", "V7"):
        p["reg_alpha"] = max(1., p["reg_alpha"] * 10)
    if variant in ("V6", "V7"):
        p["reg_lambda"] *= 5
    if variant == "V8":
        p["num_leaves"] = max(2, p["num_leaves"] // 2)
        p["min_child_samples"] *= 3
    p["n_jobs"] = 4
    return p


def start_date(period_start, variant):
    return frozen.window_start(period_start, {"V1": 3, "V2": 5, "V3": 10}.get(variant, 0))


def calendar():
    return pd.DatetimeIndex(pd.to_datetime((DATA / "provider/calendars/day.txt").read_text().splitlines()))


def mature(frame, start, cutoff):
    # Slice before target construction/use, so future changes cannot enter a fit.
    result = frozen.mature(frame, start, cutoff)
    if result.index.has_duplicates or np.isinf(result[list(frozen.ENHANCED_FEATURE_NAMES)].to_numpy()).any():
        raise ValueError("Invalid supervised sample index or infinite features")
    return result


def sample_weights(training, cutoff, variant, sessions=None):
    if variant != "V4":
        return None
    sessions = calendar() if sessions is None else pd.DatetimeIndex(sessions)
    dates = training.index.get_level_values("datetime")
    positions = sessions.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Training dates missing from frozen trading calendar")
    age = sessions.searchsorted(pd.Timestamp(cutoff), side="right") - 1 - positions
    if (age < 0).any():
        raise ValueError("Future sample received a training weight")
    weight = np.exp2(-age / 756.)
    return weight / weight.mean()


def load_frame(family):
    config = FAMILIES[family]
    features = pd.read_parquet(DATA / f"{config['input_kind']}_features.parquet")
    labels = pd.read_parquet(DATA / f"labels_h{config['horizon']}.parquet")
    frame = features.join(labels)
    frame["target"] = frame.rank_target
    return frame


def init_qlib():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=str(DATA / "provider"), region=REG_CN, kernels=1,
        exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                     "kwargs": {"uri": (OUT / "mlruns").as_uri(), "default_exp_name": NAME}})


class Store:
    def __init__(self):
        self.protocol = read(OUT / "protocol.json")
        if self.protocol["implementation_sha256"] != scientific_hash():
            raise ValueError("Research implementation changed; audit and review before resuming")

    def key(self, job):
        return frozen.identity(dict(job=job, implementation=self.protocol["implementation_sha256"],
                                    snapshot=self.protocol["snapshot_id"]))[:24]

    def record(self, job, verify=True):
        path = OUT / "tasks" / f"{self.key(job)}.json"
        if not path.exists():
            return None
        record = read(path)
        if record["status"] != "complete":
            return None
        if verify:
            for name, expected in record["artifacts"].items():
                if not Path(name).exists() or frozen.digest(name) != expected:
                    raise ValueError(f"Changed cached artifact: {name}")
        return record

    def run(self, job, execute):
        cached = self.record(job)
        if cached:
            return cached["result"]
        key, started = self.key(job), time.time()
        path = OUT / "tasks" / f"{key}.json"
        record = dict(key=key, job=job, pid=os.getpid(), status="running", started=started)
        save(path, record)
        print(f"RUN {key} {job['kind']} {job.get('family', '')} {job.get('variant', '')}", flush=True)
        try:
            result, artifacts = execute(key)
            record.update(status="complete", elapsed=time.time()-started, result=result,
                          artifacts={str(p): frozen.digest(p) for p in artifacts})
            save(path, record)
            print(f"COMPLETE {key} {record['elapsed']:.1f}s", flush=True)
            return result
        except BaseException as error:
            record.update(status="failed", error=repr(error), elapsed=time.time()-started)
            save(path, record)
            raise


def verify_account(directory):
    """Independent NAV, fee and annual compounding reconciliation."""
    daily = pd.read_parquet(directory / "daily.parquet")
    orders = pd.read_parquet(directory / "orders.parquet")
    pnl = pd.read_parquet(directory / "contributions.parquet")
    holdings = pd.read_parquet(directory / "holdings.parquet")
    if not daily.index.is_unique or holdings.duplicated(["instrument", "datetime"]).any():
        raise ValueError("Duplicate account dates or holdings")
    fills = orders.loc[orders.notional.gt(0)]
    fees = fills.groupby("datetime").fee.sum().reindex(daily.index, fill_value=0.)
    gross = pnl.groupby("datetime").pnl_before_fees.sum().reindex(daily.index, fill_value=0.)
    np.testing.assert_allclose(1e6+(gross-fees).cumsum(), daily.account, atol=1e-4, rtol=1e-10)
    np.testing.assert_allclose((1+daily.net_return).cumprod()*1e6, daily.account, atol=1e-4, rtol=1e-10)
    np.testing.assert_allclose(fees, daily.fees_cash, atol=1e-5)
    if (daily.cash < -1e-5).any() or not np.isfinite(daily[["account", "net_return", "bench"]]).all().all():
        raise ValueError("Invalid cash or daily return")
    metrics = pd.read_csv(directory / "metrics.csv")
    expected = metrics.loc[metrics.period.eq("all")].iloc[0]
    cagr = (1+daily.net_return).prod() ** (252/len(daily))-1
    if not np.isclose(cagr, expected.annualized_return, atol=1e-10):
        raise ValueError("Account CAGR differs from published metrics")
    return dict(rows=len(daily), start=str(daily.index[0].date()), end=str(daily.index[-1].date()),
                annualized_return=float(cagr), fees=float(fees.sum()))


def verify_forward_labels(horizon):
    """Recompute labels from frozen binary opens, independently of expression code."""
    labels = pd.read_parquet(DATA / f"labels_h{horizon}.parquet")
    sessions = calendar()
    positions = sessions.get_indexer(labels.index.get_level_values("datetime"))
    ends = positions+horizon+1
    expected_ends = np.full(len(labels), np.datetime64("NaT"), dtype="datetime64[ns]")
    inside = ends < len(sessions)
    expected_ends[inside] = sessions.to_numpy()[ends[inside]]
    np.testing.assert_array_equal(labels.label_end_date.to_numpy(), expected_ends)
    if labels.loc[~inside, "label"].notna().any():
        raise ValueError("Incomplete future interval has a usable label")
    compared = 0
    for stock, group in labels.groupby(level="instrument", sort=False):
        raw = np.fromfile(DATA / "provider/features" / stock.lower() / "open.day.bin", dtype="<f4")
        first, prices = int(raw[0]), raw[1:]
        loc = sessions.get_indexer(group.index.get_level_values("datetime"))
        begin, end = loc+1-first, loc+horizon+1-first
        valid = group.label.notna().to_numpy()
        if ((begin[valid] < 0) | (end[valid] >= len(prices))).any():
            raise ValueError("Finite label reaches outside frozen opening prices")
        expected = prices[end[valid]]/prices[begin[valid]]-1
        np.testing.assert_allclose(group.label.to_numpy()[valid], expected, atol=2e-7, rtol=2e-6)
        compared += int(valid.sum())
    ranked = labels.label.groupby(level="datetime").rank(method="average", pct=True)-.5
    np.testing.assert_allclose(labels.rank_target, ranked, atol=1e-7, equal_nan=True)
    return dict(horizon=horizon, finite_labels_recomputed=compared)


def audit():
    old = read(ALL / "manifest.json")
    snapshot = read(DATA / "manifest.json")
    if frozen.code_hash() != old["scientific_code_sha256"]:
        raise ValueError("Frozen scientific source changed")
    for index, (name, expected) in enumerate(snapshot["artifacts"].items(), 1):
        if frozen.digest(DATA / name) != expected:
            raise ValueError(f"Frozen data artifact changed: {name}")
        if index % 2000 == 0:
            print(f"AUDIT snapshot {index}/{len(snapshot['artifacts'])}", flush=True)
    label_audits = [verify_forward_labels(h) for h in [10,20]]
    print("AUDIT exact opening-to-opening labels passed", flush=True)
    baselines, families, immutable = {}, {}, {}
    for family, config in FAMILIES.items():
        path = SOURCE / "predictions" / f"{config['spec']}_h{config['horizon']}_{config['policy']}_m12_s42.json"
        metadata = read(path)
        prediction = Path(metadata["prediction_path"])
        if frozen.digest(prediction) != metadata["prediction_sha256"]:
            raise ValueError("Baseline prediction hash mismatch")
        frame = pd.read_parquet(prediction)
        if frame.index.has_duplicates or not np.isfinite(frame.prediction).all():
            raise ValueError("Invalid baseline predictions")
        labels = pd.read_parquet(DATA / f"labels_h{config['horizon']}.parquet")
        if not frame.index.isin(labels.index).all():
            raise ValueError("Prediction universe differs from label snapshot")
        if (frame.train_label_end_max > frame.cutoff).any():
            raise ValueError("Baseline contains noncausal training labels")
        import lightgbm as lgb
        source_features = pd.read_parquet(DATA / f"{config['input_kind']}_features.parquet")
        for period in metadata["periods"]:
            sample = frame.loc[frame.execution_date.between(period["serving_start"], period["serving_end"])].iloc[::2000]
            loaded = lgb.Booster(model_file=period["model_path"])
            np.testing.assert_allclose(loaded.predict(source_features.reindex(sample.index), num_threads=4),
                                       sample.prediction, atol=1e-12, rtol=0.)
        del source_features, loaded
        families[family] = metadata
        immutable[str(path)] = frozen.digest(path)
        immutable[str(prediction)] = frozen.digest(prediction)
        for period in metadata["periods"]:
            model = Path(period["model_path"])
            if frozen.digest(model) != period["model_sha256"]:
                raise ValueError("Baseline fitted model changed")
            immutable[str(model)] = period["model_sha256"]
        for strategy, rule in STRATEGIES.items():
            if rule["family"] != family:
                continue
            matches = [c for c in old["candidates"] if c["spec"] == config["spec"]
                and c["horizon"] == config["horizon"] and c["policy"] == config["policy"]
                and c["months"] == 12 and c["seed"] == 42 and c["rebalance"] == rule["rebalance"]
                and c["portfolio"] == rule["portfolio"]]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous original strategy identity: {strategy}")
            candidate = matches[0]
            accounts = []
            for phase, case in zip(candidate["offsets"], candidate["case_ids"]):
                directory = ALL / "accounts" / case
                if not directory.exists():
                    directory = SOURCE / "accounts" / case
                result = verify_account(directory)
                if result["rows"] != 1154:
                    raise ValueError("Baseline does not cover the common interval")
                for file in directory.iterdir():
                    if file.is_file():
                        immutable[str(file)] = frozen.digest(file)
                accounts.append(dict(offset=phase, case_id=case, directory=str(directory), audit=result))
            baselines[strategy] = dict(candidate_key=candidate["candidate_key"], accounts=accounts)
        immutable[str(path.with_name(path.stem+"_daily_ic.csv"))] = frozen.digest(path.with_name(path.stem+"_daily_ic.csv")) if path.with_name(path.stem+"_daily_ic.csv").exists() else "unused"
    # Preserve the full frozen directory inventories, not only files touched here.
    inventories = {str(base): sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())
                   for base in [SOURCE, ROOT / "models/experiments/final_ablation_asof_20261013"]}
    metadata_source = ALL / "cross_model_diagnostic/manifest.json"
    banks = read(metadata_source)["bank_names"]
    immutable[str(metadata_source)] = frozen.digest(metadata_source)
    for file in [ALL/"manifest.json", ALL/"all_full15_phase_ranking.csv", DATA/"manifest.json"]:
        immutable[str(file)] = frozen.digest(file)
    protocol = dict(name=NAME, as_of="2026-10-13", snapshot_id=snapshot["snapshot_id"],
        scientific_source_sha256=old["scientific_code_sha256"], implementation_sha256=scientific_hash(),
        variants=VARIANTS, strategies=STRATEGIES, families=families, baselines=baselines,
        protected=frozen.protected_state(), immutable=immutable, frozen_inventories=inventories,
        bank_names=banks, main_accounts=180, seed_accounts=120, friction_accounts=40,
        maximum_fits=380, max_workers=2, threads_per_worker=4, min_start_gib=MIN_START_GIB,
        min_task_gib=2.0, resource_override="User authorized 3 GiB startup on 2026-10-15",
        limitations=frozen.LIMITATIONS, bootstraps=2000, bootstrap_blocks=[60,120],
        bootstrap_seed=20261015, created=time.time())
    prior = OUT / "protocol.json"
    if prior.exists():
        existing = read(prior)
        for name in ["snapshot_id", "strategies", "protected", "immutable"]:
            if existing[name] != protocol[name]:
                raise ValueError(f"Existing study protocol differs in {name}; do not overwrite its results")
        if existing["implementation_sha256"] == protocol["implementation_sha256"]:
            return existing
        completed = [p for p in (OUT / "tasks").glob("*.json") if read(p).get("status") == "complete"]
        if completed:
            raise ValueError("Implementation changed after completed jobs; review cache compatibility before resuming")
        save(OUT / "audit_history" / f"{existing['implementation_sha256']}.json", existing)
    save(prior, protocol)
    save(OUT / "audit.json", dict(status="passed", timestamp=time.time(), baselines=baselines, labels=label_audits))
    return protocol


def fold_jobs(protocol):
    jobs = {}
    for family in FAMILIES:
        for variant in list(VARIANTS)[1:]:
            for parent in protocol["families"][family]["periods"]:
                year = parent["selection_year"]
                p = parameters(parent["config"]["parameters"], variant)
                for validation_year in range(year-3, year):
                    job = dict(kind="fold", family=family, variant=variant, parameters=p,
                               validation_year=validation_year)
                    jobs[frozen.identity(job)] = job
    return list(jobs.values())


def fit_model(training, p, trees, seed, variant, cutoff):
    import lightgbm as lgb
    require_memory(2., "四策略研究拟合")
    model = lgb.LGBMRegressor(**{**p, "n_estimators": int(trees), "random_state": int(seed)})
    model.fit(training[list(frozen.ENHANCED_FEATURE_NAMES)], training.target,
              sample_weight=sample_weights(training, cutoff, variant))
    return model


def training_tail(training, model, trees):
    dates = training.index.get_level_values("datetime")
    recent = sorted(dates.unique())[-252:]
    tail = training.loc[dates.isin(recent)]
    prediction = pd.Series(model.predict(tail[list(frozen.ENHANCED_FEATURE_NAMES)], num_iteration=trees), index=tail.index)
    return frozen.daily_ic(prediction, tail.label)


def run_fold(job, key):
    frame = load_frame(job["family"])
    p, variant, year = job["parameters"], job["variant"], job["validation_year"]
    boundary = pd.Timestamp(year, 1, 1)
    cutoff = boundary-pd.Timedelta(days=1)
    train = mature(frame, start_date(boundary, variant), cutoff)
    valid = mature(frame, boundary, pd.Timestamp(year, 12, 31))
    trees = frozen.TREES[p["learning_rate"]]
    model = fit_model(train, p, max(trees), 42, variant, cutoff)
    records = []
    for n in trees:
        score = pd.Series(model.predict(valid[list(frozen.ENHANCED_FEATURE_NAMES)], num_iteration=n), index=valid.index)
        ic = frozen.daily_ic(score, valid.label)
        if not np.isfinite(ic.rank_ic.mean()):
            raise ValueError("No finite validation Rank IC")
        records.append(dict(parameter_id="paired", trees=n, num_leaves=p["num_leaves"],
            validation_year=year, mean_rank_ic=float(ic.rank_ic.mean()), mean_ic=float(ic.ic.mean()),
            positive_rank_ic_ratio=float(ic.rank_ic.dropna().gt(0).mean()), valid_days=int(ic.rank_ic.notna().sum()),
            train_rows=len(train), valid_rows=len(valid), train_start=str(start_date(boundary, variant).date()),
            train_label_end_max=str(train.label_end_date.max().date()),
            valid_label_end_max=str(valid.label_end_date.max().date())))
    path = OUT / "folds" / f"{key}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return dict(records=records, path=str(path)), [path]


def select_all(store):
    selections = []
    for family in FAMILIES:
        for variant in VARIANTS:
            for parent in store.protocol["families"][family]["periods"]:
                year = parent["selection_year"]
                p = parameters(parent["config"]["parameters"], variant)
                if variant == "V0":
                    trees = parent["trees"]
                else:
                    records = []
                    for validation_year in range(year-3, year):
                        job = dict(kind="fold", family=family, variant=variant,
                                   parameters=p, validation_year=validation_year)
                        result = store.record(job)
                        if not result:
                            raise ValueError("Complete all validation folds before selection")
                        records += result["result"]["records"]
                    ranked = frozen.choose_candidates(pd.DataFrame(records))
                    trees = int(ranked.iloc[0].trees)
                    ranked.insert(0, "family", family)
                    ranked.insert(1, "variant", variant)
                    ranked.insert(2, "execution_year", year)
                    path = OUT / "selections" / f"{family}_{variant}_{year}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ranked.to_csv(path, index=False)
                selections.append(dict(family=family, variant=variant, execution_year=year, parameters=p, trees=int(trees)))
    save(OUT / "selected_trees.json", selections)
    return selections


def final_jobs(protocol, selections):
    jobs = []
    for choice in selections:
        family, variant, year = choice["family"], choice["variant"], choice["execution_year"]
        seeds = [42] if variant != "V0" else []
        if variant in ("V0", "V2", "V7"):
            seeds += [2024, 2025]
        for seed in seeds:
            jobs.append(dict(kind="fit", **choice, seed=seed))
    if len(jobs) != 140:
        raise ValueError("Unexpected fit scope")
    return jobs


def run_fit(job, key):
    import lightgbm as lgb
    family, variant, year, seed = job["family"], job["variant"], job["execution_year"], job["seed"]
    frame = load_frame(family)
    parent = next(p for p in read(OUT / "protocol.json")["families"][family]["periods"] if p["selection_year"] == year)
    cutoff = pd.Timestamp(parent["cutoff"])
    start = start_date(pd.Timestamp(year, 1, 1), variant)
    training = mature(frame, start, cutoff)
    model = fit_model(training, job["parameters"], job["trees"], seed, variant, cutoff)
    directory = MODELS / key
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.txt"
    model.booster_.save_model(str(model_path))
    loaded = lgb.Booster(model_file=str(model_path))
    original = pd.read_parquet(read(OUT / "protocol.json")["families"][family]["prediction_path"])
    mask = original.execution_date.between(parent["serving_start"], parent["serving_end"])
    old = original.loc[mask]
    serving = frame.reindex(old.index)
    values = loaded.predict(serving[list(frozen.ENHANCED_FEATURE_NAMES)], num_threads=4)
    np.testing.assert_allclose(values, model.predict(serving[list(frozen.ENHANCED_FEATURE_NAMES)]), atol=1e-12, rtol=0.)
    if not np.isfinite(values).all():
        raise ValueError("Invalid serving predictions")
    out = pd.DataFrame(dict(prediction=values, execution_date=old.execution_date), index=old.index)
    out["cutoff"], out["train_label_end_max"] = cutoff, training.label_end_date.max()
    path = OUT / "prediction_parts" / f"{key}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    tail = training_tail(training, loaded, job["trees"])
    tail_path = path.with_name(key+"_train_tail.parquet")
    tail.to_parquet(tail_path)
    importance = directory / "importance.csv"
    pd.DataFrame(dict(factor=list(frozen.ENHANCED_FEATURE_NAMES), gain=loaded.feature_importance("gain"),
                      split=loaded.feature_importance("split"))).to_csv(importance, index=False)
    result = dict(path=str(path), model_path=str(model_path), model_sha256=frozen.digest(model_path),
        train_rows=len(training), train_start=str(start.date()), train_signal_end=str(training.index.get_level_values("datetime").max().date()),
        label_end_max=str(training.label_end_date.max().date()), cutoff=str(cutoff.date()),
        train_tail_path=str(tail_path), **job)
    del model, loaded, training, frame
    gc.collect()
    return result, [path, model_path, importance, tail_path]


def assemble_streams(store, jobs):
    streams = []
    for family in FAMILIES:
        for variant in VARIANTS:
            seeds = [42] + ([2024, 2025] if variant in ("V0", "V2", "V7") else [])
            for seed in seeds:
                descriptor = dict(family=family, variant=variant, seed=seed)
                if variant == "V0" and seed == 42:
                    streams.append(dict(**descriptor, prediction_path=store.protocol["families"][family]["prediction_path"]))
                    continue
                pieces = []
                periods = []
                for job in jobs:
                    if all(job[k] == v for k, v in descriptor.items()):
                        record = store.record(job)
                        if not record:
                            raise ValueError("Incomplete annual fit/serve")
                        periods.append(record["result"])
                        pieces.append(pd.read_parquet(record["result"]["path"]))
                prediction = pd.concat(pieces).sort_index()
                original = pd.read_parquet(store.protocol["families"][family]["prediction_path"], columns=[])
                if not prediction.index.equals(original.index):
                    raise ValueError("Variant changed the comparison stock/date sample")
                path = OUT / "predictions" / f"{family}_{variant}_s{seed}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                prediction.to_parquet(path)
                streams.append(dict(**descriptor, prediction_path=str(path), periods=periods))
    if len(streams) != 30:
        raise ValueError("Expected 18 primary plus 12 extra-seed streams")
    save(OUT / "streams.json", streams)
    return streams


def account_jobs(protocol, streams):
    jobs = []
    for stream in streams:
        for strategy, rule in STRATEGIES.items():
            if rule["family"] != stream["family"]:
                continue
            for offset in frozen.offsets(rule["rebalance"]):
                for friction in ([0., .001, .002] if stream["variant"] == "V0" and stream["seed"] == 42 else [0.]):
                    jobs.append(dict(kind="account", strategy=strategy, variant=stream["variant"], seed=stream["seed"],
                        family=stream["family"], prediction_path=stream["prediction_path"], offset=offset,
                        friction=friction, portfolio=rule["portfolio"], rebalance=rule["rebalance"]))
    if len(jobs) != 340:
        raise ValueError("Account scope exceeded or incomplete")
    return jobs


def verify_protected(protocol, full_snapshot=False):
    if frozen.code_hash() != protocol["scientific_source_sha256"] or frozen.protected_state() != protocol["protected"]:
        raise ValueError("Protected scientific source or trading state changed")
    for name, expected in protocol["immutable"].items():
        if expected != "unused" and frozen.digest(name) != expected:
            raise ValueError(f"Immutable source changed: {name}")
    for base, expected in protocol["frozen_inventories"].items():
        actual = sorted(str(p.relative_to(base)) for p in Path(base).rglob("*") if p.is_file())
        if expected != actual:
            raise ValueError(f"Frozen directory inventory changed: {base}")
    if full_snapshot:
        for name, expected in read(DATA / "manifest.json")["artifacts"].items():
            if frozen.digest(DATA / name) != expected:
                raise ValueError(f"Snapshot changed: {name}")
