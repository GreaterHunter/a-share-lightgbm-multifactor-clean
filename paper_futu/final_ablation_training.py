"""Chronological annual selection and calendar-based fresh fits."""
from __future__ import annotations

from dataclasses import asdict
import gc
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from paper_futu.final_ablation_core import (
    AS_OF, BASE, DATA, MODELS, REPORT, Spec, Store, TREES, choose_candidates,
    daily_ic, digest, init_qlib, mature, parameter_configs, read_json, serving_periods,
    window_start,
)
from src.runtime_budget import require_memory


def load_frame(spec, horizon):
    features = pd.read_parquet(DATA / f"{spec.input_kind}_features.parquet", columns=list(spec.features))
    labels = pd.read_parquet(DATA / f"labels_h{horizon}.parquet")
    frame = features.join(labels)
    frame["target"] = frame["rank_target"] if spec.target_kind == "rank" else frame["label"]
    return frame


def make_model(spec, parameters, trees, seed):
    if spec.engine == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=1.))
    return lgb.LGBMRegressor(**{**parameters, "n_estimators": trees, "random_state": seed})


def selection_fold(store, frame, spec, horizon, config, validation_year):
    parameters = config["parameters"]
    trees = TREES[parameters["learning_rate"]] if spec.engine != "ridge" else [1]
    task = dict(spec=spec.record(), horizon=horizon, parameter=config,
                validation_year=validation_year, candidates=trees)

    def execute(key):
        require_memory(2.0, "单折LightGBM拟合")
        import psutil
        p = pd.Timestamp(validation_year, 1, 1)
        train = mature(frame, window_start(p, spec.window), p - pd.Timedelta(days=1))
        valid = mature(frame, p, pd.Timestamp(validation_year, 12, 31))
        model = make_model(spec, parameters, max(trees), 42)
        model.fit(train[list(spec.features)], train.target)
        records = []
        for n in trees:
            values = model.predict(valid[list(spec.features)], num_iteration=n) if spec.engine == "lightgbm" else model.predict(valid[list(spec.features)])
            score = pd.Series(values, index=valid.index)
            ic = daily_ic(score, valid.label)
            average = float(ic.rank_ic.mean())
            if not np.isfinite(average):
                raise ValueError("Validation fold has no finite Rank IC")
            records.append(dict(parameter_id=config["id"], trees=n, num_leaves=parameters["num_leaves"],
                                validation_year=validation_year, mean_rank_ic=average,
                                mean_ic=float(ic.ic.mean()), valid_days=int(ic.rank_ic.notna().sum()),
                                train_rows=len(train), valid_rows=len(valid),
                                train_start=str(window_start(p, spec.window).date()),
                                train_cutoff=str((p-pd.Timedelta(days=1)).date()),
                                train_label_end_max=str(train.label_end_date.max().date()),
                                valid_label_end_max=str(valid.label_end_date.max().date())))
        path = REPORT / "folds" / f"{key}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(path, index=False)
        memory = psutil.Process().memory_info().rss / 2**30
        del train, valid, model
        gc.collect()
        return records, [path], dict(max_candidate_rank_ic=max(r["mean_rank_ic"] for r in records), rss_gib=memory)
    return store.run("validation_fold", task, execute)


def selection(store, frame, spec, horizon, policy, year):
    configs = parameter_configs()
    if policy == "fixed" or spec.engine == "ridge":
        candidates = configs[:1]
    elif horizon == 20:
        candidates = configs
    else:
        h20_path = REPORT / "selections" / f"{spec.name}_h20_tuned_{year}.csv"
        if not h20_path.exists():
            raise FileNotFoundError("Complete the H20 parameter search before other horizons")
        rankings = pd.read_csv(h20_path).drop_duplicates("parameter_id").head(3)
        wanted = set(rankings.parameter_id) | {"p00"}
        candidates = [c for c in configs if c["id"] in wanted]
    records = []
    for config in candidates:
        for v in range(year - 3, year):
            records += selection_fold(store, frame, spec, horizon, config, v)
    ranked = choose_candidates(pd.DataFrame(records))
    path = REPORT / "selections" / f"{spec.name}_h{horizon}_{policy}_{year}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(path, index=False)
    row = ranked.iloc[0]
    chosen = next(c for c in candidates if c["id"] == row.parameter_id)
    return chosen, int(row.trees)


def fit_stream(spec, horizon, policy="fixed", months=12, seed=42):
    init_qlib()
    from qlib.data import D
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    store = Store()
    frame = load_frame(spec, horizon)
    choices = {y: selection(store, frame, spec, horizon, policy, y) for y in range(2022, 2027)}
    parts, metadata = [], []
    for period in serving_periods(calendar, months):
        year = period["year"]
        config, trees = choices[year]
        start = window_start(period["period_start"], spec.window)
        cutoff = period["cutoff"]
        # Fitting identity excludes serving frequency: coincident cutoffs share models.
        fit_config = dict(spec=spec.record(), horizon=horizon, config=config, trees=trees,
                          seed=seed, train_start=str(start.date()), cutoff=str(cutoff.date()))

        def fit(key):
            require_memory(2., "最终模型拟合")
            training = mature(frame, start, cutoff)
            model = make_model(spec, config["parameters"], trees, seed)
            model.fit(training[list(spec.features)], training.target)
            directory = MODELS / key
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("model.txt" if spec.engine == "lightgbm" else "model.joblib")
            if spec.engine == "lightgbm":
                model.booster_.save_model(str(path))
                loaded = lgb.Booster(model_file=str(path))
                importance = pd.DataFrame(dict(factor=list(spec.features),
                    gain=model.booster_.feature_importance("gain"), split=model.booster_.feature_importance("split")))
                importance.to_csv(directory / "importance.csv", index=False)
                extra = [directory / "importance.csv"]
            else:
                joblib.dump(model, path)
                loaded = joblib.load(path)
                extra = []
            check = training[list(spec.features)].iloc[::max(1, len(training)//2000)]
            if not np.allclose(model.predict(check), loaded.predict(check), atol=1e-12, rtol=0):
                raise ValueError("Reloaded model predictions changed")
            result = dict(model_path=str(path), model_sha256=digest(path), train_rows=len(training),
                          train_signal_end=str(training.index.get_level_values("datetime").max().date()),
                          label_end_max=str(training.label_end_date.max().date()), **fit_config)
            del training, model, loaded
            gc.collect()
            return result, [path, *extra], dict(train_rows=result["train_rows"], trees=trees)

        fitted = store.run("final_fit", fit_config, fit)
        trade_dates = calendar[(calendar >= period["execution_start"]) & (calendar <= period["execution_end"])]
        signal_dates = calendar[calendar.get_indexer(trade_dates)-1]
        signals = frame.loc[frame.index.get_level_values("datetime").isin(signal_dates), list(spec.features)]
        serving = {**fitted, "serving_start": str(period["execution_start"].date()),
                   "serving_end": str(period["execution_end"].date()), "selection_year": year, "months": months}

        def predict(key):
            loaded = lgb.Booster(model_file=fitted["model_path"]) if spec.engine == "lightgbm" else joblib.load(fitted["model_path"])
            values = loaded.predict(signals, num_threads=4) if spec.engine == "lightgbm" else loaded.predict(signals)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite predictions")
            out = pd.DataFrame({"prediction": values}, index=signals.index)
            out["execution_date"] = out.index.get_level_values("datetime").map(dict(zip(signal_dates, trade_dates)))
            out["model_sha256"] = fitted["model_sha256"]
            out["cutoff"] = cutoff
            out["train_signal_start"] = start
            out["train_label_end_max"] = pd.Timestamp(fitted["label_end_max"])
            out["model_effective_start"] = period["execution_start"]
            out["model_effective_end"] = period["execution_end"]
            out["parameter_selection_year"] = year
            path = REPORT / "prediction_parts" / f"{key}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path)
            del loaded
            return dict(path=str(path), **serving), [path], dict(prediction_rows=len(out))
        served = store.run("serve", serving, predict)
        parts.append(pd.read_parquet(served["path"]))
        metadata.append(served)
    combined = pd.concat(parts).sort_index()
    if combined.index.has_duplicates or not np.isfinite(combined.prediction).all():
        raise ValueError("Duplicate or invalid stitched predictions")
    stream = f"{spec.name}_h{horizon}_{policy}_m{months}_s{seed}"
    path = REPORT / "predictions" / f"{stream}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path)
    ic = daily_ic(combined.prediction, frame.label.reindex(combined.index))
    ic.to_csv(path.with_name(f"{stream}_daily_ic.csv"), encoding="utf-8-sig")
    from paper_futu.final_ablation_core import save_json
    save_json(path.with_suffix(".json"), dict(stream=stream, spec=spec.record(), horizon=horizon,
              policy=policy, months=months, seed=seed, prediction_path=str(path), prediction_sha256=digest(path),
              periods=metadata, mean_rank_ic=float(ic.rank_ic.mean()), valid_ic_days=int(ic.rank_ic.notna().sum())))
    print(f"STREAM {stream}: {len(combined):,} predictions", flush=True)
    return path
