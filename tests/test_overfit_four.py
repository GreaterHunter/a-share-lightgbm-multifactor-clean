"""Scientific invariants of the isolated four-strategy diagnostic study."""
import copy

import numpy as np
import pandas as pd
import pytest

from paper_futu import overfit_four_core as core
from paper_futu.overfit_four_diagnostics import paired_bootstrap, residual_correlation


def test_variants_are_paired_and_do_not_mutate_parent():
    parent = dict(core.frozen.BASE)
    saved = copy.deepcopy(parent)
    assert core.parameters(parent, "V5")["reg_alpha"] == 1
    assert core.parameters(parent, "V5")["reg_lambda"] == 1
    assert core.parameters(parent, "V6")["reg_lambda"] == 5
    assert core.parameters(parent, "V6")["reg_alpha"] == .1
    assert core.parameters(parent, "V7")["reg_alpha"] == 1
    assert core.parameters(parent, "V7")["reg_lambda"] == 5
    smaller = core.parameters(parent, "V8")
    assert smaller["num_leaves"] == 15 and smaller["min_child_samples"] == 300
    assert parent == saved
    p22 = {**parent, "reg_alpha": 0., "reg_lambda": 20., "num_leaves": 15, "min_child_samples": 1000}
    assert core.parameters(p22, "V7")["reg_lambda"] == 100
    assert core.parameters(p22, "V8")["num_leaves"] == 7


def test_weight_half_life_normalization_and_session_age():
    sessions = pd.bdate_range("2018-01-01", periods=800)
    index = pd.MultiIndex.from_arrays([["A", "B"], sessions[[0, 756]]], names=["instrument", "datetime"])
    training = pd.DataFrame(index=index)
    weights = core.sample_weights(training, sessions[-1], "V4", sessions)
    assert weights.mean() == pytest.approx(1)
    assert weights[1]/weights[0] == pytest.approx(2)
    assert core.sample_weights(training, sessions[-1], "V1", sessions) is None
    assert core.start_date("2025-01-01", "V1") == pd.Timestamp("2022-01-01")
    assert core.start_date("2025-01-01", "V3") == pd.Timestamp("2015-01-01")


def test_mature_samples_are_invariant_to_future_values():
    index = pd.MultiIndex.from_product([["A"], pd.to_datetime(["2020-12-01", "2020-12-02", "2021-01-04"])], names=["instrument", "datetime"])
    frame = pd.DataFrame(1., index=index, columns=list(core.frozen.ENHANCED_FEATURE_NAMES))
    frame["label"] = [.1, .2, np.nan]
    frame["target"] = [.1, .2, np.nan]
    frame["label_end_date"] = pd.to_datetime(["2020-12-31", "2021-01-04", None])
    a = core.mature(frame, "2010-01-01", "2020-12-31")
    frame.loc[index[1:], ["label", "target", *core.frozen.ENHANCED_FEATURE_NAMES]] = 999
    b = core.mature(frame, "2010-01-01", "2020-12-31")
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 1


def fake_protocol():
    families = {}
    for family in core.FAMILIES:
        families[family] = dict(periods=[dict(selection_year=y, config=dict(parameters=dict(core.frozen.BASE)), trees=40) for y in core.YEARS])
    return dict(families=families)


def test_budget_and_shared_fits_four_rules():
    protocol = fake_protocol()
    folds = core.fold_jobs(protocol)
    # Identical annual hyperparameters share seven unique validation years.
    assert len(folds) == 2*8*7
    choices = [dict(family=f, variant=v, execution_year=y, parameters=dict(core.frozen.BASE), trees=40)
               for f in core.FAMILIES for v in core.VARIANTS for y in core.YEARS]
    assert len(core.final_jobs(protocol, choices)) == 140
    streams = [dict(family=f, variant=v, seed=s, prediction_path="dummy")
               for f in core.FAMILIES for v in core.VARIANTS
               for s in ([42,2024,2025] if v in ("V0","V2","V7") else [42])]
    jobs = core.account_jobs(protocol, streams)
    assert len(jobs) == 340
    assert sum(j["seed"] == 42 and j["friction"] == 0 for j in jobs) == 180
    assert sum(j["friction"] > 0 for j in jobs) == 40
    assert core.STRATEGIES["S2"]["portfolio"]["n_drop"] == 3
    assert core.STRATEGIES["S6"]["portfolio"]["n_drop"] == 6


def test_style_projection_removes_known_rank_exposure():
    n = 100
    x = np.arange(n, dtype=float)
    group = pd.DataFrame(dict(prediction=x, label=x, bank=(x>50).astype(float)))
    for i, field in enumerate(core.STYLE):
        group[field] = x/(n+i+1)
    result = residual_correlation(group)
    assert result["style_subset_rank_ic"] == pytest.approx(1)
    assert np.isnan(result["style_residual_rank_ic"])
    assert result["style_coverage"] == 1
    group.loc[:49, core.STYLE[0]] = np.nan
    result = residual_correlation(group)
    assert result["style_subset_size"] == 50
    assert result["style_coverage"] == .5


def test_constant_and_missing_residual_inputs_are_unavailable():
    group = pd.DataFrame({c: np.ones(40) for c in ["prediction", "label", "bank", *core.STYLE]})
    assert np.isnan(residual_correlation(group)["style_subset_rank_ic"])
    assert np.isnan(residual_correlation(group.iloc[:29])["style_residual_rank_ic"])


def test_bootstrap_is_paired_and_measures_mean_phase_cagr():
    rng = np.random.default_rng(42)
    a = rng.normal(.0002, .01, (250, 5))
    same = paired_bootstrap(a, a, 60, repeats=120, cagr=True)
    assert same["point_delta"] == 0 and same["ci_2_5"] == 0 and same["ci_97_5"] == 0
    b = a-.0001
    actual = paired_bootstrap(a, b, 60, repeats=120, cagr=True)
    expected = ((np.prod(1+a,axis=0)**(252/250)-1)-(np.prod(1+b,axis=0)**(252/250)-1)).mean()
    assert actual["point_delta"] == pytest.approx(expected)
    assert actual == paired_bootstrap(a, b, 60, repeats=120, cagr=True)
    with pytest.raises(ValueError):
        paired_bootstrap(a[:20], b[:20], 60)


def test_new_account_adapter_does_not_rebind_frozen_globals():
    from paper_futu import final_ablation_account as original
    report, data, store = original.REPORT, original.DATA, original.Store
    from paper_futu import overfit_four_account
    assert original.REPORT == report and original.DATA == data and original.Store is store
    assert overfit_four_account.OUT != original.REPORT


def test_verified_cache_rejects_modified_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "OUT", tmp_path)
    monkeypatch.setattr(core, "scientific_hash", lambda: "test-implementation")
    core.save(tmp_path/"protocol.json", dict(implementation_sha256="test-implementation", snapshot_id="test-data"))
    store = core.Store()
    artifact = tmp_path/"example.txt"
    artifact.write_text("original", encoding="utf-8")
    job = dict(kind="unit-test")
    store.run(job, lambda key: (dict(value=1), [artifact]))
    assert store.record(job)["result"]["value"] == 1
    artifact.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed cached artifact"):
        store.record(job)


def test_controller_lock_rejects_duplicate_run(tmp_path, monkeypatch):
    from paper_futu.research_overfit_four import controller_lock
    monkeypatch.setattr(core, "OUT", tmp_path)
    with controller_lock():
        with pytest.raises((RuntimeError, BlockingIOError)):
            with controller_lock():
                pytest.fail("A duplicate controller acquired the same lock")
    with controller_lock():
        pass


@pytest.fixture
def toy_fit_memory(monkeypatch):
    # Keep the production guard and real LightGBM fit; isolate the OS probe for
    # tiny synthetic fixtures from unrelated applications on the test machine.
    from src import runtime_budget
    monkeypatch.setattr(runtime_budget, "available_memory_bytes", lambda: 3 * 1024**3)


def test_real_training_guard_still_rejects_low_memory(monkeypatch):
    from src import runtime_budget
    monkeypatch.setattr(runtime_budget, "available_memory_bytes", lambda: 1024**3)
    with pytest.raises(RuntimeError, match=r"2\.0 GiB"):
        core.require_memory(2.0, "training test")


def test_small_real_lightgbm_fit_reloads_identically(tmp_path, monkeypatch, toy_fit_memory):
    import lightgbm as lgb
    rng = np.random.default_rng(123)
    sessions = pd.bdate_range("2020-01-01", periods=60)
    monkeypatch.setattr(core, "calendar", lambda: sessions)
    dates = sessions[:30]
    index = pd.MultiIndex.from_product([[f"S{i}" for i in range(40)], dates], names=["instrument", "datetime"])
    frame = pd.DataFrame(rng.normal(size=(len(index), 15)), index=index, columns=core.frozen.ENHANCED_FEATURE_NAMES)
    frame["target"] = frame.iloc[:, 0]*.1+rng.normal(0,.01,len(frame))
    frame["label"] = frame.target
    model = core.fit_model(frame, {**core.frozen.BASE, "min_child_samples": 20}, 10, 42, "V4", dates[-1])
    path = tmp_path/"model.txt"
    model.booster_.save_model(str(path))
    loaded = lgb.Booster(model_file=str(path))
    values = frame[list(core.frozen.ENHANCED_FEATURE_NAMES)]
    np.testing.assert_allclose(model.predict(values), loaded.predict(values,num_threads=4), atol=1e-12,rtol=0)


def test_annual_fit_serves_exact_source_dates_and_writes_causal_metadata(tmp_path, monkeypatch, toy_fit_memory):
    rng = np.random.default_rng(7)
    sessions = pd.bdate_range("2021-09-01", "2022-04-01")
    monkeypatch.setattr(core, "calendar", lambda: sessions)
    dates = sessions[(sessions >= "2021-10-01") & (sessions <= "2022-02-28")]
    index = pd.MultiIndex.from_product([[f"S{i:02d}" for i in range(40)], dates], names=["instrument", "datetime"])
    frame = pd.DataFrame(rng.normal(size=(len(index),15)),index=index,columns=core.frozen.ENHANCED_FEATURE_NAMES)
    frame["target"] = frame.iloc[:,0]*.1
    frame["label"] = frame.target
    positions = sessions.get_indexer(index.get_level_values("datetime"))
    frame["label_end_date"] = sessions.to_numpy()[positions+11]
    predicted = frame.loc[frame.index.get_level_values("datetime") >= "2022-01-04", []].copy().sort_index()
    predicted["prediction"] = 0.
    predicted["execution_date"] = sessions.to_numpy()[sessions.get_indexer(predicted.index.get_level_values("datetime"))+1]
    source = tmp_path/"original.parquet"
    predicted.to_parquet(source)
    parent = dict(selection_year=2022, cutoff="2021-12-31", serving_start="2022-01-04", serving_end="2022-03-01")
    monkeypatch.setattr(core,"OUT",tmp_path/"study")
    monkeypatch.setattr(core,"MODELS",tmp_path/"models")
    monkeypatch.setattr(core,"load_frame",lambda family:frame.copy())
    core.save(core.OUT/"protocol.json",dict(families=dict(raw10=dict(prediction_path=str(source),periods=[parent]))))
    job=dict(kind="fit",family="raw10",variant="V4",execution_year=2022,seed=42,
             parameters={**core.frozen.BASE,"min_child_samples":20},trees=10)
    result,artifacts=core.run_fit(job,"toy-annual")
    out=pd.read_parquet(result["path"])
    pd.testing.assert_index_equal(out.index,predicted.index)
    assert out.execution_date.equals(predicted.execution_date)
    assert out.train_label_end_max.le(out.cutoff).all()
    assert all(path.exists() for path in artifacts)
