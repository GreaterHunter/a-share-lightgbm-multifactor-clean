"""Scope changes must preserve science keys and exclude factor-deletion candidates."""
from copy import deepcopy
import pytest

from paper_futu import resume_full_factor_ablation as resume
from paper_futu.final_ablation_core import A, Spec, code_hash


def test_scope_removes_only_future_factor_deletion_and_preserves_history():
    raw = Spec("lgb_15_raw_rank",A.features,input_kind="raw").record()
    deleted = Spec("without_book_to_price",tuple(f for f in A.features if f != "book_to_price")).record()
    old = dict(selected=[A.record(),raw,deleted],all_specs=[A.record(),raw,deleted],historical="retained")
    copy = deepcopy(old)
    new = resume.scoped_selection(old)
    assert [r["name"] for r in new["selected"]] == list(resume.ALLOWED_NAMES)
    assert new["all_specs"] == old["all_specs"]
    assert old == copy
    with pytest.raises(ValueError,match="outside"):
        resume.validate_spec(deleted)


def test_full_named_model_with_deleted_feature_is_rejected():
    bad = {**A.record(),"features":A.features[:-1]}
    with pytest.raises(ValueError,match="outside"):
        resume.validate_spec(bad)


def test_resume_entry_does_not_change_scientific_code_hash():
    assert code_hash() == resume.SCIENTIFIC_HASH
    assert resume.STAGES == ["portfolio","cadence","robustness","report"]


def test_runtime_resume_budget_has_two_specs_and_skips_model_and_tune(monkeypatch):
    for name in ["STAGES","MAX_SELECTED_SPECS","SCOPE_REVISION","selected_specs","save_json",
                 "stage_portfolio","stage_cadence","stage_robustness","stage_report"]:
        monkeypatch.setattr(resume.runner,name,getattr(resume.runner,name))
    resume.configure_runner()
    counts = resume.runner.search_counts()
    assert counts["model_streams_max"] == 4
    assert counts["coarse_replays_max"] == 160
    assert counts["fine_replays_max"] == 1408
    assert counts["parameter_configs"] == 24
    assert counts["horizon_rebalance_pairs"] == 20
    assert resume.runner.STAGES == resume.STAGES
