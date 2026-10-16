"""Combined constraints, unchanged execution arithmetic, and historical cutoffs."""
import ast
import copy
import inspect
import pandas as pd
import pytest

from paper_futu import research_s6_quarterly_regularized as experiment
from paper_futu import s6_2026_account as original_account
from paper_futu import s6_explicit_account as explicit_account


def test_combined_parameters_change_only_predeclared_constraints():
    base=dict(learning_rate=.03,num_leaves=31,min_child_samples=100,reg_alpha=.1,reg_lambda=1.,n_jobs=4,
        subsample=.8,colsample_bytree=.8)
    saved=copy.deepcopy(base)
    new=experiment.parameters(base)
    assert base==saved
    assert {k for k in base if new[k]!=base[k]}=={"num_leaves","min_child_samples","reg_alpha","reg_lambda"}
    assert new["num_leaves"]==15 and new["min_child_samples"]==300
    assert new["reg_alpha"]==1 and new["reg_lambda"]==5


def test_explicit_context_adapter_has_identical_replay_body():
    original=ast.parse(inspect.getsource(original_account.replay)).body[0]
    copied=ast.parse(inspect.getsource(explicit_account.replay)).body[0]
    assert copied.args.args[0].arg=="context"
    copied.args.args=copied.args.args[1:]
    # The only new statement binds the explicit context to a local name.
    assignment=copied.body.pop(1)
    assert isinstance(assignment,ast.Assign)
    assert ast.dump(assignment,include_attributes=False)==ast.dump(ast.parse("study = context").body[0],include_attributes=False)
    assert ast.dump(original,include_attributes=False)==ast.dump(copied,include_attributes=False)


def test_all_years_have_four_quarterly_fits_including_initial_model(monkeypatch):
    dates=pd.bdate_range("2021-12-01","2026-10-13")
    # Exclude the synthetic calendar's first three January days so the initial
    # execution session follows a previous-year cutoff, like this experiment.
    dates=dates[~((dates.month==1)&(dates.day<=3))]
    monkeypatch.setattr(experiment.parent,"calendar",lambda:dates)
    jobs=experiment.fit_jobs()
    assert len(jobs)==20
    assert len({j["cutoff"] for j in jobs})==20
    for year in range(2022,2027):
        year_jobs=[j for j in jobs if j["year"]==year]
        assert len(year_jobs)==4
        assert year_jobs[0]["cutoff"].startswith(str(year-1))
        assert all(j["cutoff"]<j["start"] for j in year_jobs)


@pytest.mark.skipif(not (experiment.DATA/"financial_events.parquet").exists(),reason="Requires the local frozen financial snapshot")
def test_disclosure_audit_uses_actual_snapshot_and_reports_missing_data():
    audit=experiment.financial_audit()
    latest=audit["coverage"][-1]
    assert audit["as_of"]=="2026-10-13"
    assert audit["max_report_date"]=="2026-06-30"
    assert latest["pool_stocks"]==latest["aligned_rows"]==300
    assert latest["general_reports"]["2026-06-30 00:00:00"]==290
    assert len(latest["missing_general_stocks"])==10
    assert audit["data_updated_for_experiment"] is False
