import pandas as pd
import pytest
from paper_futu import research_s6_h5_daily_stop13 as experiment


def test_quarterly_execution_has_preceding_session_cutoff_and_covers_all_days():
    calendar=pd.bdate_range("2021-12-31","2026-10-13")
    periods=experiment.periods(calendar)
    assert len(periods)==20
    covered=[]
    for period in periods:
        first=pd.Timestamp(period["start"])
        assert pd.Timestamp(period["cutoff"])==calendar[calendar.get_loc(first)-1]
        covered.extend(calendar[(calendar>=first)&(calendar<=period["end"])])
    assert covered==list(calendar[(calendar>=experiment.START)&(calendar<=experiment.END)])


def test_only_labels_realized_by_cutoff_are_training_samples():
    dates=pd.to_datetime(["2025-12-20","2025-12-23","2025-12-24","2026-01-01"])
    index=pd.MultiIndex.from_arrays([dates,["A"]*4],names=["datetime","instrument"])
    frame=pd.DataFrame(dict(label=[.1,.2,.3,.4],target=[0.,.1,.2,.3],
        label_end_date=pd.to_datetime(["2025-12-30","2025-12-31","2026-01-02","2026-01-09"])),index=index)
    before=experiment.mature(frame,"2010-01-01","2025-12-31")
    assert len(before)==2
    frame.loc[frame.label_end_date>pd.Timestamp("2025-12-31"),["label","target"]]=999.
    pd.testing.assert_frame_equal(before,experiment.mature(frame,"2010-01-01","2025-12-31"))


def test_fixed_weights_and_regularization():
    p=experiment.PORTFOLIO
    assert p["top_k"]==10 and p["n_drop"]==6
    assert p["top_tier_weight"]/p["top_tier_size"]==pytest.approx(.15)
    assert (1-p["top_tier_weight"])/(p["top_k"]-p["top_tier_size"])==pytest.approx(.4/6)
    assert p["risk_degree"]==.95
    assert experiment.PARAMS["reg_alpha"]==1 and experiment.PARAMS["reg_lambda"]==5
    assert experiment.PARAMS["num_leaves"]==15 and experiment.PARAMS["min_child_samples"]==300


def test_execution_year_cannot_be_added_as_validation_fold():
    with pytest.raises(ValueError): experiment.fold(2026)
