"""Independent completion verification for the one combined S6 setting."""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from paper_futu import research_s6_quarterly_regularized as experiment
from paper_futu import overfit_four_core as core


def main():
    p=experiment.check_sources()
    done=experiment.read(experiment.OUT/"completion.json")
    assert done["status"]=="complete"
    assert done["manifest_sha256"]==experiment.digest(experiment.OUT/"report_manifest.json")
    manifest=experiment.read(experiment.OUT/"report_manifest.json")
    for path,expected in manifest["artifacts"].items():
        assert experiment.digest(path)==expected,path
    expected_parameters={"reg_alpha":1.,"reg_lambda":5.,"num_leaves":15,"min_child_samples":300}
    for period in p["periods"]:
        model=experiment.cached("fit",period["cutoff"])
        assert model["label_end_max"]<=period["cutoff"]<period["start"]
        assert all(model["parameters"][k]==v for k,v in expected_parameters.items())
        assert model["trees"]==p["annual_bindings"][str(period["year"])]["trees"]
        loaded=lgb.Booster(model_file=model["model_path"])
        assert loaded.num_trees()==model["trees"] and model["reload_equal"]
    metrics=pd.read_csv(experiment.OUT/"account_period_metrics.csv")
    for mode in ("fresh","history"):
        for offset in range(5):
            result=experiment.cached("account",f"{mode}_QREG_{offset}")
            directory=Path(result["directory"])
            core.verify_account(directory)
            daily=pd.read_parquet(directory/"daily.parquet")
            holding=pd.read_parquet(directory/"holdings.parquet")
            v=holding.groupby("datetime").market_value.sum().reindex(daily.index,fill_value=0)
            np.testing.assert_allclose(v+daily.cash,daily.account,atol=1e-4,rtol=1e-10)
            wealth=np.r_[1.,(1+daily.net_return).cumprod().to_numpy()]
            mdd=-(wealth/np.maximum.accumulate(wealth)-1).min()
            row=metrics.loc[metrics["mode"].eq(mode)&metrics.case.eq("QREG")&metrics.offset.eq(offset)&metrics.period_kind.eq("all")].iloc[0]
            assert abs(row.max_drawdown-mdd)<1e-12
            dates=experiment.parent.calendar()
            start=experiment.START if mode=="fresh" else experiment.HISTORY_START
            dates=dates[(dates>=start)&(dates<=experiment.END)]
            assert daily.index.equals(dates)
            orders=pd.read_parquet(directory/"orders.parquet")
            assert orders.loc[orders.notional.gt(0)].datetime.isin(dates[offset::5]).all()
    finance=experiment.read(experiment.OUT/"financial_coverage.json")
    assert finance==experiment.financial_audit()
    core.verify_protected(core.Store().protocol,full_snapshot=True)
    experiment.save(experiment.OUT/"final_verification.json",dict(status="passed",verified=time.time(),new_fits=20,
        new_accounts=10,all_quarter_boundaries_valid=True,model_reload=True,accounting_and_drawdowns=True,
        exact_parent_execution_adapter=True,financial_coverage_matches_snapshot=True,
        parent_and_futu_unchanged=True,implementation_sha256=experiment.implementation()))
    print("FINAL VERIFICATION PASSED: 20 quarterly fits, ten accounts, financial coverage and protected sources")


if __name__=="__main__": main()
