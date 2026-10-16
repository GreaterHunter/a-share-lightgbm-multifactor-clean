"""Independent final checks of S6 period boundaries, inputs, NAV and output hashes."""
from __future__ import annotations

import gc
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from paper_futu import research_s6_2026 as study
from paper_futu import overfit_four_core as core


def main():
    p = study.check_sources()
    completion = study.read(study.OUT / "completion.json")
    manifest = study.read(study.OUT / "report_manifest.json")
    assert completion["status"] == "complete"
    assert study.digest(study.OUT / "report_manifest.json") == completion["manifest_sha256"]
    for name,expected in manifest["artifacts"].items():
        assert study.digest(name) == expected,name
    assert len(manifest["account_artifacts"]) == 55
    for account,artifacts in manifest["account_artifacts"].items():
        for name,expected in artifacts.items():
            assert study.digest(name) == expected,name
        directory = Path(next(iter(artifacts))).parent
        core.verify_account(directory)
        daily = pd.read_parquet(directory / "daily.parquet")
        values = np.r_[1.,(1+daily.net_return).cumprod().to_numpy()]
        mdd = float(-(values/np.maximum.accumulate(values)-1).min())
        mode,case,offset = account.split("_")
        summary = pd.read_csv(study.OUT / "account_period_metrics.csv")
        record = summary.loc[summary["mode"].eq(mode) & summary.case.eq(case) & summary.offset.eq(int(offset))
                             & summary.period_kind.eq("all")].iloc[0]
        assert abs(record.max_drawdown-mdd) < 1e-12
        schedule = study.calendar()
        interval = schedule[(schedule >= (study.START if mode == "fresh" else study.HISTORY_START)) & (schedule <= study.END)]
        assert daily.index.equals(interval)
        orders = pd.read_parquet(directory / "orders.parquet")
        assert orders.loc[orders.notional.gt(0)].datetime.isin(interval[int(offset)::5]).all()
    frame = study.load_frame()
    boundaries = []
    for cutoff in study.fit_cutoffs():
        result = study.cached("fit",cutoff)
        assert result and result["label_end_max"] <= cutoff
        choice = p["annual_bindings"][cutoff[:4]]
        assert result["trees"] == choice["trees"] and result["parameters"] == choice["parameters"]
        mature = core.mature(frame,"2010-01-01",cutoff)
        assert len(mature) == result["train_rows"]
        assert study.frame_fingerprint(mature) == result["training_fingerprint"]
        loaded = lgb.Booster(model_file=result["model_path"])
        assert loaded.num_trees() == choice["trees"]
        assert result["reload_equal"]
        boundaries.append(dict(cutoff=cutoff,train_signal_end=result["train_signal_end"],
            label_end_max=result["label_end_max"],trees=result["trees"],rows=result["train_rows"]))
        del mature,loaded
        gc.collect()
    assert len(boundaries) == 15
    main = study.read(study.OUT / "research_main.json")
    assert main == study.read(study.ROOT / "configs/experiments/s6_head6_40_research_main.json")
    assert main["variant"] == "V0" and main["portfolio"] == study.PORTFOLIO
    assert main["model_sha256"] == study.digest(main["model_path"])
    core.verify_protected(core.Store().protocol,full_snapshot=True)
    study.save(study.OUT / "final_verification.json",dict(status="passed",verified=time.time(),
        account_count=55,new_accounts=50,reused_accounts=5,new_fits=15,boundaries=boundaries,
        exact_training_fingerprints=True,all_account_hashes_and_accounting=True,
        independent_drawdown_includes_initial_nav=True,unchanged_rebalance_schedule=True,
        research_main_config_equal=True,protected_state_unchanged=True,implementation_sha256=study.implementation()))
    print("FINAL VERIFICATION PASSED: 55 accounts, 15 mature-label fits, protected sources unchanged")


if __name__ == "__main__":
    main()
