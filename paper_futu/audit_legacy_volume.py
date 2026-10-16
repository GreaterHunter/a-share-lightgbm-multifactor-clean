"""Read-only audit of the existing 2026-10-12 paper signal's volume units."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from paper_futu.cn_common import ROOT, STATE_DIR, load_deployment
from paper_futu.research_alpha158_morning import paths_for, save_json
from src.build_ranked_dataset import cross_sectional_rank
from src.factor_library import ENHANCED_FEATURE_NAMES, FEATURE_EXPRESSIONS


def main():
    source = STATE_DIR / "2026-10-12"
    report = paths_for(pd.Timestamp("2026-10-13"))["report"]
    report.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(source / "features_raw.parquet")
    original = pd.read_csv(source / "predictions.csv").set_index(["instrument", "datetime"])
    original.index = original.index.set_levels(pd.to_datetime(original.index.levels[1]), level=1)
    model_dir, metadata = load_deployment()
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    if booster.feature_name() != ENHANCED_FEATURE_NAMES:
        raise ValueError("Existing model uses unexpected feature names")
    ranked = cross_sectional_rank(raw, ENHANCED_FEATURE_NAMES)
    old_prediction = pd.Series(booster.predict(ranked, num_threads=4), index=ranked.index)
    stored = original.reindex(old_prediction.index).prediction
    if not np.allclose(old_prediction, stored, rtol=0, atol=1e-9):
        raise ValueError("Cannot reproduce the frozen old signal before volume correction")

    provider = paths_for(pd.Timestamp("2026-10-13"))["data"] / "provider"
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1)
    affected = ["volume_ratio_5_20", "price_volume_corr_20d"]
    corrected = D.features(
        raw.index.get_level_values("instrument").tolist(),
        [FEATURE_EXPRESSIONS[name] for name in affected],
        start_time="2026-10-12", end_time="2026-10-12", freq="day",
    )
    corrected.columns = affected
    if not raw.index.isin(corrected.index).all():
        raise ValueError("Some frozen signal stocks are missing from the verified provider")
    repaired = raw.copy()
    repaired.loc[:, affected] = corrected.reindex(raw.index)[affected]
    new_ranked = cross_sectional_rank(repaired, ENHANCED_FEATURE_NAMES)
    new_prediction = pd.Series(booster.predict(new_ranked, num_threads=4), index=raw.index)
    compared = pd.DataFrame({"original_prediction": old_prediction,
                             "corrected_prediction": new_prediction,
                             "old_volume_ratio": raw[affected[0]],
                             "corrected_volume_ratio": repaired[affected[0]]})
    compared["original_rank"] = compared.original_prediction.rank(ascending=False, method="first")
    compared["corrected_rank"] = compared.corrected_prediction.rank(ascending=False, method="first")
    compared.to_csv(report / "frozen_signal_volume_scale_audit.csv", encoding="utf-8-sig")
    eligible = original.reindex(compared.index).eligible.astype(bool)
    old_top = compared.loc[eligible].nsmallest(10, "original_rank")
    new_top = compared.loc[eligible].nsmallest(10, "corrected_rank")
    candidate = compared.xs("SZ300316", level="instrument")
    result = {"signal_date": "2026-10-12", "frozen_model_sha256": metadata["model_sha256"],
              "observations": len(compared), "old_volume_ratio_median": float(raw[affected[0]].median()),
              "corrected_volume_ratio_median": float(repaired[affected[0]].median()),
              "old_top10": old_top.index.get_level_values("instrument").tolist(),
              "corrected_top10": new_top.index.get_level_values("instrument").tolist(),
              "top10_overlap": len(set(old_top.index.get_level_values("instrument"))
                                   & set(new_top.index.get_level_values("instrument"))),
              "SZ300316_original_rank": float(candidate.original_rank.iloc[0]),
              "SZ300316_corrected_rank": float(candidate.corrected_rank.iloc[0]),
              "no_orders_or_existing_plans_modified": True}
    save_json(report / "frozen_signal_volume_scale_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
