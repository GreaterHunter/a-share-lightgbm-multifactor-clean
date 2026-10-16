"""Refit the frozen annual 2026 CSI300 model for paper trading."""

from __future__ import annotations

from copy import deepcopy
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

from src.config import PROJECT_ROOT, load_main_strategy_config, load_model_config, load_research_config
from src.factor_library import ENHANCED_FEATURE_NAMES
from src.forward_labels import label_contract, select_mature_period
from src.train_walk_forward_cv import build_validation_folds, fit_selection_fold, summarize_iteration_candidates

OUTPUT = PROJECT_ROOT / "models/paper_futu/cn_2026_open_to_open"
SOURCE = PROJECT_ROOT / "data/processed/cn_latest_mature_open10_20261012/supervised_dataset.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--portfolio-config", type=Path)
    parser.add_argument("--dataset", type=Path, default=SOURCE, help="Open-to-open dataset with label_end_date")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if not output.resolve().is_relative_to((PROJECT_ROOT / "models").resolve()):
        raise ValueError("Model output must stay inside the project's models directory")
    output.mkdir(parents=True, exist_ok=True)
    strategy = load_main_strategy_config()
    portfolio = {name: strategy[name] for name in ["top_k", "n_drop", "top_tier_size", "top_tier_weight", "risk_degree", "rebalance_every", "buy_cost", "sell_cost", "minimum_cost", "initial_cash"]}
    if args.portfolio_config:
        path = args.portfolio_config if args.portfolio_config.is_absolute() else PROJECT_ROOT / args.portfolio_config
        overrides = yaml.safe_load(path.read_text(encoding="utf-8"))
        if set(overrides).difference(portfolio):
            raise ValueError("Portfolio configuration contains unsupported fields")
        portfolio.update(overrides)
    if not 0 < portfolio["n_drop"] <= portfolio["top_k"] or not 0 < portfolio["top_tier_size"] <= portfolio["top_k"]:
        raise ValueError("Invalid TopK portfolio configuration")
    research = load_research_config()
    parameters = deepcopy(load_model_config()["parameters"])
    parameters["n_jobs"] = 4
    source = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    contract = label_contract(10)
    dataset = pd.read_parquet(source)
    if not {contract["raw_label"], contract["target"], "label_end_date"}.issubset(dataset.columns):
        raise ValueError("拒绝旧收盘标签的数据集；请先构建开盘→开盘标签")
    dates = dataset.index.get_level_values("datetime")
    # Keep the project's annual refit boundary: no 2026 outcomes enter fitting.
    dataset = dataset.loc[dates < pd.Timestamp("2026-01-01")].copy()
    if dataset.index.duplicated().any():
        raise ValueError("训练数据股票日期重复")
    label, target = contract["raw_label"], contract["target"]
    records = []
    for fold in build_validation_folds(2026, int(strategy["validation_folds"]), 2010):
        print(f"Selection fold {fold['validation_year']}: {fold['train_start']} to {fold['train_end']}", flush=True)
        records.extend(fit_selection_fold(
            dataset, fold, ENHANCED_FEATURE_NAMES, target, label, parameters,
            strategy["iteration_candidates"], 10, int(research["min_cross_section_size"]),
        ))
        pd.DataFrame(records).to_csv(output / "validation_folds.csv", index=False)
    candidates = summarize_iteration_candidates(records)
    candidates.to_csv(output / "iteration_candidates.csv", index=False)
    chosen = int(candidates.iloc[0]["candidate_iteration"])
    train = select_mature_period(dataset, "2010-01-01", "2025-12-31")
    parameters["n_estimators"] = chosen
    print(f"Refitting fresh model on {len(train):,} rows with {chosen} trees", flush=True)
    model = lgb.LGBMRegressor(**parameters)
    model.fit(train[ENHANCED_FEATURE_NAMES], train[target])
    path = output / "model.txt"
    model.booster_.save_model(str(path))
    metadata = {
        **contract,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_year": 2026, "training_mode": strategy["training_mode"],
        "train_start": "2010-01-01", "train_end": "2025-12-31",
        "last_retained_signal_date": str(train.index.get_level_values("datetime").max().date()),
        "last_label_end_date": str(train.label_end_date.max().date()),
        "validation_years": [2023, 2024, 2025], "train_rows": len(train),
        "label_horizon": 10, "rebalance_every": 10, "chosen_iteration": chosen,
        "selection_mean_rank_ic": float(candidates.iloc[0]["mean_validation_rank_ic"]),
        "features": ENHANCED_FEATURE_NAMES, "parameters": parameters,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "no_2026_training_outcomes": True,
        "portfolio": portfolio,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
