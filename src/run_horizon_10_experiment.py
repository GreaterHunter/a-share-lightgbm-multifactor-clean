"""Build and train a ranked-enhanced annual rolling horizon experiment."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from src.build_ranked_dataset import cross_sectional_rank
from src.config import PROJECT_ROOT, load_model_config, load_research_config
from src.factor_library import ENHANCED_FEATURE_NAMES
from src.forward_labels import OPEN_TO_OPEN, LEGACY_OPEN_TO_CLOSE, label_contract, label_end_dates
from src.run_temporal_ablation import (
    ROLLING_FOLDS,
    fit_one_fold,
    save_prediction_frame,
)


DEFAULT_HORIZON = 10


def build_label_expression(horizon: int, price_basis: str = OPEN_TO_OPEN) -> str:
    """Active default matches next-open trading; legacy experiments opt in."""
    return label_contract(horizon, price_basis)["label_expression"]


def build_horizon_dataset(
    config: dict,
    horizon: int,
    price_basis: str = OPEN_TO_OPEN,
) -> tuple[pd.DataFrame, str, str]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    contract = label_contract(horizon, price_basis)
    raw_label_name, rank_label_name = contract["raw_label"], contract["target"]
    label_expression = contract["label_expression"]
    source_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ranked_enhanced_multifactor_dataset.parquet"
    )
    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / (f"ranked_enhanced_open_{horizon}d_dataset.parquet" if price_basis == OPEN_TO_OPEN
           else f"ranked_enhanced_{horizon}d_dataset.parquet")
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Ranked enhanced dataset not found: {source_path}")

    ranked_features = pd.read_parquet(
        source_path,
        columns=ENHANCED_FEATURE_NAMES,
    )
    instruments = D.instruments(config["universe"])
    raw_label_panel = D.features(
        instruments=instruments,
        fields=[label_expression],
        start_time=config["train_start"],
        end_time=config["test_end"],
        freq="day",
    )
    raw_label_panel.columns = [raw_label_name]

    dataset = ranked_features.join(raw_label_panel, how="inner")
    if price_basis == OPEN_TO_OPEN:
        dataset["label_end_date"] = label_end_dates(dataset.index, D.calendar(freq="day"), horizon)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=[raw_label_name])
    all_features_missing = dataset[ENHANCED_FEATURE_NAMES].isna().all(axis=1)
    dataset = dataset.loc[~all_features_missing].copy()

    dataset[rank_label_name] = cross_sectional_rank(
        frame=dataset,
        columns=[raw_label_name],
    )[raw_label_name].astype("float32")
    dataset = dataset.dropna(subset=[rank_label_name]).sort_index()

    if dataset.index.duplicated().any():
        raise ValueError(f"{horizon}-day dataset contains duplicate stock-date rows")
    if not dataset[rank_label_name].between(-0.5, 0.5).all():
        raise ValueError(f"{horizon}-day ranked target is outside [-0.5, 0.5]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path)
    print(f"{horizon}-day dataset shape: {dataset.shape}")
    print(f"{horizon}-day dataset saved to: {output_path}")
    return dataset, raw_label_name, rank_label_name


def train_rolling_models(
    dataset: pd.DataFrame,
    config: dict,
    horizon: int,
    raw_label: str,
    rank_label: str,
) -> None:
    model_name = f"ranked_enhanced_{horizon}d"
    experiment_name = f"horizon_{horizon}d_rolling"
    model_config = load_model_config()
    spec = {
        "features": ENHANCED_FEATURE_NAMES,
        "target": rank_label,
        "evaluation_label": raw_label,
        "retained": [raw_label, rank_label],
    }
    predictions = []
    records = []

    for fold in ROLLING_FOLDS:
        print(f"Training {horizon}-day model: {fold['fold']}")
        _, fold_test, metrics = fit_one_fold(
            dataset=dataset,
            model_name=model_name,
            spec=spec,
            fold=fold,
            experiment_name=experiment_name,
            model_parameters=model_config["parameters"],
            training_parameters=model_config["training"],
            label_horizon=horizon,
            min_cross_section_size=int(config["min_cross_section_size"]),
        )
        predictions.append(fold_test)
        records.append(metrics)

    combined = pd.concat(predictions, axis=0).sort_index()
    if combined.index.duplicated().any():
        raise ValueError(f"{horizon}-day rolling predictions contain duplicate rows")
    prediction_path = save_prediction_frame(
        combined,
        experiment_name,
        model_name,
        "test",
    )

    summary = pd.DataFrame(records)
    summary_path = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / experiment_name
        / "training_summary.csv"
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n{horizon}-day rolling training summary:")
    print(summary.to_string(index=False))
    print(prediction_path)
    print(summary_path)


def run_horizon_experiment(horizon: int) -> None:
    config = load_research_config()
    qlib.init(
        provider_uri=str(PROJECT_ROOT / config["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    dataset, raw_label, rank_label = build_horizon_dataset(
        config,
        horizon,
        price_basis=LEGACY_OPEN_TO_CLOSE,  # Archived single-fold experiment, not deployable.
    )
    train_rolling_models(
        dataset,
        config,
        horizon,
        raw_label,
        rank_label,
    )
    del dataset
    gc.collect()


def main() -> None:
    run_horizon_experiment(DEFAULT_HORIZON)


if __name__ == "__main__":
    main()
