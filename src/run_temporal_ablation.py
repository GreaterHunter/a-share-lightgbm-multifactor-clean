"""Train four factor variants under static and annual rolling windows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from src.config import PROJECT_ROOT, load_model_config, load_research_config
from src.evaluate_model import calculate_daily_model_ic
from src.factor_library import (
    ENHANCED_FEATURE_NAMES,
    LABEL_NAME,
    PRICE_VOLUME_FEATURE_NAMES,
    RANK_LABEL_NAME,
)
from src.prepare_dataset import purge_last_signal_dates
from src.train_lightgbm import build_prediction_frame


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "baseline": {
        "dataset": "multifactor_dataset.parquet",
        "features": PRICE_VOLUME_FEATURE_NAMES,
        "target": LABEL_NAME,
        "retained": [LABEL_NAME],
    },
    "ranked": {
        "dataset": "ranked_multifactor_dataset.parquet",
        "features": PRICE_VOLUME_FEATURE_NAMES,
        "target": RANK_LABEL_NAME,
        "retained": [LABEL_NAME, RANK_LABEL_NAME],
    },
    "enhanced": {
        "dataset": "enhanced_multifactor_dataset.parquet",
        "features": ENHANCED_FEATURE_NAMES,
        "target": LABEL_NAME,
        "retained": [LABEL_NAME],
    },
    "ranked_enhanced": {
        "dataset": "ranked_enhanced_multifactor_dataset.parquet",
        "features": ENHANCED_FEATURE_NAMES,
        "target": RANK_LABEL_NAME,
        "retained": [LABEL_NAME, RANK_LABEL_NAME],
    },
}


STATIC_FOLD = {
    "fold": "static_2023_2024",
    "train_start": "2010-01-01",
    "train_end": "2020-12-31",
    "valid_start": "2021-01-01",
    "valid_end": "2022-12-31",
    "test_start": "2023-01-01",
    "test_end": "2024-12-31",
}


ROLLING_FOLDS = [
    {
        "fold": "test_2022",
        "train_start": "2010-01-01",
        "train_end": "2020-12-31",
        "valid_start": "2021-01-01",
        "valid_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
    },
    {
        "fold": "test_2023",
        "train_start": "2011-01-01",
        "train_end": "2021-12-31",
        "valid_start": "2022-01-01",
        "valid_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
    },
    {
        "fold": "test_2024",
        "train_start": "2012-01-01",
        "train_end": "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
]


def select_period(
    dataset: pd.DataFrame,
    start: str,
    end: str,
    label_horizon: int,
    purge_end: bool,
) -> pd.DataFrame:
    dates = dataset.index.get_level_values("datetime")
    mask = (
        (dates >= pd.Timestamp(start))
        & (dates <= pd.Timestamp(end))
    )
    frame = dataset.loc[mask].copy()

    if frame.empty:
        raise ValueError(f"时间区间没有样本：{start} 至 {end}")

    if purge_end:
        frame, _ = purge_last_signal_dates(
            frame=frame,
            label_horizon=label_horizon,
        )

    return frame


def fit_one_fold(
    dataset: pd.DataFrame,
    model_name: str,
    spec: dict[str, Any],
    fold: dict[str, str],
    experiment_name: str,
    model_parameters: dict[str, Any],
    training_parameters: dict[str, Any],
    label_horizon: int,
    min_cross_section_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_names = spec["features"]
    target_name = spec["target"]
    evaluation_label_name = spec.get(
        "evaluation_label",
        LABEL_NAME,
    )

    train_data = select_period(
        dataset,
        fold["train_start"],
        fold["train_end"],
        label_horizon,
        purge_end=True,
    )
    valid_data = select_period(
        dataset,
        fold["valid_start"],
        fold["valid_end"],
        label_horizon,
        purge_end=True,
    )
    test_data = select_period(
        dataset,
        fold["test_start"],
        fold["test_end"],
        label_horizon,
        purge_end=False,
    )

    selection_model = lgb.LGBMRegressor(
        **deepcopy(model_parameters)
    )
    selection_model.fit(
        train_data[feature_names],
        train_data[target_name],
        eval_set=[
            (valid_data[feature_names], valid_data[target_name]),
        ],
        eval_names=["valid"],
        eval_metric=model_parameters["metric"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=training_parameters[
                    "early_stopping_rounds"
                ],
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iteration = int(selection_model.best_iteration_)
    valid_predictions = selection_model.predict(
        valid_data[feature_names],
        num_iteration=best_iteration,
    )
    valid_prediction_frame = build_prediction_frame(
        frame=valid_data,
        predictions=valid_predictions,
        retained_columns=spec["retained"],
    )

    refit_data = select_period(
        dataset,
        fold["train_start"],
        fold["valid_end"],
        label_horizon,
        purge_end=True,
    )
    refit_parameters = deepcopy(model_parameters)
    refit_parameters["n_estimators"] = best_iteration
    final_model = lgb.LGBMRegressor(**refit_parameters)
    final_model.fit(
        refit_data[feature_names],
        refit_data[target_name],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    test_predictions = final_model.predict(test_data[feature_names])
    test_prediction_frame = build_prediction_frame(
        frame=test_data,
        predictions=test_predictions,
        retained_columns=spec["retained"],
    )

    valid_daily_ic = calculate_daily_model_ic(
        prediction_frame=valid_prediction_frame,
        min_cross_section_size=min_cross_section_size,
        label_name=evaluation_label_name,
    )
    valid_mse = float(
        (
            valid_prediction_frame["prediction"]
            - valid_prediction_frame[target_name]
        ).pow(2).mean()
    )

    models_dir = (
        PROJECT_ROOT
        / "models"
        / "experiments"
        / experiment_name
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(
        str(models_dir / f"{model_name}_{fold['fold']}.txt")
    )

    metrics = {
        "experiment": experiment_name,
        "fold": fold["fold"],
        "model": model_name,
        "target": target_name,
        "best_iteration": best_iteration,
        "train_start": fold["train_start"],
        "train_end": fold["train_end"],
        "valid_start": fold["valid_start"],
        "valid_end": fold["valid_end"],
        "test_start": fold["test_start"],
        "test_end": fold["test_end"],
        "train_rows": len(train_data),
        "valid_rows": len(valid_data),
        "refit_rows": len(refit_data),
        "test_rows": len(test_data),
        "valid_mse": valid_mse,
        "valid_mean_ic": float(valid_daily_ic["ic"].mean()),
        "valid_mean_rank_ic": float(valid_daily_ic["rank_ic"].mean()),
        "valid_positive_rank_ic_ratio": float(
            (valid_daily_ic["rank_ic"] > 0).mean()
        ),
    }

    return valid_prediction_frame, test_prediction_frame, metrics


def save_prediction_frame(
    frame: pd.DataFrame,
    experiment_name: str,
    model_name: str,
    split_name: str,
) -> Path:
    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / experiment_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_{split_name}_predictions.parquet"
    frame.sort_index().to_parquet(output_path)
    return output_path


def main() -> None:
    research_config = load_research_config()
    model_config = load_model_config()
    model_parameters = model_config["parameters"]
    training_parameters = model_config["training"]
    label_horizon = int(research_config["label_horizon"])
    min_cross_section_size = int(
        research_config["min_cross_section_size"]
    )

    training_records = []

    for model_name, spec in MODEL_SPECS.items():
        dataset_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / spec["dataset"]
        )
        if not dataset_path.exists():
            raise FileNotFoundError(f"找不到实验数据集：{dataset_path}")

        print(f"\n读取模型数据：{model_name}")
        dataset = pd.read_parquet(dataset_path)

        print(f"训练静态两年验证实验：{model_name}")
        static_valid, static_test, static_metrics = fit_one_fold(
            dataset=dataset,
            model_name=model_name,
            spec=spec,
            fold=STATIC_FOLD,
            experiment_name="static_2y",
            model_parameters=model_parameters,
            training_parameters=training_parameters,
            label_horizon=label_horizon,
            min_cross_section_size=min_cross_section_size,
        )
        save_prediction_frame(
            static_valid, "static_2y", model_name, "valid"
        )
        save_prediction_frame(
            static_test, "static_2y", model_name, "test"
        )
        training_records.append(static_metrics)

        rolling_predictions = []
        for fold in ROLLING_FOLDS:
            print(f"训练年度滚动：{model_name} / {fold['fold']}")
            _, fold_test, fold_metrics = fit_one_fold(
                dataset=dataset,
                model_name=model_name,
                spec=spec,
                fold=fold,
                experiment_name="rolling_annual",
                model_parameters=model_parameters,
                training_parameters=training_parameters,
                label_horizon=label_horizon,
                min_cross_section_size=min_cross_section_size,
            )
            rolling_predictions.append(fold_test)
            training_records.append(fold_metrics)

        combined_rolling = pd.concat(
            rolling_predictions,
            axis=0,
        ).sort_index()
        if combined_rolling.index.duplicated().any():
            raise ValueError(f"{model_name}滚动预测存在重复索引")
        save_prediction_frame(
            combined_rolling,
            "rolling_annual",
            model_name,
            "test",
        )

        del dataset

    summary = pd.DataFrame(training_records)
    summary_path = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / "temporal_ablation_training_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n全部模型训练完成")
    print(summary.to_string(index=False))
    print(summary_path)


if __name__ == "__main__":
    main()
