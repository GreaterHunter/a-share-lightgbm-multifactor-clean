from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    load_model_config,
    load_research_config,
)
from src.factor_library import (
    LABEL_NAME,
    PRICE_VOLUME_FEATURE_NAMES,
)
from src.prepare_dataset import split_dataset


def build_feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    booster = model.booster_

    gain_importance = booster.feature_importance(
        importance_type="gain",
    )

    split_importance = booster.feature_importance(
        importance_type="split",
    )

    importance = pd.DataFrame(
        {
            "factor": feature_names,
            "gain": gain_importance,
            "split": split_importance,
        }
    )

    total_gain = importance["gain"].sum()

    if total_gain > 0:
        importance["gain_ratio"] = (
            importance["gain"]
            / total_gain
        )
    else:
        importance["gain_ratio"] = 0.0

    return (
        importance.sort_values(
            by="gain",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_prediction_frame(
    frame: pd.DataFrame,
    predictions,
    retained_columns: list[str],
) -> pd.DataFrame:
    prediction_frame = frame[
        retained_columns
    ].copy()

    prediction_frame["prediction"] = (
        predictions
    )

    return prediction_frame


def validate_training_input(
    dataset: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    retained_columns: list[str],
) -> None:
    required_columns = (
        feature_names
        + [target_name]
        + retained_columns
    )

    missing_columns = (
        set(required_columns)
        - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "训练数据缺少必要列："
            f"{sorted(missing_columns)}"
        )

    if dataset[target_name].isna().any():
        raise ValueError(
            f"训练目标存在缺失值："
            f"{target_name}"
        )


def train_lightgbm_experiment(
    dataset_path: Path,
    feature_names: list[str],
    target_name: str,
    model_name: str,
    retained_columns: list[str],
) -> dict[str, Any]:
    research_config = load_research_config()
    model_config = load_model_config()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"找不到训练数据集："
            f"{dataset_path}"
        )

    dataset = pd.read_parquet(
        dataset_path
    )

    validate_training_input(
        dataset=dataset,
        feature_names=feature_names,
        target_name=target_name,
        retained_columns=retained_columns,
    )

    splits = split_dataset(
        dataset=dataset,
        config=research_config,
    )

    train_data = splits["train"]
    valid_data = splits["valid"]

    x_train = train_data[
        feature_names
    ]

    y_train = train_data[
        target_name
    ]

    x_valid = valid_data[
        feature_names
    ]

    y_valid = valid_data[
        target_name
    ]

    model_parameters = model_config[
        "parameters"
    ]

    train_config = model_config[
        "training"
    ]

    model = lgb.LGBMRegressor(
        **model_parameters
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[
            (x_valid, y_valid),
        ],
        eval_names=[
            "valid",
        ],
        eval_metric=model_parameters[
            "metric"
        ],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=train_config[
                    "early_stopping_rounds"
                ]
            ),
            lgb.log_evaluation(
                period=train_config[
                    "log_evaluation_period"
                ]
            ),
        ],
    )

    valid_predictions = model.predict(
        x_valid,
        num_iteration=model.best_iteration_,
    )

    valid_prediction_frame = (
        build_prediction_frame(
            frame=valid_data,
            predictions=valid_predictions,
            retained_columns=retained_columns,
        )
    )

    models_dir = (
        PROJECT_ROOT
        / "models"
    )

    reports_dir = (
        PROJECT_ROOT
        / "reports"
        / "model"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_dir
        / f"{model_name}.txt"
    )

    importance_path = (
        reports_dir
        / f"{model_name}_feature_importance.csv"
    )

    valid_prediction_path = (
        reports_dir
        / f"{model_name}_valid_predictions.parquet"
    )

    model.booster_.save_model(
        str(model_path),
        num_iteration=model.best_iteration_,
    )

    feature_importance = (
        build_feature_importance(
            model=model,
            feature_names=feature_names,
        )
    )

    feature_importance.to_csv(
        importance_path,
        index=False,
        encoding="utf-8-sig",
    )

    valid_prediction_frame.to_parquet(
        valid_prediction_path,
    )

    print("LightGBM训练完成")
    print(f"模型名称：{model_name}")
    print(f"预测目标：{target_name}")
    print(
        "最佳迭代轮数："
        f"{model.best_iteration_}"
    )

    print("\n验证集最佳评估结果：")
    print(model.best_score_)

    print("\n特征重要性：")
    print(
        feature_importance.to_string(
            index=False,
        )
    )

    print("\n保存位置：")
    print(model_path)
    print(importance_path)
    print(valid_prediction_path)

    return {
        "model": model,
        "feature_importance": (
            feature_importance
        ),
        "valid_predictions": (
            valid_prediction_frame
        ),
        "model_path": model_path,
        "importance_path": importance_path,
        "valid_prediction_path": (
            valid_prediction_path
        ),
    }


def main() -> None:
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "multifactor_dataset.parquet"
    )

    model_config = load_model_config()

    train_lightgbm_experiment(
        dataset_path=dataset_path,
        feature_names=PRICE_VOLUME_FEATURE_NAMES,
        target_name=LABEL_NAME,
        model_name=model_config[
            "model_name"
        ],
        retained_columns=[
            LABEL_NAME,
        ],
    )


if __name__ == "__main__":
    main()
