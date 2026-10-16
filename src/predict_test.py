"""Generate test predictions for the registered ablation models."""

import argparse
import lightgbm as lgb
import pandas as pd

from src.config import PROJECT_ROOT, load_research_config
from src.factor_library import (
    ENHANCED_FEATURE_NAMES,
    LABEL_NAME,
    PRICE_VOLUME_FEATURE_NAMES,
    RANK_LABEL_NAME,
)
from src.prepare_dataset import split_dataset
from src.train_lightgbm import build_prediction_frame


MODEL_SPECS = {
    "lightgbm_baseline": {
        "dataset_filename": "multifactor_dataset.parquet",
        "feature_names": PRICE_VOLUME_FEATURE_NAMES,
        "retained_columns": [LABEL_NAME],
    },
    "lightgbm_ranked": {
        "dataset_filename": "ranked_multifactor_dataset.parquet",
        "feature_names": PRICE_VOLUME_FEATURE_NAMES,
        "retained_columns": [LABEL_NAME, RANK_LABEL_NAME],
    },
    "lightgbm_enhanced": {
        "dataset_filename": "enhanced_multifactor_dataset.parquet",
        "feature_names": ENHANCED_FEATURE_NAMES,
        "retained_columns": [LABEL_NAME],
    },
}


def generate_test_predictions(model_name: str) -> None:
    spec = MODEL_SPECS[model_name]
    feature_names = spec["feature_names"]

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / spec["dataset_filename"]
    )
    model_path = PROJECT_ROOT / "models" / f"{model_name}.txt"
    output_path = (
        PROJECT_ROOT
        / "reports"
        / "model"
        / f"{model_name}_test_predictions.parquet"
    )

    for path in [dataset_path, model_path]:
        if not path.exists():
            raise FileNotFoundError(f"找不到必要文件：{path}")

    config = load_research_config()
    dataset = pd.read_parquet(dataset_path)
    test_data = split_dataset(dataset, config)["test"]

    booster = lgb.Booster(model_file=str(model_path))
    saved_feature_names = booster.feature_name()

    if saved_feature_names != feature_names:
        raise ValueError(
            f"{model_name}的模型特征与当前配置不一致："
            f"model={saved_feature_names}, "
            f"expected={feature_names}"
        )

    predictions = booster.predict(
        test_data[feature_names],
        num_iteration=booster.current_iteration(),
    )

    prediction_frame = build_prediction_frame(
        frame=test_data,
        predictions=predictions,
        retained_columns=spec["retained_columns"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_parquet(output_path)

    dates = prediction_frame.index.get_level_values("datetime")
    print("测试集预测已生成")
    print(f"模型：{model_name}")
    print(f"模型迭代轮数：{booster.current_iteration()}")
    print(f"股票日期样本数：{len(prediction_frame)}")
    print(f"测试信号区间：{dates.min().date()} 至 {dates.max().date()}")
    print(f"保存位置：{output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test predictions for ablation models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    return parser.parse_args()


def main(model_names: list[str]) -> None:
    for model_name in model_names:
        generate_test_predictions(model_name)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(model_names=arguments.models)
