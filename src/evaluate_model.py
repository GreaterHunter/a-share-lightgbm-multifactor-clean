import argparse

import numpy as np
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    load_model_config,
    load_research_config,
)
from src.evaluate_factors import (
    calculate_information_ratio,
    calculate_one_day_factor_ic,
)
from src.factor_library import LABEL_NAME, RANK_LABEL_NAME


MODEL_TARGET_NAMES = {
    "lightgbm_ranked": RANK_LABEL_NAME,
}


def validate_prediction_frame(
    prediction_frame: pd.DataFrame,
    target_name: str = LABEL_NAME,
) -> None:
    required_columns = {
        LABEL_NAME,
        target_name,
        "prediction",
    }

    missing_columns = (
        required_columns
        - set(prediction_frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"预测文件缺少列："
            f"{sorted(missing_columns)}"
        )

    required_index_names = {
        "instrument",
        "datetime",
    }

    missing_index_names = (
        required_index_names
        - set(prediction_frame.index.names)
    )

    if missing_index_names:
        raise ValueError(
            f"预测文件索引缺少："
            f"{sorted(missing_index_names)}"
        )

    duplicate_count = (
        prediction_frame.index
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"预测文件存在"
            f"{duplicate_count}条重复索引"
        )

    numeric_values = prediction_frame[
        [
            target_name,
            "prediction",
        ]
    ].to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "标签或模型预测中存在"
            "缺失值或无穷值"
        )


def calculate_daily_model_ic(
    prediction_frame: pd.DataFrame,
    min_cross_section_size: int,
    label_name: str = LABEL_NAME,
) -> pd.DataFrame:
    records = []

    daily_groups = prediction_frame.groupby(
        level="datetime",
        sort=True,
    )

    for signal_date, day_panel in daily_groups:
        result = calculate_one_day_factor_ic(
            day_panel=day_panel,
            factor_name="prediction",
            min_cross_section_size=(
                min_cross_section_size
            ),
            label_name=label_name,
        )

        records.append(
            {
                "datetime": signal_date,
                "observations": result[
                    "observations"
                ],
                "ic": result["ic"],
                "rank_ic": result["rank_ic"],
            }
        )

    daily_ic = pd.DataFrame(records)

    daily_ic = daily_ic.dropna(
        subset=[
            "ic",
            "rank_ic",
        ],
        how="all",
    )

    return daily_ic.reset_index(
        drop=True
    )


def summarize_model_performance(
    prediction_frame: pd.DataFrame,
    daily_ic: pd.DataFrame,
    model_name: str,
    target_name: str,
) -> pd.DataFrame:
    if daily_ic.empty:
        raise ValueError(
            "没有可用于模型评价的有效交易日"
        )

    labels = prediction_frame[target_name]

    predictions = prediction_frame[
        "prediction"
    ]

    model_mse = (
        (labels - predictions) ** 2
    ).mean()

    mean_ic = daily_ic["ic"].mean()
    std_ic = daily_ic["ic"].std()

    mean_rank_ic = daily_ic[
        "rank_ic"
    ].mean()

    std_rank_ic = daily_ic[
        "rank_ic"
    ].std()

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "target_name": target_name,
                "observations": len(
                    prediction_frame
                ),
                "valid_days": len(daily_ic),
                "model_mse": model_mse,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir_non_annualized": (
                    calculate_information_ratio(
                        mean_value=mean_ic,
                        std_value=std_ic,
                    )
                ),
                "positive_ic_ratio": (
                    (daily_ic["ic"] > 0)
                    .mean()
                ),
                "mean_rank_ic": mean_rank_ic,
                "std_rank_ic": std_rank_ic,
                "rank_icir_non_annualized": (
                    calculate_information_ratio(
                        mean_value=mean_rank_ic,
                        std_value=std_rank_ic,
                    )
                ),
                "positive_rank_ic_ratio": (
                    (daily_ic["rank_ic"] > 0)
                    .mean()
                ),
            }
        ]
    )

    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved validation prediction file.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model artifact prefix; defaults to configs/model.yaml.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
    )
    return parser.parse_args()


def main(
    model_name: str | None = None,
    split_name: str = "valid",
) -> None:
    research_config = load_research_config()
    if model_name is None:
        model_config = load_model_config()
        model_name = model_config["model_name"]

    target_name = MODEL_TARGET_NAMES.get(
        model_name,
        LABEL_NAME,
    )

    prediction_path = (
        PROJECT_ROOT
        / "reports"
        / "model"
        / f"{model_name}_{split_name}_predictions.parquet"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"找不到验证集预测文件："
            f"{prediction_path}"
        )

    prediction_frame = pd.read_parquet(
        prediction_path
    )

    validate_prediction_frame(
        prediction_frame,
        target_name=target_name,
    )

    min_cross_section_size = int(
        research_config[
            "min_cross_section_size"
        ]
    )

    daily_ic = calculate_daily_model_ic(
        prediction_frame=prediction_frame,
        min_cross_section_size=(
            min_cross_section_size
        ),
    )

    summary = summarize_model_performance(
        prediction_frame=prediction_frame,
        daily_ic=daily_ic,
        model_name=model_name,
        target_name=target_name,
    )

    reports_dir = (
        PROJECT_ROOT
        / "reports"
        / "model"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_ic_path = (
        reports_dir
        / f"{model_name}_{split_name}_daily_ic.csv"
    )

    summary_path = (
        reports_dir
        / f"{model_name}_{split_name}_metrics.csv"
    )

    daily_ic.to_csv(
        daily_ic_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("模型验证数据检查通过")
    print(f"模型名称：{model_name}")
    print(
        f"股票日期样本数："
        f"{len(prediction_frame)}"
    )
    print(
        f"有效IC交易日数："
        f"{len(daily_ic)}"
    )

    print("\n每日模型IC前10行：")
    print(
        daily_ic.head(10).to_string(
            index=False
        )
    )

    print("\n验证集模型汇总：")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\n报告保存位置：")
    print(daily_ic_path)
    print(summary_path)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(
        model_name=arguments.model_name,
        split_name=arguments.split,
    )
