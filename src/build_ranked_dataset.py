import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.factor_library import (
    LABEL_NAME,
    PRICE_VOLUME_FEATURE_NAMES,
    RANK_LABEL_NAME,
)
from src.prepare_dataset import validate_dataset


def cross_sectional_rank(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    ranked = (
        frame[
            columns
        ]
        .groupby(
            level="datetime",
            sort=False,
        )
        .rank(
            method="average",
            pct=True,
        )
    )

    return ranked - 0.5


def build_ranked_dataset(
    dataset: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    validate_dataset(
        dataset=dataset,
        feature_names=feature_names,
    )

    ranked_features = cross_sectional_rank(
        frame=dataset,
        columns=feature_names,
    )

    ranked_target = cross_sectional_rank(
        frame=dataset,
        columns=[LABEL_NAME],
    )[LABEL_NAME]

    ranked_dataset = ranked_features.copy()

    ranked_dataset[LABEL_NAME] = dataset[
        LABEL_NAME
    ]

    ranked_dataset[RANK_LABEL_NAME] = (
        ranked_target
    )

    rank_columns = (
        feature_names
        + [RANK_LABEL_NAME]
    )

    ranked_dataset[rank_columns] = (
        ranked_dataset[rank_columns]
        .astype("float32")
    )

    output_columns = (
        feature_names
        + [
            LABEL_NAME,
            RANK_LABEL_NAME,
        ]
    )

    return ranked_dataset[
        output_columns
    ].sort_index()


def validate_ranked_dataset(
    ranked_dataset: pd.DataFrame,
    feature_names: list[str],
) -> None:
    expected_columns = (
        feature_names
        + [
            LABEL_NAME,
            RANK_LABEL_NAME,
        ]
    )

    missing_columns = (
        set(expected_columns)
        - set(ranked_dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            f"排名数据集缺少列："
            f"{sorted(missing_columns)}"
        )

    if ranked_dataset[
        RANK_LABEL_NAME
    ].isna().any():
        raise ValueError(
            "排名标签存在缺失值"
        )

    rank_columns = (
        feature_names
        + [RANK_LABEL_NAME]
    )

    rank_values = ranked_dataset[
        rank_columns
    ].to_numpy()

    infinite_count = int(
        np.isinf(rank_values).sum()
    )

    if infinite_count > 0:
        raise ValueError(
            f"排名数据存在无穷值："
            f"{infinite_count}"
        )

    finite_values = rank_values[
        np.isfinite(rank_values)
    ]

    if finite_values.size == 0:
        raise ValueError(
            "排名数据不存在有效数值"
        )

    minimum_rank = float(
        finite_values.min()
    )

    maximum_rank = float(
        finite_values.max()
    )

    tolerance = 1e-12

    if (
        minimum_rank < -0.5 - tolerance
        or maximum_rank > 0.5 + tolerance
    ):
        raise ValueError(
            "排名数值超出预期范围："
            f"{minimum_rank} 至 {maximum_rank}"
        )


def main() -> None:
    input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "multifactor_dataset.parquet"
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ranked_multifactor_dataset.parquet"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到原始多因子数据集："
            f"{input_path}"
        )

    dataset = pd.read_parquet(
        input_path
    )

    feature_names = PRICE_VOLUME_FEATURE_NAMES

    ranked_dataset = build_ranked_dataset(
        dataset=dataset,
        feature_names=feature_names,
    )

    validate_ranked_dataset(
        ranked_dataset=ranked_dataset,
        feature_names=feature_names,
    )

    original_label = dataset[
        LABEL_NAME
    ]

    preserved_label = ranked_dataset[
        LABEL_NAME
    ]

    if not preserved_label.equals(
        original_label
    ):
        raise ValueError(
            "排名处理意外修改了原始收益标签"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranked_dataset.to_parquet(
        output_path,
    )

    rank_columns = (
        feature_names
        + [RANK_LABEL_NAME]
    )

    rank_values = ranked_dataset[
        rank_columns
    ].to_numpy()

    finite_rank_values = rank_values[
        np.isfinite(rank_values)
    ]

    print("排名数据集构建完成")

    print("\n数据前10行：")
    print(
        ranked_dataset.head(10)
    )

    print("\n数据形状（行数, 列数）：")
    print(ranked_dataset.shape)

    print("\n排名数值范围：")
    print(
        f"{finite_rank_values.min():.6f}"
        f" 至 "
        f"{finite_rank_values.max():.6f}"
    )

    print("\n原始收益标签描述统计：")
    print(
        ranked_dataset[
            LABEL_NAME
        ].describe()
    )

    print("\n排名标签描述统计：")
    print(
        ranked_dataset[
            RANK_LABEL_NAME
        ].describe()
    )

    print("\n排名特征缺失比例：")
    print(
        ranked_dataset[
            feature_names
        ]
        .isna()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print("\n保存位置：")
    print(output_path)


if __name__ == "__main__":
    main()
