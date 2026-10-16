"""Build daily cross-sectional ranks for all 15 enhanced factors."""

import pandas as pd

from src.build_ranked_dataset import (
    build_ranked_dataset,
    validate_ranked_dataset,
)
from src.config import PROJECT_ROOT
from src.factor_library import ENHANCED_FEATURE_NAMES, LABEL_NAME


def main() -> None:
    input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "enhanced_multifactor_dataset.parquet"
    )
    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ranked_enhanced_multifactor_dataset.parquet"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"找不到增强数据集：{input_path}")

    dataset = pd.read_parquet(input_path)
    ranked_dataset = build_ranked_dataset(
        dataset=dataset,
        feature_names=ENHANCED_FEATURE_NAMES,
    )
    validate_ranked_dataset(
        ranked_dataset=ranked_dataset,
        feature_names=ENHANCED_FEATURE_NAMES,
    )

    if not ranked_dataset[LABEL_NAME].equals(dataset[LABEL_NAME]):
        raise ValueError("排名处理意外修改了原始收益标签")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_dataset.to_parquet(output_path)

    print("15因子排名增强数据集构建完成")
    print(f"数据形状：{ranked_dataset.shape}")
    print(f"特征数量：{len(ENHANCED_FEATURE_NAMES)}")
    print(f"保存位置：{output_path}")


if __name__ == "__main__":
    main()
