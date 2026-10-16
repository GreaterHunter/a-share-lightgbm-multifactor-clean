import numpy as np
import qlib
from qlib.config import REG_CN
from qlib.data import D

from src.config import PROJECT_ROOT, load_research_config
from src.factor_library import (
    FEATURE_EXPRESSIONS,
    LABEL_EXPRESSION,
    LABEL_NAME,
)

def main() -> None:
    config = load_research_config()

    provider_uri = PROJECT_ROOT / config["provider_uri"]

    qlib.init(provider_uri= str(provider_uri), region=REG_CN)

    feature_names = list(
        FEATURE_EXPRESSIONS.keys()
    )

    feture_expressions = list(
        FEATURE_EXPRESSIONS.values()
    )

    fields = feture_expressions + [LABEL_EXPRESSION]

    column_names = feature_names + [LABEL_NAME]

    instruments = D.instruments(config["universe"])

    dataset = D.features(
        instruments=instruments,
        fields=fields,
        start_time=config["train_start"],
        end_time=config["test_end"],
        freq="day",
    )

    dataset.columns = column_names

    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    dataset = dataset.dropna(subset=[LABEL_NAME])

    all_features_missing = (
        dataset[feature_names]
        .isna()
        .all(axis=1)
    )

    dataset = dataset.loc[
        ~all_features_missing
    ]

    dataset = dataset.sort_index()

    missing_ratio = (
        dataset[feature_names]
        .isna()
        .mean()
        .sort_values(ascending=False)
    )

    datetimes = dataset.index.get_level_values("datetime")

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "multifactor_dataset.parquet"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(output_path)

    print("多因子数据集前10行：")
    print(dataset.head(10))

    print("\n数据形状（行数, 列数）：")
    print(dataset.shape)

    print("\n实际数据时间范围：")
    print(f"{datetimes.min()} 至 {datetimes.max()}")

    print("\n各特征缺失比例：")
    print(missing_ratio)

    print("\n特征数量：")
    print(len(feature_names))

    print("\n保存位置：")
    print(output_path)


if __name__ == "__main__":
    main()