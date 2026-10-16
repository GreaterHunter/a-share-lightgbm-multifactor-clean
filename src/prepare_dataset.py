import numpy as np
import pandas as pd

from src.config import  PROJECT_ROOT, load_research_config
from src.factor_library import FEATURE_EXPRESSIONS, LABEL_NAME

def validate_dataset(dataset: pd.DataFrame,
                     feature_names: list[str], ) -> pd.Series:
    expected_columns = feature_names + [LABEL_NAME]

    missing_columns = (
        set(expected_columns) - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(f"数据缺少列：{sorted(missing_columns)}")

    expected_index_names = ["instrument", "datetime"]

    if list(dataset.index.names) != expected_index_names:
        raise ValueError("数据集索引必须是 instrument 和 datetime")

    duplicate_count = int(dataset.index.duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(f"发现重复的股票日期样本：{duplicate_count}")

    numeric_values = dataset[expected_columns].to_numpy()

    infinite_count = int(np.isinf(numeric_values).sum())

    if infinite_count > 0:
        raise ValueError(
            f"发现无穷值：{infinite_count}"
        )

    missing_label_count = int(dataset[LABEL_NAME].isna().sum())

    if missing_label_count > 0:
        raise ValueError(
            f"发现标签缺失：{missing_label_count}"
        )

    all_features_missing_count = int(
        dataset[feature_names]
        .isna()
        .all(axis=1)
        .sum()
    )

    if all_features_missing_count > 0:
        raise ValueError(
            "发现全部特征缺失的样本："
            f"{all_features_missing_count}"
        )

    print("数据质量检查通过：")
    print(f"重复索引数量：{duplicate_count}")
    print(f"无穷值数量：{infinite_count}")
    print(f"标签缺失数量：{missing_label_count}")
    print(
        "全部特征缺失的样本数："
        f"{all_features_missing_count}"
    )

    missing_summary = (
        dataset[feature_names]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_ratio")
    )

    missing_summary.index.name = "factor"

    return missing_summary

def purge_last_signal_dates(
    frame: pd.DataFrame,
    label_horizon: int,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """清除标签尚未在当前时间区间内完全实现的末尾样本。"""

    if label_horizon <= 0:
        raise ValueError(
            "label_horizon必须为正整数"
        )

    dates = frame.index.get_level_values(
        "datetime"
    )

    unique_dates = (
        pd.DatetimeIndex(
            dates.unique()
        )
        .sort_values()
    )

    if len(unique_dates) <= label_horizon:
        raise ValueError(
            "交易日数量不足，"
            "无法执行标签边界清除"
        )

    purged_dates = unique_dates[
        -label_horizon:
    ]

    keep_mask = ~dates.isin(
        purged_dates
    )

    cleaned_frame = frame.loc[
        keep_mask
    ].copy()

    return cleaned_frame, purged_dates

def split_dataset(
    dataset: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:

    train_start = pd.Timestamp(config["train_start"])
    train_end = pd.Timestamp(config["train_end"])

    valid_start = pd.Timestamp(config["valid_start"])
    valid_end = pd.Timestamp(config["valid_end"])

    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])

    if not (
        train_start <= train_end
        < valid_start <= valid_end
        < test_start <= test_end
    ):
        raise ValueError("训练、验证、测试时间配置错误或发生重叠")

    dates = dataset.index.get_level_values("datetime")

    train_mask = (
    (dates >= train_start)
    & (dates <= train_end)
    )

    valid_mask = (
        (dates >= valid_start)
        & (dates <= valid_end)
    )

    test_mask = (
        (dates >= test_start)
        & (dates <= test_end)
    )

    raw_splits = {
        "train": dataset.loc[train_mask].copy(),
        "valid": dataset.loc[valid_mask].copy(),
        "test": dataset.loc[test_mask].copy(),
    }

    for split_name, frame in raw_splits.items():
        if frame.empty:
            raise ValueError(
                f"{split_name} 数据集为空"
            )

    assigned_rows = sum(
        len(frame)
        for frame in raw_splits.values()
    )

    if assigned_rows != len(dataset):
        raise ValueError(
            "部分样本没有被分配到任何时间区间："
            f"总行数={len(dataset)}，"
            f"已分配={assigned_rows}"
        )

    label_horizon = int(
        config["label_horizon"]
    )

    splits = {}

    for split_name, frame in raw_splits.items():
        cleaned_frame, purged_dates = (
            purge_last_signal_dates(
                frame=frame,
                label_horizon=label_horizon,
            )
        )

        splits[split_name] = cleaned_frame

        removed_rows = (
                len(frame) - len(cleaned_frame)
        )

        print(
            f"{split_name}清除最后"
            f"{label_horizon}个信号日："
            f"{purged_dates.min().date()} 至 "
            f"{purged_dates.max().date()}，"
            f"删除{removed_rows}行"
        )

    return splits

def build_split_summary(
        splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    records = []

    for split_name, frame in splits.items():
        dates = frame.index.get_level_values("datetime")

        instruments = frame.index.get_level_values("instrument")

        records.append(
            {
                "split": split_name,
                "rows": len(frame),
                "instruments": instruments.nunique(),
                "start_date": dates.min(),
                "end_date": dates.max(),
                "label_mean": frame[LABEL_NAME].mean(),
                "label_std": frame[LABEL_NAME].std(),
            }
        )

    return pd.DataFrame(records).set_index("split")

def main() -> None:
    config = load_research_config()

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "multifactor_dataset.parquet"
    )

    dataset = pd.read_parquet(dataset_path)

    feature_names = list(FEATURE_EXPRESSIONS.keys())

    missing_summary = validate_dataset(dataset, feature_names)

    splits = split_dataset(dataset, config)

    split_summary = build_split_summary(splits)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    missing_summary_path = (
        reports_dir
        / "feature_missing_summary.csv"
    )

    split_summary_path = (
        reports_dir
        / "data_split_summary.csv"
    )

    missing_summary.to_csv(missing_summary_path, encoding="utf-8-sig",)
    split_summary.to_csv(split_summary_path, encoding="utf-8-sig",)

    print("\n特征缺失比例：")
    print(missing_summary)

    print("\n数据集切分汇总：")
    print(split_summary)

    print("\n报告保存位置：")
    print(missing_summary_path)
    print(split_summary_path)


if __name__ == "__main__":
    main()
