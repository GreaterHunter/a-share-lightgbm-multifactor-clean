import pandas as pd

from src.config import PROJECT_ROOT, load_research_config
from src.factor_library import FEATURE_EXPRESSIONS, LABEL_NAME
from src.prepare_dataset import split_dataset

def calculate_one_day_factor_ic(
        day_panel: pd.DataFrame,
        factor_name: str,
        min_cross_section_size: int,
        label_name: str = LABEL_NAME,

) -> dict[str, float | int]:
    pair = (
        day_panel[
            [factor_name, label_name]
        ]
        .dropna()
    )

    observations = len(pair)

    has_enough_observations = (
        observations >= min_cross_section_size
    )

    factor_has_variation = (
        pair[factor_name].nunique() >= 2
    )

    label_has_variation = (
        pair[label_name].nunique() >= 2
    )

    if not(
        has_enough_observations
        and factor_has_variation
        and label_has_variation
    ):
        return {
        "observations": observations,
        "ic": float("nan"),
        "rank_ic": float("nan"),
    }

    ic = pair[factor_name].corr(
        pair[label_name],
        method="pearson",
    )

    rank_ic = pair[factor_name].corr(
        pair[label_name],
        method="spearman",
    )

    return {
        "observations": observations,
        "ic": float(ic),
        "rank_ic": float(rank_ic),
    }
def calculate_daily_factor_ic(
        frame: pd.DataFrame,
        split_name: str,
        feature_names: list[str],
        min_cross_section_size: int,
) -> pd.DataFrame:

    records = []

    daily_groups = frame.groupby(
        level="datetime",
        sort=True,
    )

    for datetime, day_panel in daily_groups:
        for factor_name in feature_names:
            result = calculate_one_day_factor_ic(
                day_panel,
                factor_name,
                min_cross_section_size,
            )
            records.append(
                {
                    "split": split_name,
                    "datetime": datetime,
                    "factor": factor_name,
                    "observations": result["observations"],
                    "ic": result["ic"],
                    "rank_ic": result["rank_ic"],
                }
            )

    return pd.DataFrame(records)

def calculate_information_ratio(
        mean_value: float,
        std_value: float,
) -> float:
    if pd.isna(std_value) or std_value == 0:
        return float("nan")

    return mean_value / std_value

def summarize_factor_ic(
        daily_ic: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    grouped = daily_ic.groupby(["split", "factor"], sort=True)
    for (
        split_name,
        factor_name,
    ), factor_daily_ic in grouped:
        valid_ic = factor_daily_ic[
            "ic"
        ].dropna()

        valid_rank_ic = factor_daily_ic[
            "rank_ic"
        ].dropna()

        mean_ic = valid_ic.mean()
        std_ic = valid_ic.std()

        mean_rank_ic = valid_rank_ic.mean()
        std_rank_ic = valid_rank_ic.std()

        records.append(
            {
                "split": split_name,
                "factor": factor_name,
                "valid_days": len(valid_ic),
                "mean_observations": (
                    factor_daily_ic[
                        "observations"
                    ].mean()
                ),
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": (
                    calculate_information_ratio(
                        mean_value=mean_ic,
                        std_value=std_ic,
                    )
                ),
                "positive_ic_ratio": (
                    (valid_ic > 0).mean()
                ),
                "mean_rank_ic": mean_rank_ic,
                "std_rank_ic": std_rank_ic,
                "rank_icir": (
                    calculate_information_ratio(
                        mean_value=mean_rank_ic,
                        std_value=std_rank_ic,
                    )
                ),
                "positive_rank_ic_ratio": (
                    (valid_rank_ic > 0).mean()
                ),
                "abs_mean_rank_ic": abs(
                    mean_rank_ic
                ),
            }
        )

    summary = pd.DataFrame(records)

    summary = summary.sort_values(
        by=[
            "split",
            "abs_mean_rank_ic",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return summary.reset_index(
        drop=True
    )

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

    min_cross_section_size = int(config["min_cross_section_size"])

    splits = split_dataset(dataset, config)

    analysis_split_names = ["train", "valid",]

    daily_results = []

    for split_name in analysis_split_names:
        split_daily_ic = (
            calculate_daily_factor_ic(
                frame=splits[split_name],
                split_name=split_name,
                feature_names=feature_names,
                min_cross_section_size=min_cross_section_size,
            )
        )

        daily_results.append(split_daily_ic)

    daily_ic = pd.concat(daily_results, ignore_index=True)

    summary = summarize_factor_ic(daily_ic)
    reports_dir = PROJECT_ROOT / "reports"

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_ic_path = (
            reports_dir
            / "factor_daily_ic_train_valid.parquet"
    )
    daily_ic.to_parquet(daily_ic_path, index = False)
    summary_path = (
            reports_dir
            / "factor_ic_summary_train_valid.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    display_columns = [
        "split",
        "factor",
        "valid_days",
        "mean_ic",
        "icir",
        "mean_rank_ic",
        "rank_icir",
        "positive_rank_ic_ratio",
    ]

    print("训练集与验证集因子IC汇总：")

    print(
        summary[
            display_columns
        ].to_string(index=False)
    )

    print("\n报告保存位置：")
    print(daily_ic_path)
    print(summary_path)


if __name__ == "__main__":
    main()



