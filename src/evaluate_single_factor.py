import pandas as pd

from src.config import PROJECT_ROOT, load_research_config


def calculate_one_day_rank_ic(
        day_panel: pd.DataFrame,
        factor_name: str,
        label_name: str,
) -> pd.Series:
    factor_values = day_panel[factor_name]
    future_returns =  day_panel[label_name]

    ic = factor_values.corr(
        future_returns,
        method="pearson",
    )

    rank_ic =  factor_values.corr(
        future_returns,
        method="spearman",
    )

    return pd.Series(
        {
            "ic": ic,
            "rank_ic": rank_ic,
        }
    )

def main() -> None:
    config = load_research_config()

    factor_window =  config["factor_window"]
    label_horizon = config["label_horizon"]

    factor_name = f"mom_{factor_window}d"
    label_name = f"future_return_{label_horizon}d"

    factor_panel_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "factor_panel_sample.parquet"
    )

    factor_panel = pd.read_parquet(factor_panel_path)

    if factor_name not in factor_panel.columns:
        raise ValueError(f"缺少因子列: {factor_name}")
    if label_name not in factor_panel.columns:
        raise ValueError(f"缺少标签列：{label_name}")

    if "datetime" not in factor_panel.index.names:
        raise ValueError("因子面板索引缺少 datetime")

    daily_ic = (
        factor_panel.groupby(level = "datetime", sort=True)
        .apply(
            lambda  day_panel: calculate_one_day_rank_ic(
            day_panel = day_panel,
            factor_name = factor_name,
            label_name = label_name,)
               )
        .dropna(how = "all")
    )

    if daily_ic.empty:
        raise ValueError("没有可用于计算 IC 的有效交易日")
    mean_ic = daily_ic["ic"].mean()
    std_ic = daily_ic["ic"].std()
    icir = mean_ic/std_ic if std_ic != 0 else float("nan")
    positive_ic_ratio = (daily_ic["ic"] > 0).mean()

    mean_rank_ic = daily_ic["rank_ic"].mean()
    std_rank_ic = daily_ic["rank_ic"].std()
    rank_icir = (
        mean_rank_ic/std_rank_ic
        if std_rank_ic != 0
        else float("nan")
    )
    positive_rank_ic_ratio = (daily_ic["rank_ic"] > 0).mean()

    summary = pd.DataFrame(
        {
            "mean_ic": [mean_ic],
            "std_ic": [std_ic],
            "icir": [icir],
            "positive_ic_ratio": [positive_ic_ratio],
            "mean_rank_ic": [mean_rank_ic],
            "std_rank_ic": [std_rank_ic],
            "rank_icir": [rank_icir],
            "positive_rank_ic_ratio": [positive_rank_ic_ratio],
        },
        index=[factor_name],
    )

    summary.index.name = "factor"

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    daily_ic_path = reports_dir / f"{factor_name}_daily_ic.csv"
    summary_path = reports_dir / f"{factor_name}_ic_summary.csv"

    daily_ic.to_csv(daily_ic_path, encoding="utf-8-sig")
    summary.to_csv(summary_path, encoding="utf-8-sig")

    print("每日 IC 与 Rank IC：")
    print(daily_ic)

    print("\nIC 汇总：")
    print(summary)


if __name__ == "__main__":
    main()






