"""Train the main model with annual walk-forward time-series validation."""

from __future__ import annotations

import gc
import hashlib
import json
import warnings
from copy import deepcopy
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import qlib
from qlib.config import REG_CN

from src.analyze_portfolio_robustness import run_tiered_backtest
from src.compare_models import (
    enrich_daily_report,
    summarize_report,
    summarize_report_by_year,
)
from src.config import (
    PROJECT_ROOT,
    load_main_strategy_config,
    load_model_config,
    load_research_config,
)
from src.evaluate_model import calculate_daily_model_ic
from src.factor_library import ENHANCED_FEATURE_NAMES
from src.forward_labels import label_contract, select_mature_period
from src.run_horizon_10_experiment import build_horizon_dataset
from src.run_temporal_ablation import save_prediction_frame, select_period
from src.train_lightgbm import build_prediction_frame


EXPERIMENT_NAME = "main_open_to_open_walk_forward_cv"
MODEL_NAME = "ranked_enhanced_open_10d"


def select_training_period(dataset, start, end, horizon, raw_label_name):
    if raw_label_name.startswith("future_open_return_"):
        return select_mature_period(dataset, start, end)
    # Explicit historical experiments still use their original closing labels.
    return select_period(dataset, start, end, horizon, purge_end=True)


def build_validation_folds(
    test_year: int,
    maximum_folds: int,
    data_start_year: int,
) -> list[dict[str, str]]:
    """Build expanding-window one-year time-series validation folds."""
    folds = []
    first_validation_year = test_year - maximum_folds
    for validation_year in range(first_validation_year, test_year):
        if validation_year <= data_start_year:
            continue
        folds.append(
            {
                "validation_year": str(validation_year),
                "train_start": f"{data_start_year}-01-01",
                "train_end": f"{validation_year - 1}-12-31",
                "valid_start": f"{validation_year}-01-01",
                "valid_end": f"{validation_year}-12-31",
            }
        )
    if len(folds) < 2:
        raise ValueError(
            f"测试年{test_year}只能构造{len(folds)}个验证折，至少需要2折"
        )
    return folds


def latest_training_period(
    test_year: int,
    data_start_year: int,
) -> tuple[str, str]:
    """Return the expanding window used by the final test-year model."""
    return (
        f"{data_start_year}-01-01",
        f"{test_year - 1}-12-31",
    )


def fit_selection_fold(
    dataset: pd.DataFrame,
    fold: dict[str, str],
    feature_names: list[str],
    target_name: str,
    raw_label_name: str,
    model_parameters: dict,
    iteration_candidates: list[int],
    label_horizon: int,
    min_cross_section_size: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate every fixed tree-count candidate on one validation fold."""
    train_data = select_training_period(
        dataset,
        fold["train_start"],
        fold["train_end"],
        label_horizon,
        raw_label_name,
    )
    valid_data = select_training_period(
        dataset,
        fold["valid_start"],
        fold["valid_end"],
        label_horizon,
        raw_label_name,
    )
    fit_parameters = deepcopy(model_parameters)
    fit_parameters["n_estimators"] = max(iteration_candidates)
    model = lgb.LGBMRegressor(**fit_parameters)
    model.fit(
        train_data[feature_names],
        train_data[target_name],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    records = []
    for candidate_iteration in iteration_candidates:
        predictions = model.predict(
            valid_data[feature_names],
            num_iteration=candidate_iteration,
        )
        prediction_frame = build_prediction_frame(
            frame=valid_data,
            predictions=predictions,
            retained_columns=[raw_label_name, target_name],
        )
        daily_ic = calculate_daily_model_ic(
            prediction_frame=prediction_frame,
            min_cross_section_size=min_cross_section_size,
            label_name=raw_label_name,
        )
        validation_mse = float(
            (
                prediction_frame["prediction"]
                - prediction_frame[target_name]
            ).pow(2).mean()
        )
        records.append(
            {
                **fold,
                "candidate_iteration": candidate_iteration,
                "train_rows": len(train_data),
                "valid_rows": len(valid_data),
                "train_label_end_max": str(train_data.label_end_date.max()) if "label_end_date" in train_data else None,
                "valid_label_end_max": str(valid_data.label_end_date.max()) if "label_end_date" in valid_data else None,
                "valid_mse": validation_mse,
                "valid_mean_ic": float(daily_ic["ic"].mean()),
                "valid_mean_rank_ic": float(daily_ic["rank_ic"].mean()),
                "valid_positive_rank_ic_ratio": float(
                    (daily_ic["rank_ic"] > 0).mean()
                ),
            }
        )
    return records


def summarize_iteration_candidates(
    fold_records: list[dict[str, float | int | str]],
) -> pd.DataFrame:
    """Aggregate candidate performance across folds and rank candidates."""
    records = pd.DataFrame(fold_records)
    summary = (
        records.groupby("candidate_iteration", as_index=False)
        .agg(
            fold_count=("validation_year", "nunique"),
            mean_validation_rank_ic=("valid_mean_rank_ic", "mean"),
            std_validation_rank_ic=("valid_mean_rank_ic", "std"),
            minimum_validation_rank_ic=("valid_mean_rank_ic", "min"),
            mean_validation_ic=("valid_mean_ic", "mean"),
            mean_validation_mse=("valid_mse", "mean"),
        )
        .sort_values(
            [
                "mean_validation_rank_ic",
                "std_validation_rank_ic",
                "candidate_iteration",
            ],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    return summary


def fit_final_year_model(
    dataset: pd.DataFrame,
    test_year: int,
    chosen_iteration: int,
    feature_names: list[str],
    target_name: str,
    raw_label_name: str,
    model_parameters: dict,
    data_start_year: int,
    label_horizon: int,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Refit a fresh final model on all history available before test year."""
    train_start, train_end = latest_training_period(
        test_year,
        data_start_year,
    )
    final_train = select_training_period(
        dataset,
        train_start,
        train_end,
        label_horizon,
        raw_label_name,
    )
    test_data = select_training_period(
        dataset,
        f"{test_year}-01-01",
        f"{test_year}-12-31",
        label_horizon,
        raw_label_name,
    )
    final_parameters = deepcopy(model_parameters)
    final_parameters["n_estimators"] = chosen_iteration
    final_model = lgb.LGBMRegressor(**final_parameters)
    final_model.fit(
        final_train[feature_names],
        final_train[target_name],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    predictions = final_model.predict(test_data[feature_names])
    prediction_frame = build_prediction_frame(
        frame=test_data,
        predictions=predictions,
        retained_columns=[raw_label_name, target_name],
    )
    model_dir = PROJECT_ROOT / "models" / "experiments" / EXPERIMENT_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL_NAME}_test_{test_year}.txt"
    final_model.booster_.save_model(str(model_path))
    metadata = {**label_contract(label_horizon), "test_year": test_year,
                "training_information_cutoff": train_end,
                "last_label_end_date": str(final_train.label_end_date.max()),
                "chosen_iteration": chosen_iteration,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return prediction_frame, {
        "test_year": test_year,
        "chosen_iteration": chosen_iteration,
        "final_train_start": train_start,
        "final_train_end": train_end,
        "final_train_rows": len(final_train),
        "test_rows": len(test_data),
    }


def run_walk_forward_training() -> Path:
    research = load_research_config()
    strategy = load_main_strategy_config()
    model_config = load_model_config()
    model_config["parameters"]["n_jobs"] = 4
    horizon = int(strategy["label_horizon"])
    validation_folds = int(strategy["validation_folds"])
    iteration_candidates = [
        int(value) for value in strategy["iteration_candidates"]
    ]
    test_start_year = int(strategy["test_start_year"])
    test_end_year = int(strategy["test_end_year"])
    data_start_year = pd.Timestamp(research["train_start"]).year
    min_cross_section_size = int(research["min_cross_section_size"])

    qlib.init(
        provider_uri=str(PROJECT_ROOT / research["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    dataset, raw_label, rank_label = build_horizon_dataset(research, horizon, strategy["label_price_basis"])
    all_predictions = []
    fold_records = []
    candidate_summary_records = []
    yearly_records = []
    validation_cache: dict[
        str, list[dict[str, float | int | str]]
    ] = {}

    for test_year in range(test_start_year, test_end_year + 1):
        folds = build_validation_folds(
            test_year=test_year,
            maximum_folds=validation_folds,
            data_start_year=data_start_year,
        )
        print(f"\n测试年份 {test_year}：执行 {len(folds)} 个时间序列验证折")
        current_fold_records = []
        for fold in folds:
            validation_year = fold["validation_year"]
            if validation_year not in validation_cache:
                print(
                    f"  训练 {fold['train_start']} 至 {fold['train_end']}，"
                    f"验证 {fold['valid_start']} 至 {fold['valid_end']}",
                    flush=True,
                )
                validation_cache[validation_year] = fit_selection_fold(
                    dataset=dataset,
                    fold=fold,
                    feature_names=ENHANCED_FEATURE_NAMES,
                    target_name=rank_label,
                    raw_label_name=raw_label,
                    model_parameters=model_config["parameters"],
                    iteration_candidates=iteration_candidates,
                    label_horizon=horizon,
                    min_cross_section_size=min_cross_section_size,
                )
            else:
                print(f"  复用验证年份 {validation_year} 的训练证据")
            for cached_record in validation_cache[validation_year]:
                record = dict(cached_record)
                record["test_year"] = test_year
                fold_records.append(record)
                current_fold_records.append(record)

        candidate_summary = summarize_iteration_candidates(
            current_fold_records
        )
        candidate_summary.insert(0, "test_year", test_year)
        candidate_summary_records.append(candidate_summary)
        chosen_iteration = int(candidate_summary.iloc[0]["candidate_iteration"])
        print(
            "  候选轮数跨折表现：\n"
            + candidate_summary.to_string(index=False)
            + f"\n  选择平均验证 Rank IC 最高的轮数：{chosen_iteration}",
            flush=True,
        )
        test_predictions, yearly_record = fit_final_year_model(
            dataset=dataset,
            test_year=test_year,
            chosen_iteration=chosen_iteration,
            feature_names=ENHANCED_FEATURE_NAMES,
            target_name=rank_label,
            raw_label_name=raw_label,
            model_parameters=model_config["parameters"],
            data_start_year=data_start_year,
            label_horizon=horizon,
        )
        yearly_record["validation_fold_count"] = len(folds)
        yearly_record["selection_mean_validation_rank_ic"] = float(
            candidate_summary.iloc[0]["mean_validation_rank_ic"]
        )
        yearly_records.append(yearly_record)
        all_predictions.append(test_predictions)

    combined = pd.concat(all_predictions).sort_index()
    if combined.index.duplicated().any():
        raise ValueError("主模型滚动预测包含重复股票日期")
    years = combined.index.get_level_values("datetime").year
    if years.min() != test_start_year or years.max() != test_end_year:
        raise ValueError("主模型预测年份范围与配置不一致")

    prediction_path = save_prediction_frame(
        combined,
        EXPERIMENT_NAME,
        MODEL_NAME,
        "test",
    )
    output_dir = PROJECT_ROOT / "reports" / "experiments" / EXPERIMENT_NAME
    pd.DataFrame(fold_records).to_csv(
        output_dir / "validation_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(candidate_summary_records, ignore_index=True).to_csv(
        output_dir / "iteration_candidate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(yearly_records).to_csv(
        output_dir / "training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n多折年度训练摘要：")
    print(pd.DataFrame(yearly_records).to_string(index=False))
    del dataset, combined
    gc.collect()
    return prediction_path


def run_walk_forward_backtest() -> None:
    """Backtest the combined 2022-2024 out-of-sample predictions."""
    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
    )
    research = load_research_config()
    strategy = load_main_strategy_config()
    config = {**research, **strategy}
    qlib.init(
        provider_uri=str(PROJECT_ROOT / research["provider_uri"]),
        region=REG_CN,
        kernels=1,
    )
    prediction_path = (
        PROJECT_ROOT
        / "reports"
        / "experiments"
        / EXPERIMENT_NAME
        / f"{MODEL_NAME}_test_predictions.parquet"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"缺少多折主模型预测：{prediction_path}；请先运行训练阶段"
        )
    predictions = pd.read_parquet(prediction_path).sort_index()
    for year in range(int(strategy["test_start_year"]), int(strategy["test_end_year"]) + 1):
        model_path = PROJECT_ROOT / "models" / "experiments" / EXPERIMENT_NAME / f"{MODEL_NAME}_test_{year}.txt"
        metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
        expected = label_contract(int(strategy["label_horizon"]))
        if (any(metadata.get(key) != expected[key] for key in ["label_price_basis", "label_expression", "target"])
                or metadata["model_sha256"] != hashlib.sha256(model_path.read_bytes()).hexdigest()):
            raise ValueError("回测模型元数据、标签口径或模型哈希不一致")
    raw_label = label_contract(int(strategy["label_horizon"]))["raw_label"]
    if raw_label not in predictions:
        raise ValueError("拒绝使用旧收盘标签的缓存预测，必须重跑开盘→开盘训练")
    portfolio_name = "ranked_enhanced_open_10d_rebalance_10d_walk_forward_cv"
    raw_report = run_tiered_backtest(
        prediction_frame=predictions,
        config=config,
        rebalance_every=int(strategy["rebalance_every"]),
        top_k=int(strategy["top_k"]),
        top_tier_size=int(strategy["top_tier_size"]),
        top_tier_weight=float(strategy["top_tier_weight"]),
        period_end=strategy["test_end"],
    )
    report = enrich_daily_report(raw_report, portfolio_name)
    summary = summarize_report(report, portfolio_name)
    yearly = summarize_report_by_year(report, portfolio_name)
    ic_records = []
    dates = predictions.index.get_level_values("datetime")
    for year in range(
        int(strategy["test_start_year"]),
        int(strategy["test_end_year"]) + 1,
    ):
        yearly_predictions = predictions.loc[dates.year == year]
        daily_ic = calculate_daily_model_ic(
            prediction_frame=yearly_predictions,
            min_cross_section_size=int(research["min_cross_section_size"]),
            label_name=raw_label,
        )
        ic_records.append(
            {
                "year": year,
                "valid_days": len(daily_ic),
                "mean_ic": daily_ic["ic"].mean(),
                "mean_rank_ic": daily_ic["rank_ic"].mean(),
                "positive_rank_ic_ratio": (daily_ic["rank_ic"] > 0).mean(),
            }
        )
    output_dir = PROJECT_ROOT / "reports" / "main_strategy_open_to_open"
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "backtest_daily.csv", encoding="utf-8-sig")
    summary.to_csv(
        output_dir / "backtest_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    yearly.to_csv(
        output_dir / "backtest_yearly.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(ic_records).to_csv(
        output_dir / "test_ic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n多折主策略总体回测：")
    print(summary.to_string(index=False))
    print("\n多折主策略逐年回测：")
    print(yearly.to_string(index=False))


if __name__ == "__main__":
    run_walk_forward_training()
    run_walk_forward_backtest()
