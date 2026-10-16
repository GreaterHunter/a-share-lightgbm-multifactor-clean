from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(filename: str) -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / filename

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须包含键值映射：{config_path}")
    return config

def load_data_config() -> dict[str, Any]:
    return load_config("data.yaml")


def load_research_config() -> dict[str, Any]:
    return load_config("research.yaml")


def load_model_config() -> dict[str, Any]:
    return load_config("model.yaml")


def load_main_strategy_config() -> dict[str, Any]:
    """Load and validate the frozen production-candidate research settings."""
    config = load_config("main_strategy.yaml")
    required = {
        "strategy_name",
        "model_variant",
        "training_mode",
        "validation_folds",
        "iteration_candidates",
        "iteration_selection_metric",
        "test_start_year",
        "test_end_year",
        "test_end",
        "label_horizon",
        "label_price_basis",
        "execution_price_basis",
        "rebalance_every",
        "top_k",
        "n_drop",
        "top_tier_size",
        "top_tier_weight",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"主策略配置缺少字段：{sorted(missing)}")
    if config["model_variant"] != "ranked_enhanced":
        raise ValueError("主流程目前只支持 ranked_enhanced 模型")
    if config["label_price_basis"] != "open_to_open" or config["execution_price_basis"] != "next_session_open":
        raise ValueError("主策略的标签和执行必须统一为开盘→开盘")
    positive_integer_fields = [
        "label_horizon",
        "rebalance_every",
        "top_k",
        "n_drop",
        "top_tier_size",
    ]
    for field in positive_integer_fields:
        if not isinstance(config[field], int) or config[field] <= 0:
            raise ValueError(f"{field} 必须是正整数")
    if config["n_drop"] > config["top_k"]:
        raise ValueError("n_drop 不能大于 top_k")
    if config["top_tier_size"] > config["top_k"]:
        raise ValueError("top_tier_size 不能大于 top_k")
    if not 0 < float(config["top_tier_weight"]) < 1:
        raise ValueError("top_tier_weight 必须在 0 和 1 之间")
    if config["training_mode"] != "annual_expanding_time_series_cv":
        raise ValueError("主流程必须使用年度扩展窗口时间序列多折验证")
    candidates = config["iteration_candidates"]
    if (
        not isinstance(candidates, list)
        or len(candidates) < 2
        or any(not isinstance(value, int) or value <= 0 for value in candidates)
        or candidates != sorted(set(candidates))
    ):
        raise ValueError("iteration_candidates 必须是递增且不重复的正整数列表")
    if config["iteration_selection_metric"] != "mean_rank_ic":
        raise ValueError("主流程目前只支持按跨折平均 Rank IC 选择迭代轮数")
    if config["test_start_year"] > config["test_end_year"]:
        raise ValueError("test_start_year 不能晚于 test_end_year")
    if str(config["test_end"]) != f"{config['test_end_year']}-12-31":
        raise ValueError("test_end 必须与 test_end_year 一致")
    return config
