import argparse
import sys
import types
import warnings

import numpy as np
import pandas as pd
import qlib

from qlib.backtest.executor import (
    SimulatorExecutor,
)
from qlib.config import REG_CN
from qlib.contrib.evaluate import (
    backtest_daily,
)
from qlib.data import D

from src.config import (
    PROJECT_ROOT,
    load_research_config,
)
from src.evaluate_model import (
    validate_prediction_frame,
)


VALIDATION_MODEL_NAMES = [
    "lightgbm_baseline",
    "lightgbm_ranked",
    "lightgbm_enhanced",
]

TEST_MODEL_NAMES = [
    "lightgbm_baseline",
]


def load_topk_dropout_strategy():
    """
    避免Qlib 0.9.7在Windows中加载本项目不使用的
    cvxpy/osqp增强指数优化模块。
    """
    module_name = (
        "qlib.contrib.strategy.optimizer"
    )

    if module_name not in sys.modules:
        optimizer_stub = types.ModuleType(
            module_name
        )

        class EnhancedIndexingOptimizer:
            def __init__(
                self,
                *args,
                **kwargs,
            ):
                raise RuntimeError(
                    "本项目没有启用增强指数优化器"
                )

        optimizer_stub.EnhancedIndexingOptimizer = (
            EnhancedIndexingOptimizer
        )

        sys.modules[
            module_name
        ] = optimizer_stub

    from qlib.contrib.strategy.signal_strategy import (
        TopkDropoutStrategy,
    )

    return TopkDropoutStrategy


def load_prediction_frame(
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    prediction_path = (
        PROJECT_ROOT
        / "reports"
        / "model"
        / f"{model_name}_{split_name}_predictions.parquet"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"找不到预测文件："
            f"{prediction_path}"
        )

    prediction_frame = pd.read_parquet(
        prediction_path
    )

    validate_prediction_frame(
        prediction_frame
    )

    return prediction_frame


def validate_comparable_frames(
    prediction_frames: dict[str, pd.DataFrame],
    model_names: list[str],
) -> None:
    reference_frame = prediction_frames[
        model_names[0]
    ]

    for model_name in model_names[1:]:
        comparison_frame = (
            prediction_frames[
                model_name
            ]
        )

        if not comparison_frame.index.equals(
            reference_frame.index
        ):
            raise ValueError(
                f"{model_name}与基准模型"
                "使用的股票日期样本不一致"
            )


def build_rebalance_signal(
    prediction_frame: pd.DataFrame,
    rebalance_every: int,
) -> tuple[pd.Series, pd.DatetimeIndex]:
    if rebalance_every <= 0:
        raise ValueError(
            "rebalance_every必须是正整数"
        )

    datetimes = (
        prediction_frame.index
        .get_level_values("datetime")
    )

    unique_dates = pd.DatetimeIndex(
        datetimes.unique()
    ).sort_values()

    signal_dates = unique_dates[
        ::rebalance_every
    ]

    if len(signal_dates) == 0:
        raise ValueError(
            "没有可用的调仓信号日期"
        )

    signal_mask = datetimes.isin(
        signal_dates
    )

    signal = prediction_frame.loc[
        signal_mask,
        "prediction",
    ].copy()

    signal = signal.sort_index()
    signal.name = "score"

    return signal, signal_dates


def find_first_trade_date(
    first_signal_date: pd.Timestamp,
    period_end: str,
) -> pd.Timestamp:
    calendar = pd.DatetimeIndex(
        D.calendar(
            start_time=first_signal_date,
            end_time=period_end,
            freq="day",
        )
    )

    if first_signal_date not in calendar:
        raise ValueError(
            "第一期信号日不在Qlib交易日历中"
        )

    signal_location = calendar.get_loc(
        first_signal_date
    )

    next_location = signal_location + 1

    if next_location >= len(calendar):
        raise ValueError(
            "第一期信号之后没有可交易日期"
        )

    return calendar[
        next_location
    ]


def run_one_model_backtest(
    model_name: str,
    prediction_frame: pd.DataFrame,
    config: dict,
    period_end: str,
) -> pd.DataFrame:
    topk_dropout_strategy = (
        load_topk_dropout_strategy()
    )

    signal, signal_dates = (
        build_rebalance_signal(
            prediction_frame=prediction_frame,
            rebalance_every=int(
                config[
                    "rebalance_every"
                ]
            ),
        )
    )

    first_trade_date = (
        find_first_trade_date(
            first_signal_date=(
                signal_dates[0]
            ),
            period_end=period_end,
        )
    )

    strategy = topk_dropout_strategy(
        signal=signal,
        topk=int(
            config["top_k"]
        ),
        n_drop=int(
            config["n_drop"]
        ),
        hold_thresh=1,
        only_tradable=True,
        forbid_all_trade_at_limit=False,
        risk_degree=float(
            config["risk_degree"]
        ),
    )

    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        verbose=False,
    )

    print(
        f"\n开始回测：{model_name}"
    )
    print(
        f"信号日期数量："
        f"{len(signal_dates)}"
    )
    print(
        f"首次交易日期："
        f"{first_trade_date.date()}"
    )

    report, _ = backtest_daily(
        start_time=first_trade_date,
        end_time=pd.Timestamp(
            period_end
        ),
        strategy=strategy,
        executor=executor,
        account=float(
            config["initial_cash"]
        ),
        benchmark=str(
            config["benchmark"]
        ),
        exchange_kwargs={
            "limit_threshold": float(
                config[
                    "limit_threshold"
                ]
            ),
            "deal_price": "open",
            "open_cost": float(
                config["buy_cost"]
            ),
            "close_cost": float(
                config["sell_cost"]
            ),
            "min_cost": float(
                config[
                    "minimum_cost"
                ]
            ),
            "trade_unit": 100,
        },
    )

    if report.empty:
        raise ValueError(
            f"{model_name}没有产生回测结果"
        )

    return report


def enrich_daily_report(
    report: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    required_columns = {
        "return",
        "cost",
        "bench",
        "turnover",
    }

    missing_columns = (
        required_columns
        - set(report.columns)
    )

    if missing_columns:
        raise ValueError(
            "Qlib报告缺少字段："
            f"{sorted(missing_columns)}"
        )

    result = report.copy()

    result["net_return"] = (
        result["return"]
        - result["cost"]
    )

    result["active_return"] = (
        result["net_return"]
        - result["bench"]
    )

    result["portfolio_nav"] = (
        1.0 + result["net_return"]
    ).cumprod()

    result["benchmark_nav"] = (
        1.0 + result["bench"]
    ).cumprod()

    result["relative_excess_nav"] = (
        result["portfolio_nav"]
        / result["benchmark_nav"]
    )

    result.insert(
        loc=0,
        column="model",
        value=model_name,
    )

    return result


def calculate_max_drawdown(
    daily_returns: pd.Series,
) -> float:
    wealth = (
        1.0 + daily_returns
    ).cumprod()

    wealth_with_initial = pd.concat(
        [
            pd.Series(
                [1.0],
                dtype="float64",
            ),
            wealth.reset_index(
                drop=True
            ),
        ],
        ignore_index=True,
    )

    running_peak = (
        wealth_with_initial.cummax()
    )

    drawdown = (
        wealth_with_initial
        / running_peak
        - 1.0
    )

    return float(
        drawdown.min()
    )


def summarize_report(
    report: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    net_returns = report[
        "net_return"
    ].fillna(0.0)

    benchmark_returns = report[
        "bench"
    ].fillna(0.0)

    active_returns = report[
        "active_return"
    ].fillna(0.0)

    trading_days = len(report)

    if trading_days == 0:
        raise ValueError(
            f"{model_name}没有有效交易日"
        )

    portfolio_wealth = (
        1.0 + net_returns
    ).prod()

    benchmark_wealth = (
        1.0 + benchmark_returns
    ).prod()

    annualized_return = (
        portfolio_wealth
        ** (252 / trading_days)
        - 1.0
    )

    benchmark_annualized_return = (
        benchmark_wealth
        ** (252 / trading_days)
        - 1.0
    )

    annualized_excess_return = (
        (
            portfolio_wealth
            / benchmark_wealth
        )
        ** (252 / trading_days)
        - 1.0
    )

    net_std = net_returns.std(
        ddof=1
    )

    active_std = active_returns.std(
        ddof=1
    )

    sharpe = (
        net_returns.mean()
        / net_std
        * np.sqrt(252)
        if net_std > 0
        else float("nan")
    )

    information_ratio = (
        active_returns.mean()
        / active_std
        * np.sqrt(252)
        if active_std > 0
        else float("nan")
    )

    return pd.DataFrame(
        [
            {
                "model": model_name,
                "trading_days": (
                    trading_days
                ),
                "total_return": (
                    portfolio_wealth
                    - 1.0
                ),
                "annualized_return": (
                    annualized_return
                ),
                "benchmark_total_return": (
                    benchmark_wealth
                    - 1.0
                ),
                "benchmark_annualized_return": (
                    benchmark_annualized_return
                ),
                "total_excess_return": (
                    portfolio_wealth
                    / benchmark_wealth
                    - 1.0
                ),
                "annualized_excess_return": (
                    annualized_excess_return
                ),
                "sharpe": sharpe,
                "information_ratio": (
                    information_ratio
                ),
                "max_drawdown": (
                    calculate_max_drawdown(
                        net_returns
                    )
                ),
                "average_daily_turnover": (
                    report[
                        "turnover"
                    ].mean()
                ),
                "total_cost_rate_sum": (
                    report[
                        "cost"
                    ].sum()
                ),
                "positive_active_day_ratio": (
                    (
                        active_returns > 0
                    ).mean()
                ),
            }
        ]
    )


def summarize_report_by_year(
    report: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    yearly_summaries = []

    for year, year_report in report.groupby(
        report.index.year,
        sort=True,
    ):
        year_summary = summarize_report(
            report=year_report,
            model_name=model_name,
        )
        year_summary.insert(
            loc=1,
            column="year",
            value=int(year),
        )
        yearly_summaries.append(year_summary)

    return pd.concat(
        yearly_summaries,
        ignore_index=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Qlib account backtest for saved predictions.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Artifact model names; defaults depend on the split.",
    )
    return parser.parse_args()


def main(
    split_name: str = "valid",
    model_names: list[str] | None = None,
) -> None:
    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
    )

    config = load_research_config()

    if model_names is None:
        model_names = (
            VALIDATION_MODEL_NAMES
            if split_name == "valid"
            else TEST_MODEL_NAMES
        )

    period_end = config[f"{split_name}_end"]

    provider_uri = (
        PROJECT_ROOT
        / config["provider_uri"]
    )

    qlib.init(
        provider_uri=str(
            provider_uri
        ),
        region=REG_CN,
        kernels=1,
    )

    prediction_frames = {
        model_name: load_prediction_frame(
            model_name=model_name,
            split_name=split_name,
        )
        for model_name in model_names
    }

    validate_comparable_frames(
        prediction_frames=prediction_frames,
        model_names=model_names,
    )

    all_reports = []
    all_summaries = []
    all_yearly_summaries = []

    for model_name in model_names:
        raw_report = (
            run_one_model_backtest(
                model_name=model_name,
                prediction_frame=(
                    prediction_frames[
                        model_name
                    ]
                ),
                config=config,
                period_end=period_end,
            )
        )

        report = enrich_daily_report(
            report=raw_report,
            model_name=model_name,
        )

        summary = summarize_report(
            report=report,
            model_name=model_name,
        )

        all_reports.append(
            report
        )

        all_summaries.append(
            summary
        )

        if split_name == "test":
            all_yearly_summaries.append(
                summarize_report_by_year(
                    report=report,
                    model_name=model_name,
                )
            )

    daily_comparison = pd.concat(
        all_reports,
        axis=0,
    )

    model_comparison = pd.concat(
        all_summaries,
        ignore_index=True,
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

    report_prefix = (
        "validation"
        if split_name == "valid"
        else "test"
    )

    daily_path = (
        reports_dir
        / f"{report_prefix}_daily_backtest.csv"
    )

    comparison_path = (
        reports_dir
        / f"{report_prefix}_model_comparison.csv"
    )

    daily_comparison.to_csv(
        daily_path,
        encoding="utf-8-sig",
    )

    model_comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n{split_name}账户式回测完成"
    )

    print("\n模型收益对比：")
    print(
        model_comparison.to_string(
            index=False
        )
    )

    print("\n保存位置：")
    print(daily_path)
    print(comparison_path)

    if split_name == "test":
        yearly_comparison = pd.concat(
            all_yearly_summaries,
            ignore_index=True,
        )
        yearly_path = (
            reports_dir
            / "test_yearly_performance.csv"
        )
        yearly_comparison.to_csv(
            yearly_path,
            index=False,
            encoding="utf-8-sig",
        )
        print("\n逐年测试结果：")
        print(yearly_comparison.to_string(index=False))
        print(yearly_path)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(
        split_name=arguments.split,
        model_names=arguments.models,
    )
