"""Generate presentation-ready figures for the frozen main strategy."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT


REPORT_DIR = PROJECT_ROOT / "reports" / "main_strategy_open_to_open"
FIGURE_DIR = REPORT_DIR / "figures"


def load_reports() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_path = REPORT_DIR / "backtest_daily.csv"
    yearly_path = REPORT_DIR / "backtest_yearly.csv"
    for path in (daily_path, yearly_path):
        if not path.exists():
            raise FileNotFoundError(
                f"缺少主策略报告：{path}；请先运行主策略回测"
            )
    daily = pd.read_csv(daily_path, parse_dates=["datetime"])
    yearly = pd.read_csv(yearly_path)
    daily = daily.sort_values("datetime").set_index("datetime")
    return daily, yearly


def plot_nav(daily: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(daily.index, daily["portfolio_nav"], label="Strategy", linewidth=2)
    axis.plot(daily.index, daily["benchmark_nav"], label="CSI300", linewidth=1.6)
    axis.plot(
        daily.index,
        daily["relative_excess_nav"],
        label="Relative excess",
        linewidth=1.6,
    )
    axis.set_title("Main Strategy NAV (2022-2024)")
    axis.set_ylabel("Net asset value")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "nav_curve.png", dpi=180)
    plt.close(figure)


def plot_drawdown(daily: pd.DataFrame) -> None:
    strategy_drawdown = daily["portfolio_nav"] / daily["portfolio_nav"].cummax() - 1
    benchmark_drawdown = daily["benchmark_nav"] / daily["benchmark_nav"].cummax() - 1
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.fill_between(
        daily.index,
        strategy_drawdown,
        0,
        label="Strategy",
        alpha=0.45,
    )
    axis.plot(
        daily.index,
        benchmark_drawdown,
        label="CSI300",
        linewidth=1.4,
    )
    axis.set_title("Drawdown")
    axis.set_ylabel("Drawdown")
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "drawdown.png", dpi=180)
    plt.close(figure)


def plot_yearly_returns(yearly: pd.DataFrame) -> None:
    years = yearly["year"].astype(str)
    width = 0.36
    positions = range(len(yearly))
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(
        [position - width / 2 for position in positions],
        yearly["annualized_return"],
        width=width,
        label="Strategy",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        yearly["benchmark_annualized_return"],
        width=width,
        label="CSI300",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(list(positions), years)
    axis.set_title("Annual Return Comparison")
    axis.set_ylabel("Annualized return")
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "annual_returns.png", dpi=180)
    plt.close(figure)


def main() -> None:
    daily, yearly = load_reports()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_nav(daily)
    plot_drawdown(daily)
    plot_yearly_returns(yearly)
    print("主策略图表已生成：")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
