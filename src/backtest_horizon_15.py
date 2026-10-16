"""Backtest the 15-day model with 5-, 10-, and 15-day schedules."""

from src.backtest_horizon_10 import run_horizon_backtest


if __name__ == "__main__":
    run_horizon_backtest(
        horizon=15,
        rebalance_periods=(5, 10, 15),
    )
