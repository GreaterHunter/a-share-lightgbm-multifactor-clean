"""Compare strict Top10 and limited-replacement Top10 on frozen predictions."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN

from paper_futu.backtest_recent_window import DATE, PROVIDER, REPORT
from src.analyze_portfolio_robustness import run_tiered_backtest
from src.compare_models import calculate_max_drawdown, enrich_daily_report
from src.config import load_main_strategy_config, load_research_config

OUTPUT = REPORT / "top10_comparison"


def summarize(report, name, n_drop):
    window = report.tail(30).copy()
    if len(window) != 30 or str(window.index.min().date()) != "2026-08-24" or str(window.index.max().date()) != DATE:
        raise ValueError("Top10 comparison window differs from Top50")
    wealth = float((1 + window.net_return).prod())
    benchmark = float((1 + window.bench).prod())
    before = report.loc[report.index < window.index.min()].iloc[-1]
    if not np.isclose(window.account.iloc[-1] / before.account, wealth, atol=1e-10):
        raise ValueError("Top10 account NAV does not match compounded net return")
    return {"variant": name, "top_k": 10, "n_drop": n_drop,
            "start": str(window.index.min().date()), "end": str(window.index.max().date()),
            "trading_days": 30, "strategy_net_return": wealth - 1,
            "benchmark_return": benchmark - 1, "geometric_excess_return": wealth / benchmark - 1,
            "arithmetic_excess_percentage_points": (wealth - benchmark) * 100,
            "max_drawdown": calculate_max_drawdown(window.net_return),
            "positive_days": int(window.net_return.gt(0).sum()),
            "cost_rate_sum": float(window.cost.sum()),
            "mean_daily_turnover": float(window.turnover.mean()),
            "equal_weight_each_stock_target": 0.095,
            "continuous_account_since": str(report.index.min().date())}


def main():
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    predictions = pd.read_parquet(REPORT / "predictions.parquet")
    codes = sorted(predictions.index.get_level_values("instrument").unique())
    config = {**load_research_config(), **load_main_strategy_config()}
    records = []
    for name, drop in [("strict_top10", 10), ("top10_max_one_replacement", 1)]:
        options = {**config, "top_k": 10, "n_drop": drop}
        print(f"Backtesting {name}: 10 stocks, n_drop={drop}, equal weighted, 10-day schedule", flush=True)
        raw = run_tiered_backtest(predictions, options, rebalance_every=10,
            top_k=10, top_tier_size=10, top_tier_weight=0.40, period_end=DATE, quote_codes=codes)
        report = enrich_daily_report(raw, name)
        result = summarize(report, name, drop)
        records.append(result)
        report.to_csv(OUTPUT / f"{name}_full_daily.csv", encoding="utf-8-sig")
        report.tail(30).to_csv(OUTPUT / f"{name}_last_30_daily.csv", encoding="utf-8-sig")
        print(json.dumps(result, indent=2), flush=True)
    baseline = json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))
    baseline.update({"variant": "original_tiered_top50", "top_k": 50, "n_drop": 5})
    comparison = pd.DataFrame([baseline, *records])
    comparison.to_csv(OUTPUT / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "comparison.json").write_text(comparison.to_json(orient="records", indent=2), encoding="utf-8")
    print("\nRecent 30-session comparison:")
    print(comparison[["variant", "strategy_net_return", "benchmark_return", "geometric_excess_return", "max_drawdown"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
