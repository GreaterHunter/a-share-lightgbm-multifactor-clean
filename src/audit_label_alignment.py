"""Compare the old close-exit label with the strategy's open-to-open holding period.

Research-only diagnostic: this never refits or deploys the paper-trading model.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from paper_futu.cn_common import HISTORICAL_PROVIDER
from src.build_ranked_dataset import cross_sectional_rank
from src.config import PROJECT_ROOT, load_research_config


def close_exit_expression(horizon: int) -> str:
    """T close signal, buy T+1 open, value at T+H close (old target)."""
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    return f"Ref($close, -{horizon}) / Ref($open, -1) - 1"


def open_exit_expression(horizon: int) -> str:
    """T close signal, buy T+1 open, exit at T+H+1 open."""
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    return f"Ref($open, -{horizon + 1}) / Ref($open, -1) - 1"


def latest_mature_signal(calendar: pd.DatetimeIndex, as_of: pd.Timestamp,
                         horizon: int, *, open_exit: bool) -> pd.Timestamp:
    """An unknown future price is never a training label, even after scaling."""
    positions = calendar[calendar <= as_of]
    required_forward_days = horizon + int(open_exit)
    if len(positions) <= required_forward_days:
        raise ValueError("Not enough observed trading days for label maturity")
    return positions[-required_forward_days - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("Horizon must be positive")
    qlib.init(provider_uri=str(HISTORICAL_PROVIDER), region=REG_CN, kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    instruments = D.instruments(load_research_config()["universe"])
    panel = D.features(
        instruments=instruments,
        fields=[close_exit_expression(args.horizon), open_exit_expression(args.horizon)],
        start_time=args.start, end_time=args.end, freq="day",
    )
    panel.columns = ["old_close_exit", "aligned_open_exit"]
    pair = panel.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        raise ValueError("No pairwise observed labels for this audit period")
    old_rank = cross_sectional_rank(pair, ["old_close_exit"])["old_close_exit"]
    new_rank = cross_sectional_rank(pair, ["aligned_open_exit"])["aligned_open_exit"]
    scaled_rank = cross_sectional_rank(pair.assign(scaled=pair.old_close_exit / args.horizon),
                                       ["scaled"])["scaled"]
    stats = {
        "start": args.start,
        "end": args.end,
        "horizon": args.horizon,
        "paired_stock_days": len(pair),
        "old_label": close_exit_expression(args.horizon),
        "aligned_label": open_exit_expression(args.horizon),
        "pearson_old_vs_aligned": float(pair.old_close_exit.corr(pair.aligned_open_exit)),
        "mean_absolute_label_difference": float((pair.old_close_exit - pair.aligned_open_exit).abs().mean()),
        "mean_absolute_rank_difference": float((old_rank - new_rank).abs().mean()),
        "max_rank_difference_after_dividing_old_label_by_horizon": float((old_rank - scaled_rank).abs().max()),
        "provider_last_observed_date": str(calendar.max().date()),
        "latest_mature_old_signal_date": str(latest_mature_signal(calendar, calendar.max(), args.horizon,
                                                                   open_exit=False).date()),
        "latest_mature_aligned_signal_date": str(latest_mature_signal(calendar, calendar.max(), args.horizon,
                                                                       open_exit=True).date()),
    }
    output = PROJECT_ROOT / "reports" / "paper_futu" / f"label_alignment_{args.horizon}d.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
