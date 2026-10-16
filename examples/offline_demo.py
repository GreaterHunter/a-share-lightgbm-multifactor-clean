"""Synthetic data only: demonstrate the shared open-to-open label contract."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.forward_labels import label_end_dates, select_mature_period

FEATURES = ["return_1d", "mom_5d", "volatility_20d"]


def make_panel(seed: int = 42) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Business days here are synthetic sessions, not the Chinese exchange calendar."""
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2020-01-01", periods=180)
    parts = []
    for number in range(32):
        close = 20 * np.exp(np.cumsum(rng.normal(0, .015, len(calendar))))
        previous = np.r_[20., close[:-1]]
        opened = previous * np.exp(rng.normal(0, .005, len(calendar)))
        frame = pd.DataFrame({"open": opened, "close": close}, index=calendar)
        frame["return_1d"] = frame.close.pct_change(fill_method=None)
        frame["mom_5d"] = frame.close.pct_change(5, fill_method=None)
        frame["volatility_20d"] = frame.return_1d.rolling(20, min_periods=20).std()
        frame["raw_label"] = frame.open.shift(-11) / frame.open.shift(-1) - 1
        frame["instrument"] = f"SYNTH{number:03d}"
        frame.index.name = "datetime"
        parts.append(frame.reset_index().set_index(["instrument", "datetime"]))
    panel = pd.concat(parts).sort_index()
    panel["label_end_date"] = label_end_dates(panel.index, calendar, 10)
    panel["target"] = panel.raw_label.groupby(level="datetime").rank(
        method="average", pct=True
    ) - .5
    return panel, calendar


def run_demo() -> dict:
    panel, calendar = make_panel()
    cutoff = calendar[119]
    train = select_mature_period(panel, str(calendar[0]), str(cutoff)).dropna(
        subset=FEATURES + ["target"]
    )
    test = select_mature_period(panel, str(calendar[120]), str(calendar[-1])).dropna(
        subset=FEATURES + ["target"]
    )
    # Fixed educational parameters; no tuning on the displayed evaluation period.
    model = lgb.LGBMRegressor(
        n_estimators=20, num_leaves=7, min_child_samples=30,
        random_state=42, n_jobs=1, verbosity=-1,
    )
    model.fit(train[FEATURES], train.target)
    scored = test[["raw_label"]].assign(prediction=model.predict(test[FEATURES]))
    daily = [part.prediction.corr(part.raw_label, method="spearman")
             for _, part in scored.groupby(level="datetime")
             if part.prediction.nunique() >= 2 and part.raw_label.nunique() >= 2]
    values = np.asarray(daily, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "data": "SYNTHETIC ONLY; not strategy performance or real exchange sessions",
        "label": "Open(T+11)/Open(T+1)-1",
        "train_cutoff": cutoff.date().isoformat(),
        "max_train_label_end": train.label_end_date.max().date().isoformat(),
        "train_rows": len(train), "evaluation_rows": len(test),
        "valid_ic_days": len(finite),
        "mean_rank_ic": float(finite.mean()) if len(finite) else None,
        "trading": "none; no network, account access or file output",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
