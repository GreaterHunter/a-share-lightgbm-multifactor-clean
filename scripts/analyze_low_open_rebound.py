"""Read-only historical low-open/rebound diagnostics; never place orders."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "data/qlib/qlib_bin"
OUT = ROOT / "reports/diagnostics/low_open_rebound_20261014"
START, END = "2022-01-01", "2025-12-31"
FIELDS = ["$open", "$close", "Ref($close, 1)", "$volume", "$factor", "Ref($factor, 1)"]
EPS = 1e-6


def counts(frame: pd.DataFrame, raw: bool = False, rounded: bool = False,
           low_threshold: float = EPS) -> pd.DataFrame:
    op, cl, pc = (frame[x].astype(float) for x in ["open", "close", "prev_close"])
    if raw:
        op, cl, pc = op / frame.factor, cl / frame.factor, pc / frame.prev_factor
    if rounded:
        op, cl, pc = op.round(2), cl.round(2), pc.round(2)
    gap, intraday = op / pc - 1, cl / op - 1
    low, up = gap.lt(-low_threshold), intraday.gt(EPS)
    recover = (cl / pc - 1).ge(-EPS)
    values = pd.DataFrame({
        "observations": 1,
        "low_open": low.astype(int),
        "low_open_up_close": (low & up).astype(int),
        "low_open_recover_previous_close": (low & recover).astype(int),
        "up_close": up.astype(int),
        "sum_gap": gap,
        "sum_intraday": intraday,
        "sum_intraday_given_low": intraday.where(low, 0),
        "low_open_1pct": gap.lt(-0.01).astype(int),
        "low_open_1pct_up_close": (gap.lt(-0.01) & up).astype(int),
        "sum_intraday_given_low_1pct": intraday.where(gap.lt(-0.01), 0),
    }, index=frame.index)
    return values.groupby(level="datetime").sum()


def summarize(daily: pd.DataFrame, scope: str, distinct: dict) -> list[dict]:
    rows = []
    for period, group in [("all", daily), *[(str(y), g) for y, g in daily.groupby(daily.index.year)]]:
        s = group.sum()
        n, low = int(s.observations), int(s.low_open)
        row = {"scope": scope, "period": period, "start": str(group.index.min().date()),
               "end": str(group.index.max().date()), "trading_days": len(group),
               "stock_day_observations": n, "distinct_stocks": len(distinct.get((scope, period), set())),
               "low_open_observations": low, "low_up_observations": int(s.low_open_up_close),
               "low_open_share": low / n, "low_up_share_all": s.low_open_up_close / n,
               "up_close_given_low_open": s.low_open_up_close / low,
               "recover_previous_close_given_low_open": s.low_open_recover_previous_close / low,
               "up_close_unconditional": s.up_close / n,
               "mean_gap_return": s.sum_gap / n, "mean_intraday_return": s.sum_intraday / n,
               "mean_intraday_return_given_low_open": s.sum_intraday_given_low / low,
               "day_equal_weight_up_given_low": (group.low_open_up_close / group.low_open.replace(0, np.nan)).mean(),
               "up_given_low_at_least_1pct": s.low_open_1pct_up_close / s.low_open_1pct,
               "mean_intraday_given_low_at_least_1pct": s.sum_intraday_given_low_1pct / s.low_open_1pct}
        rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1)
    parts, distinct, coverage = {}, {}, []

    def record(scope, frame, raw=False, rounded=False, low_threshold=EPS):
        if frame.empty:
            return
        parts.setdefault(scope, []).append(counts(frame, raw, rounded, low_threshold))
        for period, group in [("all", frame), *[(str(y), g) for y, g in frame.groupby(frame.index.get_level_values("datetime").year)]]:
            distinct.setdefault((scope, period), set()).update(group.index.get_level_values("instrument"))

    def clean(frame):
        frame = frame.copy()
        frame.columns = ["open", "close", "prev_close", "volume", "factor", "prev_factor"]
        mask = np.isfinite(frame).all(axis=1) & frame.gt(0).all(axis=1)
        return frame.loc[mask]

    pattern = re.compile(r"(?:SH(?:600|601|603|605|688)\d{3}|SZ(?:000|001|002|003|300|301|302)\d{3}|BJ\d{6})$")
    codes = sorted(p.name.upper() for p in (PROVIDER / "features").iterdir()
                   if p.is_dir() and pattern.fullmatch(p.name.upper()))
    for offset in range(0, len(codes), 200):
        batch = codes[offset:offset+200]
        quoted = D.features(batch, FIELDS, start_time=START, end_time=END, freq="day")
        valid = clean(quoted)
        code_index = valid.index.get_level_values("instrument")
        for scope, frame in [("all_a_available", valid),
                             ("sh_sz_a", valid.loc[~code_index.str.startswith("BJ")]),
                             ("beijing_a", valid.loc[code_index.str.startswith("BJ")])]:
            record(scope, frame)
        mainland = valid.loc[~code_index.str.startswith("BJ")]
        record("sh_sz_a_raw_prices", mainland, raw=True)
        record("sh_sz_a_raw_rounded", mainland, raw=True, rounded=True)
        record("sh_sz_a_adj_gap_below_10bp", mainland, low_threshold=0.001)
        unchanged = np.isclose(mainland.factor, mainland.prev_factor, rtol=1e-5, atol=0)
        record("sh_sz_a_no_factor_change", mainland.loc[unchanged])
        coverage.append({"batch_first": batch[0], "codes": len(batch),
                         "queried_rows": len(quoted), "valid_rows": len(valid),
                         "gap_over_50pct": int((valid.open / valid.prev_close - 1).abs().gt(0.5).sum()),
                         "intraday_over_50pct": int((valid.close / valid.open - 1).abs().gt(0.5).sum())})
        print(f"Processed {min(offset+200, len(codes))}/{len(codes)} stocks", flush=True)
        del quoted, valid, mainland
        gc.collect()

    for scope, universe in [("historical_csi300", D.instruments("csi300")), ("csi300_index", ["SH000300"])]:
        frame = clean(D.features(universe, FIELDS, start_time=START, end_time=END, freq="day"))
        record(scope, frame)
        record(scope + "_raw_rounded", frame, raw=True, rounded=True)
        record(scope + "_adj_gap_below_10bp", frame, low_threshold=0.001)

    rows = []
    for scope, chunks in parts.items():
        daily = pd.concat(chunks).groupby(level=0).sum().sort_index()
        assert (daily.low_open_up_close <= daily.low_open).all()
        assert (daily.low_open <= daily.observations).all()
        assert (daily.low_open_recover_previous_close <= daily.low_open_up_close).all()
        rows.extend(summarize(daily, scope, distinct))
        daily.to_csv(OUT / f"{scope}_daily_counts.csv", encoding="utf-8-sig")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps({
        "provider": str(PROVIDER), "start": START, "end": END,
        "universe": "available historical equity files, excluding funds and indices; CSI300 index is separate",
        "low_open": "adjusted open / previous-session adjusted close - 1 < -1e-6",
        "up_close": "close / open - 1 > 1e-6 (not a monotonic intraday path)",
        "recover": "close / previous-session adjusted close - 1 >= -1e-6",
        "denominator": "valid stock-session observations; conditional and unconditional ratios separately reported",
        "filter": "finite positive open, close, previous close, volume, factor, previous factor",
        "available_instrument_files": len(codes), "batch_coverage": coverage,
        "limitations": "local third-party data; daily OHLC cannot show intraday paths; individual-stock outcomes are correlated; BJ codes may include code migrations and are excluded from headline SH/SZ statistics",
        "rounding_audit": "Factor varies slightly on ordinary days, including indices. Adjusted microscopic gaps can misclassify flat opening. raw_rounded uses reconstructed unadjusted prices rounded to CNY0.01; this is a literal previous-close comparison, not exchange ex-right reference price. adj_gap_below_10bp is a sensitivity check. no_factor_change is NOT a valid corporate-action filter and should not be used for conclusions.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = result.loc[result.scope.isin(["sh_sz_a", "historical_csi300", "csi300_index", "beijing_a"]) ]
    print(selected[["scope", "period", "stock_day_observations", "low_open_share", "up_close_given_low_open",
                    "recover_previous_close_given_low_open", "mean_gap_return", "mean_intraday_return",
                    "mean_intraday_return_given_low_open"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
