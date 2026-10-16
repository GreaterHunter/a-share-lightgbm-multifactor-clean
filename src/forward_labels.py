"""The price/time contract for active next-open strategies (legacy is explicit)."""

from __future__ import annotations

import pandas as pd

OPEN_TO_OPEN = "open_to_open"
LEGACY_OPEN_TO_CLOSE = "legacy_open_to_close"
MORNING_OPEN_PROXY = "observed_auction_open_to_future_open_proxy"


def label_contract(horizon: int, price_basis: str = OPEN_TO_OPEN) -> dict:
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if price_basis == OPEN_TO_OPEN:
        return {
            "label_price_basis": OPEN_TO_OPEN,
            "label_horizon": horizon,
            "label_expression": f"Ref($open, -{horizon + 1}) / Ref($open, -1) - 1",
            "raw_label": f"future_open_return_{horizon}d",
            "target": f"target_open_rank_{horizon}d",
            "label_end_offset": horizon + 1,
            "execution_price_basis": "next_session_open",
        }
    if price_basis == LEGACY_OPEN_TO_CLOSE:
        return {
            "label_price_basis": LEGACY_OPEN_TO_CLOSE,
            "label_horizon": horizon,
            "label_expression": f"Ref($close, -{horizon}) / Ref($open, -1) - 1",
            "raw_label": f"future_return_{horizon}d",
            "target": f"target_rank_{horizon}d",
            "label_end_offset": horizon,
        }
    if price_basis == MORNING_OPEN_PROXY:
        # Index is the observation session S (not the previous close T).
        # The observed auction open is a forecast reference, NOT a fill obtained
        # after observing that price. Deploying this needs post-observation fills.
        return {
            "label_price_basis": MORNING_OPEN_PROXY,
            "label_horizon": horizon,
            "label_expression": f"Ref($open, -{horizon}) / $open - 1",
            "raw_label": f"future_observed_open_return_{horizon}d",
            "target": f"target_morning_rank_{horizon}d",
            "label_end_offset": horizon,
            "signal_availability": "after_confirmed_opening_auction",
            "execution_price_basis": "post_observation_actual_fill",
            "requires_intraday_execution_validation": True,
        }
    raise ValueError(f"Unsupported label price basis: {price_basis}")


def label_end_dates(index: pd.MultiIndex, calendar: pd.DatetimeIndex,
                    horizon: int, price_basis: str = OPEN_TO_OPEN) -> pd.Series:
    """Count exchange sessions, not rows/days; missing future dates remain NaT."""
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Trading calendar must be unique and increasing")
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    positions = calendar.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("A signal date is absent from the trading calendar")
    ends = positions + label_contract(horizon, price_basis)["label_end_offset"]
    values = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]", name="label_end_date")
    available = ends < len(calendar)
    values.iloc[available] = calendar.take(ends[available]).to_numpy()
    return values


def select_mature_period(frame: pd.DataFrame, start: str, cutoff: str) -> pd.DataFrame:
    """Use only outcomes actually known by cutoff; never purge mature rows twice."""
    if "label_end_date" not in frame:
        raise ValueError("Open-to-open training requires actual label_end_date")
    dates = frame.index.get_level_values("datetime")
    end_dates = pd.to_datetime(frame["label_end_date"])
    start, cutoff = pd.Timestamp(start), pd.Timestamp(cutoff)
    result = frame.loc[(dates >= start) & (dates <= cutoff)
                       & end_dates.notna() & end_dates.le(cutoff)].copy()
    if result.empty:
        raise ValueError(f"No mature samples: {start.date()} to {cutoff.date()}")
    if (result["label_end_date"] <= result.index.get_level_values("datetime")).any():
        raise ValueError("Label end must follow its signal date")
    return result


def validate_open_label_metadata(metadata: dict) -> None:
    expected = label_contract(metadata["label_horizon"])
    for key in ["label_price_basis", "label_expression", "target", "execution_price_basis"]:
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Open-to-open model contract mismatch: {key}")
