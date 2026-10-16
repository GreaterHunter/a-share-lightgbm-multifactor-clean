"""Build research features from yesterday's completed bars and today's auction open.

The quote input is deliberately narrowed to the few fields known after the
opening auction. This module neither contacts a broker nor places an order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from src.alpha158_morning import alpha158_registry, feature_names
from src.build_enhanced_dataset import (
    merge_latest_event,
    prepare_annual_eps_events,
    prepare_general_report_events,
    validate_point_in_time,
)
from src.build_ranked_dataset import cross_sectional_rank


def validated_opening_quotes(
    quotes: pd.DataFrame,
    observation_date: pd.Timestamp,
    observed_by: pd.Timestamp,
) -> pd.DataFrame:
    """Accept only a same-session snapshot timestamped 09:25 or later.

    `observed_by` is the actual decision deadline. An end-of-day saved quote is
    suitable for an explicitly labelled historical replay, not for a 09:31
    live decision.
    """
    columns = ["code", "update_time", "open_price", "prev_close_price", "suspension"]
    missing = set(columns) - set(quotes.columns)
    if missing:
        raise ValueError(f"Missing opening quote fields: {sorted(missing)}")
    frame = quotes[columns].copy()
    if frame.code.duplicated().any():
        raise ValueError("Duplicate opening quote codes")
    stamp = pd.to_datetime(frame.update_time, errors="coerce")
    start = pd.Timestamp(observation_date).normalize() + pd.Timedelta(hours=9, minutes=25)
    deadline = pd.Timestamp(observed_by)
    if deadline.normalize() != start.normalize() or deadline < start:
        raise ValueError("Decision deadline must be on the observation session after 09:25")
    if stamp.isna().any() or stamp.lt(start).any() or stamp.gt(deadline).any():
        raise ValueError("Opening quote is stale, future-dated, or arrived after the decision deadline")
    frame["update_time"] = stamp
    return frame.set_index("code")


def build_morning_observation(
    provider_uri: str,
    previous_session: pd.Timestamp,
    observation_date: pd.Timestamp,
    members: list[str],
    quotes: pd.DataFrame,
    observed_by: pd.Timestamp,
    financial_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw and ranked 162-feature stock-by-observation-date panels."""
    previous_session = pd.Timestamp(previous_session).normalize()
    observation_date = pd.Timestamp(observation_date).normalize()
    if previous_session >= observation_date or not members or len(set(members)) != len(members):
        raise ValueError("Previous session and unique stock membership are required")
    opening = validated_opening_quotes(quotes, observation_date, observed_by)
    expected_codes = {code[:2] + "." + code[2:] for code in members}
    if not expected_codes.issubset(opening.index):
        raise ValueError("Opening quotes do not cover the requested stocks")

    qlib.init(provider_uri=str(provider_uri), region=REG_CN, kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    if previous_session not in calendar:
        raise ValueError("Previous session is absent from the Qlib calendar")
    registry = alpha158_registry()
    yesterday = D.features(
        members,
        list(registry.values()) + ["$close / ($factor + 1e-12)"],
        start_time=str(previous_session.date()),
        end_time=str(previous_session.date()),
        freq="day",
    )
    yesterday.columns = list(registry) + ["previous_raw_close"]
    yesterday = yesterday.droplevel("datetime").reindex(members)
    yesterday.index.name = "instrument"
    panel = yesterday.reset_index()
    panel["code"] = panel.instrument.str[:2] + "." + panel.instrument.str[2:]
    known = opening[["open_price", "prev_close_price", "suspension"]].copy()
    panel = panel.join(known, on="code", validate="one_to_one")
    tradable = (
        ~panel.suspension.astype(bool)
        & np.isfinite(panel.open_price)
        & np.isfinite(panel.prev_close_price)
        & panel.open_price.gt(0)
        & panel.prev_close_price.gt(0)
    )
    panel = panel.loc[tradable].copy()
    panel["morning_open_gap"] = panel.open_price / panel.prev_close_price - 1
    panel["datetime"] = pd.Series(observation_date, index=panel.index).astype("datetime64[ns]")
    panel = merge_latest_event(panel, prepare_general_report_events(financial_events), "fundamental_available_date")
    panel = merge_latest_event(panel, prepare_annual_eps_events(financial_events), "annual_eps_available_date")
    validate_point_in_time(panel)
    price = panel.previous_raw_close.where(panel.previous_raw_close.gt(0))
    panel["book_to_price"] = panel.book_value_per_share / price
    panel["earnings_to_price"] = panel.annual_eps / price
    raw = panel.set_index(["instrument", "datetime"])[feature_names()].replace([np.inf, -np.inf], np.nan)
    if raw.index.has_duplicates or raw.isna().all(axis=1).any():
        raise ValueError("Invalid morning observation")
    ranked = cross_sectional_rank(raw, feature_names()).astype("float32")
    return raw, ranked
