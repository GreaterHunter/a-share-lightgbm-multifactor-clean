"""Official complete Alpha158, with an explicit morning information boundary."""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pandas as pd
from qlib.contrib.data.loader import Alpha158DL

from src.factor_library import FUNDAMENTAL_FEATURE_NAMES
from src.forward_labels import MORNING_OPEN_PROXY, label_contract, label_end_dates


def alpha158_registry() -> dict[str, str]:
    expressions, names = Alpha158DL.get_feature_config()
    if len(names) != 158 or len(set(names)) != 158:
        raise ValueError("Expected the complete official 158-feature specification")
    return {f"alpha158_{name}": expression for name, expression in zip(names, expressions)}


def morning_registry() -> dict[str, str]:
    # Shifting the WHOLE expression also shifts normalizers such as $close and
    # $volume. Shifting only the numerator would leak today's later information.
    result = {name: f"Ref({expression}, 1)" for name, expression in alpha158_registry().items()}
    result["morning_open_gap"] = "$open / Ref($close, 1) - 1"
    return result


def feature_names(include_open: bool = True) -> list[str]:
    return list(alpha158_registry()) + FUNDAMENTAL_FEATURE_NAMES + (["morning_open_gap"] if include_open else [])


def specification(horizon: int) -> dict:
    registry = morning_registry()
    return {
        **label_contract(horizon, MORNING_OPEN_PROXY),
        "factor_set": "complete_official_Alpha158_plus_3_fundamentals_plus_morning_gap",
        "alpha158_count": 158, "feature_count": 162, "features": feature_names(),
        "expressions": registry,
        "factor_source": "https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py",
        "factor_source_sha256": hashlib.sha256(inspect.getsource(Alpha158DL).encode()).hexdigest(),
        "feature_recipe_sha256": hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest(),
        "forbidden_current_session_inputs": ["close", "high", "low", "daily_volume", "daily_vwap"],
        "fundamental_price_denominator": "previous_session_unadjusted_close",
        "fundamental_available_cutoff": "morning_session_date; disclosed date + 1 calendar day",
    }


def normalize_local_bar(snapshot: dict, adjusted_factor: float) -> dict:
    """Match verified community Qlib units: hands, amount/1000, adjusted prices.

    Futu volume is shares. The legacy provider stores volume in hands / factor
    and amount in thousand CNY, so 10 * amount / volume equals adjusted VWAP.
    The experiment independently verifies these units against overlapping bars.
    """
    if not np.isfinite(adjusted_factor) or adjusted_factor <= 0:
        raise ValueError("Invalid local adjustment factor")
    if bool(snapshot["suspension"]):
        return {**dict.fromkeys(["open", "close", "high", "low", "volume", "amount", "vwap"], np.nan),
                "factor": adjusted_factor}
    volume, turnover = float(snapshot["volume"]), float(snapshot["turnover"])
    if volume <= 0 or turnover <= 0 or not np.isfinite([volume, turnover]).all():
        raise ValueError("Trading bar has invalid volume or turnover")
    result = {name: float(snapshot[field]) * adjusted_factor for name, field in
              [("open", "open_price"), ("close", "last_price"), ("high", "high_price"), ("low", "low_price")]}
    result.update(volume=volume / 100 / adjusted_factor, amount=turnover / 1000,
                  vwap=turnover / volume * adjusted_factor, factor=adjusted_factor)
    if not result["low"] - 1e-5 <= result["vwap"] <= result["high"] + 1e-5:
        raise ValueError("VWAP is outside the completed day's high/low")
    return result


def morning_end_dates(index, calendar, horizon):
    return label_end_dates(index, calendar, horizon, MORNING_OPEN_PROXY)
