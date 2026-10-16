FEATURE_EXPRESSIONS: dict[str, str] = {
    "return_1d": "$close / Ref($close, 1) - 1",

    "mom_5d": "$close / Ref($close, 5) - 1",
    "mom_20d": "$close / Ref($close, 20) - 1",
    "mom_60d": "$close / Ref($close, 60) - 1",

    "volatility_20d": (
        "Std($close / Ref($close, 1) - 1, 20)"
    ),
    "volatility_60d": (
        "Std($close / Ref($close, 1) - 1, 60)"
    ),

    "intraday_return": "$close / $open - 1",
    "overnight_gap": "$open / Ref($close, 1) - 1",

    "amplitude_20d": (
        "Mean(($high - $low) / $close, 20)"
    ),

    "volume_ratio_5_20": (
        "Mean($volume, 5) / Mean($volume, 20) - 1"
    ),

    "price_position_20d": (
        "($close - Min($low, 20)) "
        "/ (Max($high, 20) - Min($low, 20) + 1e-12)"
    ),

    "price_volume_corr_20d": (
        "Corr("
        "$close / Ref($close, 1) - 1, "
        "$volume / Ref($volume, 1) - 1, "
        "20"
        ")"
    ),
}

PRICE_VOLUME_FEATURE_NAMES = list(
    FEATURE_EXPRESSIONS.keys()
)

FUNDAMENTAL_FEATURE_NAMES = [
    "book_to_price",
    "earnings_to_price",
    "operating_cashflow_to_revenue",
]

ENHANCED_FEATURE_NAMES = (
    PRICE_VOLUME_FEATURE_NAMES
    + FUNDAMENTAL_FEATURE_NAMES
)

LABEL_NAME = "future_return_5d"

LABEL_EXPRESSION = (
    "Ref($close, -5) / Ref($open, -1) - 1"
)

RANK_LABEL_NAME = "target_rank_5d"
