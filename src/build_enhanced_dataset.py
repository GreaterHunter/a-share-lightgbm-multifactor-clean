"""Build the 12 price-volume + 3 point-in-time fundamental dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from src.config import PROJECT_ROOT, load_research_config
from src.factor_library import FUNDAMENTAL_FEATURE_NAMES


BASE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multifactor_dataset.parquet"
)
FUNDAMENTAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "fundamentals.parquet"
)
DISCLOSURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "fundamental_disclosure_dates.parquet"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "enhanced_multifactor_dataset.parquet"
)
QUALITY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data"
    / "enhanced_factor_quality.csv"
)


def load_raw_close(config: dict) -> pd.DataFrame:
    """Load actual historical close rather than the adjusted Qlib close."""

    provider_uri = PROJECT_ROOT / config["provider_uri"]
    qlib.init(
        provider_uri=str(provider_uri),
        region=REG_CN,
        kernels=1,
    )

    instruments = D.instruments(config["universe"])
    raw_close = D.features(
        instruments=instruments,
        fields=["$close / ($factor + 1e-12)"],
        start_time=config["train_start"],
        end_time=config["test_end"],
        freq="day",
    )
    raw_close.columns = ["raw_close"]

    raw_close = raw_close.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return raw_close.sort_index()


def load_disclosed_fundamentals() -> pd.DataFrame:
    """Join financial values to their official actual disclosure dates."""

    required_paths = [
        FUNDAMENTAL_PATH,
        DISCLOSURE_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required data: {path}"
            )

    fundamentals = pd.read_parquet(
        FUNDAMENTAL_PATH,
        columns=[
            "instrument",
            "report_date",
            "book_value_per_share",
            "basic_eps",
            "operating_cashflow_to_revenue",
        ],
    )
    disclosures = pd.read_parquet(
        DISCLOSURE_PATH,
        columns=[
            "instrument",
            "report_date",
            "disclosure_date",
        ],
    )

    for frame in [fundamentals, disclosures]:
        frame["report_date"] = pd.to_datetime(
            frame["report_date"],
            errors="coerce",
        )

    disclosures["disclosure_date"] = pd.to_datetime(
        disclosures["disclosure_date"],
        errors="coerce",
    )

    if fundamentals.duplicated(
        ["instrument", "report_date"]
    ).any():
        raise ValueError(
            "Fundamental data has duplicate report rows"
        )

    if disclosures.duplicated(
        ["instrument", "report_date"]
    ).any():
        raise ValueError(
            "Disclosure data has duplicate report rows"
        )

    financials = fundamentals.merge(
        disclosures,
        on=["instrument", "report_date"],
        how="inner",
        validate="one_to_one",
    )
    financials = financials.dropna(
        subset=["disclosure_date"]
    )

    # Use the report no earlier than the calendar day after disclosure.
    # The source exposes a reliable date but not a consistently precise time.
    financials["available_date"] = (
        financials["disclosure_date"]
        + pd.Timedelta(days=1)
    )

    invalid_time = (
        financials["available_date"]
        <= financials["report_date"]
    )
    if invalid_time.any():
        raise ValueError(
            "Found financial data available before its report period ended"
        )

    return financials


def prepare_general_report_events(
    financials: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare report events for book value and cash-flow quality."""

    events = financials[
        [
            "instrument",
            "report_date",
            "available_date",
            "book_value_per_share",
            "operating_cashflow_to_revenue",
        ]
    ].copy()

    events = events.sort_values(
        ["instrument", "available_date", "report_date"],
        kind="stable",
    )

    # If two reports become available on the same day, keep the later report.
    events = events.drop_duplicates(
        ["instrument", "available_date"],
        keep="last",
    )

    return events.rename(
        columns={
            "report_date": "fundamental_report_date",
            "available_date": "fundamental_available_date",
        }
    )


def prepare_annual_eps_events(
    financials: pd.DataFrame,
) -> pd.DataFrame:
    """Use only disclosed annual EPS, avoiding cumulative interim EPS."""

    annual = financials.loc[
        financials["report_date"].dt.month.eq(12),
        [
            "instrument",
            "report_date",
            "available_date",
            "basic_eps",
        ],
    ].copy()

    annual = annual.rename(
        columns={
            "report_date": "annual_eps_report_date",
            "available_date": "annual_eps_available_date",
            "basic_eps": "annual_eps",
        }
    )

    return annual


def merge_latest_event(
    daily_panel: pd.DataFrame,
    events: pd.DataFrame,
    event_date_column: str,
) -> pd.DataFrame:
    """For each stock-date, attach the latest event already available then."""

    left = daily_panel.sort_values(
        ["datetime", "instrument"],
        kind="stable",
    )
    right = events.sort_values(
        [event_date_column, "instrument"],
        kind="stable",
    )

    return pd.merge_asof(
        left,
        right,
        left_on="datetime",
        right_on=event_date_column,
        by="instrument",
        direction="backward",
        allow_exact_matches=True,
    )


def validate_point_in_time(panel: pd.DataFrame) -> None:
    """Reject any row that received a financial value before it was available."""

    checks = [
        "fundamental_available_date",
        "annual_eps_available_date",
    ]

    for column in checks:
        invalid = (
            panel[column].notna()
            & (panel[column] > panel["datetime"])
        )
        if invalid.any():
            raise ValueError(
                f"Point-in-time violation in {column}: "
                f"{int(invalid.sum())} rows"
            )


def build_quality_report(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Report factor coverage and a compact distribution audit."""

    rows = []
    for factor in FUNDAMENTAL_FEATURE_NAMES:
        values = panel[factor]
        rows.append(
            {
                "factor": factor,
                "non_missing_rows": int(values.notna().sum()),
                "coverage_ratio": float(values.notna().mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not BASE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing base dataset: {BASE_DATASET_PATH}"
        )

    config = load_research_config()
    base_dataset = pd.read_parquet(
        BASE_DATASET_PATH
    )
    raw_close = load_raw_close(config)

    panel = base_dataset.join(
        raw_close,
        how="left",
        validate="one_to_one",
    ).reset_index()

    financials = load_disclosed_fundamentals()
    general_events = prepare_general_report_events(
        financials
    )
    annual_eps_events = prepare_annual_eps_events(
        financials
    )

    panel = merge_latest_event(
        daily_panel=panel,
        events=general_events,
        event_date_column="fundamental_available_date",
    )
    panel = merge_latest_event(
        daily_panel=panel,
        events=annual_eps_events,
        event_date_column="annual_eps_available_date",
    )

    validate_point_in_time(panel)

    valid_price = panel["raw_close"].where(
        panel["raw_close"] > 0
    )
    panel["book_to_price"] = (
        panel["book_value_per_share"]
        / valid_price
    )
    panel["earnings_to_price"] = (
        panel["annual_eps"]
        / valid_price
    )

    panel[FUNDAMENTAL_FEATURE_NAMES] = panel[
        FUNDAMENTAL_FEATURE_NAMES
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    panel = panel.set_index(
        ["instrument", "datetime"]
    ).sort_index()

    quality = build_quality_report(
        panel.reset_index()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    QUALITY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_parquet(OUTPUT_PATH)
    quality.to_csv(
        QUALITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Enhanced dataset built")
    print(f"Rows: {len(panel):,}")
    print(f"Columns: {len(panel.columns):,}")
    print("\nFundamental factor quality:")
    print(quality.to_string(index=False))
    print("\nSaved:")
    print(OUTPUT_PATH)
    print(QUALITY_PATH)


if __name__ == "__main__":
    main()
