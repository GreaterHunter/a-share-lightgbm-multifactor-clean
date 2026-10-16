"""Download and audit historical financial-report disclosure dates from CNINFO.

The CNINFO reservation-disclosure endpoint returns market-wide schedules for a
single report period.  Raw snapshots are cached period by period so a failed
run can resume without requesting completed periods again.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import akshare as ak
import pandas as pd

from src.config import PROJECT_ROOT


START_YEAR = 2008
END_YEAR = 2024

PERIOD_ENDINGS = {
    "一季": "03-31",
    "半年报": "06-30",
    "三季": "09-30",
    "年报": "12-31",
}

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "disclosure_dates"
)
FUNDAMENTAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "fundamentals.parquet"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "fundamental_disclosure_dates.parquet"
)
QUALITY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data"
    / "disclosure_date_quality.csv"
)


STANDARD_COLUMNS = [
    "symbol",
    "stock_name",
    "first_scheduled_date",
    "schedule_change_1",
    "schedule_change_2",
    "schedule_change_3",
    "disclosure_date",
]

DATE_COLUMNS = [
    "first_scheduled_date",
    "schedule_change_1",
    "schedule_change_2",
    "schedule_change_3",
    "disclosure_date",
]


def download_one_period(
    year: int,
    period_name: str,
    retries: int = 3,
) -> pd.DataFrame:
    """Download one market-wide report period, using the local cache if present."""

    cache_path = RAW_DIR / f"{year}_{period_name}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    query_period = f"{year}{period_name}"

    for attempt in range(1, retries + 1):
        try:
            frame = ak.stock_report_disclosure(
                market="沪深京",
                period=query_period,
            )

            if frame.empty:
                raise ValueError(
                    f"CNINFO returned no rows for {query_period}"
                )

            if len(frame.columns) != len(STANDARD_COLUMNS):
                raise ValueError(
                    f"Unexpected columns for {query_period}: "
                    f"{frame.columns.tolist()}"
                )

            frame = frame.copy()
            frame.columns = STANDARD_COLUMNS

            frame["symbol"] = (
                frame["symbol"]
                .astype("string")
                .str.extract(r"(\d{6})", expand=False)
            )

            for column in DATE_COLUMNS:
                frame[column] = pd.to_datetime(
                    frame[column],
                    errors="coerce",
                )

            frame["report_date"] = pd.Timestamp(
                f"{year}-{PERIOD_ENDINGS[period_name]}"
            )
            frame["query_period"] = query_period
            frame["source"] = "CNINFO reservation disclosure"
            frame["source_url"] = (
                "https://www.cninfo.com.cn/new/commonUrl?"
                "url=data/yuyuepilu"
            )
            frame["retrieved_at_utc"] = pd.Timestamp(
                datetime.now(timezone.utc)
            ).tz_localize(None)

            frame = frame.dropna(subset=["symbol"])
            frame = frame.drop_duplicates(
                subset=["symbol", "report_date"],
                keep="last",
            )

            frame.to_parquet(
                cache_path,
                index=False,
            )

            return frame

        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)

    raise RuntimeError("Unexpected retry state")


def build_instrument_lookup(
    fundamentals: pd.DataFrame,
) -> dict[str, str]:
    """Map six-digit symbols to the project's exchange-prefixed identifiers."""

    instruments = (
        fundamentals["instrument"]
        .dropna()
        .astype("string")
        .drop_duplicates()
    )

    symbol_to_instrument = {
        instrument[-6:]: instrument
        for instrument in instruments
    }

    if len(symbol_to_instrument) != len(instruments):
        raise ValueError(
            "Instrument-to-symbol mapping is not one-to-one"
        )

    return symbol_to_instrument


def build_quality_report(
    fundamentals: pd.DataFrame,
    disclosures: pd.DataFrame,
) -> pd.DataFrame:
    """Measure disclosure-date coverage against the existing financial rows."""

    financial_keys = fundamentals[
        ["instrument", "report_date"]
    ].copy()
    financial_keys["report_date"] = pd.to_datetime(
        financial_keys["report_date"],
        errors="coerce",
    )

    financial_keys = financial_keys.loc[
        financial_keys["report_date"].dt.year.between(
            START_YEAR,
            END_YEAR,
        )
    ].drop_duplicates()

    matched = financial_keys.merge(
        disclosures[
            ["instrument", "report_date", "disclosure_date"]
        ],
        on=["instrument", "report_date"],
        how="left",
        validate="one_to_one",
    )

    matched["year"] = matched["report_date"].dt.year
    matched["has_disclosure_date"] = (
        matched["disclosure_date"].notna()
    )

    quality = (
        matched.groupby("year", sort=True)
        .agg(
            financial_rows=("instrument", "size"),
            matched_disclosure_rows=(
                "has_disclosure_date",
                "sum",
            ),
        )
        .reset_index()
    )

    quality["coverage_ratio"] = (
        quality["matched_disclosure_rows"]
        / quality["financial_rows"]
    )

    total_row = pd.DataFrame(
        {
            "year": ["TOTAL"],
            "financial_rows": [
                quality["financial_rows"].sum()
            ],
            "matched_disclosure_rows": [
                quality["matched_disclosure_rows"].sum()
            ],
        }
    )
    total_row["coverage_ratio"] = (
        total_row["matched_disclosure_rows"]
        / total_row["financial_rows"]
    )

    return pd.concat(
        [quality, total_row],
        ignore_index=True,
    )


def main() -> None:
    if not FUNDAMENTAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing fundamentals file: {FUNDAMENTAL_PATH}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)

    fundamentals = pd.read_parquet(
        FUNDAMENTAL_PATH,
        columns=["instrument", "report_date"],
    )
    symbol_to_instrument = build_instrument_lookup(
        fundamentals
    )

    period_frames = []
    total_periods = (
        (END_YEAR - START_YEAR + 1)
        * len(PERIOD_ENDINGS)
    )
    completed = 0

    for year in range(START_YEAR, END_YEAR + 1):
        for period_name in PERIOD_ENDINGS:
            frame = download_one_period(
                year=year,
                period_name=period_name,
            )
            period_frames.append(frame)
            completed += 1
            print(
                f"Downloaded {completed}/{total_periods}: "
                f"{year}{period_name}, rows={len(frame):,}",
                flush=True,
            )
            time.sleep(0.2)

    disclosures = pd.concat(
        period_frames,
        ignore_index=True,
    )
    disclosures["instrument"] = disclosures[
        "symbol"
    ].map(symbol_to_instrument)

    disclosures = disclosures.dropna(
        subset=["instrument"]
    )
    disclosures = disclosures.sort_values(
        ["instrument", "report_date"],
        kind="stable",
    )
    disclosures = disclosures.drop_duplicates(
        subset=["instrument", "report_date"],
        keep="last",
    )

    if disclosures.duplicated(
        ["instrument", "report_date"]
    ).any():
        raise ValueError(
            "Duplicate instrument/report_date rows remain"
        )

    disclosures.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    quality = build_quality_report(
        fundamentals=fundamentals,
        disclosures=disclosures,
    )
    quality.to_csv(
        QUALITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDisclosure-date download complete")
    print(f"Rows: {len(disclosures):,}")
    print(
        "Instruments: "
        f"{disclosures['instrument'].nunique():,}"
    )
    print("\nCoverage report:")
    print(quality.to_string(index=False))
    print("\nSaved:")
    print(OUTPUT_PATH)
    print(QUALITY_PATH)


if __name__ == "__main__":
    main()
