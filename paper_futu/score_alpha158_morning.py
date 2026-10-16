"""Read-only Top10 preview after an observed opening auction; never trades."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from paper_futu.cn_common import CHINA_TZ, ROOT
from paper_futu.research_alpha158_morning import SOURCE, digest, paths_for
from src.alpha158_morning import feature_names
from src.morning_observations import build_morning_observation


def fetch_open_quotes(codes: list[str]) -> pd.DataFrame:
    """Futu OpenD quote access only; no trade context or order API is opened."""
    from futu import OpenQuoteContext, RET_OK

    quote = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        parts = []
        for begin in range(0, len(codes), 100):
            status, payload = quote.get_market_snapshot(codes[begin:begin + 100])
            if status != RET_OK:
                raise RuntimeError(f"OpenD opening snapshot failed: {payload}")
            parts.append(payload)
        return pd.concat(parts, ignore_index=True)
    finally:
        quote.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-as-of", required=True, help="Independent data/model cutoff, e.g. 2026-10-13")
    parser.add_argument("--morning-date", required=True)
    parser.add_argument("--members", help="Verified index-publisher constituents CSV")
    parser.add_argument("--quotes-csv", help="Explicit historical replay; no OpenD call")
    parser.add_argument("--replay", action="store_true", help="Allows after-09:31 saved quotes; no execution claim")
    args = parser.parse_args()

    training_as_of = pd.Timestamp(args.training_as_of).normalize()
    observation = pd.Timestamp(args.morning_date).normalize()
    paths = paths_for(training_as_of)
    provider = paths["data"] / "provider"
    metadata_path = paths["model"] / "with_open/metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("162-feature model not trained; run the independent data and train stages first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = metadata_path.with_name("model.txt")
    if digest(model_path) != metadata["model_sha256"] or metadata["features"] != feature_names():
        raise ValueError("Morning model hash or feature order changed")
    if metadata["deployment_status"] != "offline_only_requires_post_observation_execution_validation":
        raise ValueError("Unrecognized research model status")
    ready = json.loads((provider / "morning_snapshot_ready.json").read_text(encoding="utf-8"))
    if ready["as_of"] != str(training_as_of.date()):
        raise ValueError("Provider cutoff does not match the selected model")
    previous = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines()))[-1]
    future = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day_future.txt").read_text().splitlines()))
    if previous != training_as_of or observation != future[future > previous][0]:
        raise ValueError("Morning preview requires the next exchange session after complete historical bars")

    member_path = Path(args.members) if args.members else ROOT / f"data/interim/futu_cn_paper/{observation:%Y-%m-%d}/members.csv"
    members = pd.read_csv(member_path)
    if members.instrument.nunique() != 300 or members.duplicated("instrument").any():
        raise ValueError("Current CSI300 constituent list must contain 300 unique stocks")
    if "publisher_asof" not in members or pd.to_datetime(members.publisher_asof).max() > observation:
        raise ValueError("Constituent effective-date provenance is missing or future-dated")
    if args.quotes_csv:
        if not args.replay:
            raise ValueError("Saved end-of-day snapshots require the explicit --replay flag")
        quotes = pd.read_csv(args.quotes_csv)
        observed_by = observation + pd.Timedelta(hours=23, minutes=59)
    else:
        if args.replay or datetime.now(CHINA_TZ).date() != observation.date():
            raise ValueError("Live opening quotes require today's session date")
        observed_by = pd.Timestamp(datetime.now(CHINA_TZ).replace(tzinfo=None))
        if not observation + pd.Timedelta(hours=9, minutes=25) <= observed_by <= observation + pd.Timedelta(hours=9, minutes=31):
            raise ValueError("Read-only opening preview is available from 09:25 through 09:31")
        quotes = fetch_open_quotes(members.code.tolist())

    raw, ranked = build_morning_observation(
        str(provider), previous, observation, members.instrument.tolist(), quotes,
        observed_by, pd.read_parquet(SOURCE / "financial_events.parquet"),
    )
    if len(ranked) < 250:
        raise ValueError("Too few current stocks have valid observed opening prices")
    model = lgb.Booster(model_file=str(model_path))
    scores = model.predict(ranked[feature_names()], num_threads=4)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite morning model prediction")
    result = pd.DataFrame({"prediction": scores}, index=ranked.index).reset_index()
    result = result.merge(members[["instrument", "name"]], on="instrument", validate="one_to_one")
    result["eligible"] = ~result.name.str.contains("ST", na=False)
    result = result.sort_values(["prediction", "instrument"], ascending=[False, True])
    top = result.loc[result.eligible].head(10)
    if len(top) != 10:
        raise ValueError("Fewer than ten eligible morning candidates")
    paths["report"].mkdir(parents=True, exist_ok=True)
    destination = paths["report"] / f"read_only_morning_preview_{observation:%Y%m%d}.csv"
    result.to_csv(destination, index=False)
    print(top[["instrument", "name", "prediction"]].to_string(index=False))
    print(f"READ ONLY: {destination}; snapshot deadline {observed_by}; no order submitted")


if __name__ == "__main__":
    main()
