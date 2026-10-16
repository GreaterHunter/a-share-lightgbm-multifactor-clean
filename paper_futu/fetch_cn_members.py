"""Save current CSI300 constituents from the index publisher, not a guessed list."""

import argparse
from datetime import datetime, timezone
import io
import json

import pandas as pd
import requests

from paper_futu.cn_common import STATE_DIR, to_futu_code

URL = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000300cons.xls"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    directory = STATE_DIR / args.date
    directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=45)
    response.raise_for_status()
    data = pd.read_excel(io.BytesIO(response.content))
    if data.shape[1] != 9 or len(data) != 300:
        raise ValueError("CSI official constituent file has an unexpected schema")
    asof = pd.to_datetime(data.iloc[:, 0].astype(str), format="%Y%m%d")
    if asof.max() > pd.Timestamp(args.date) or (pd.Timestamp(args.date) - asof.min()).days > 7:
        raise ValueError("Constituent file is future-dated or stale")
    symbols = data.iloc[:, 4].astype(str).str.zfill(6)
    instruments = symbols.map(lambda code: ("SH" if code.startswith("6") else "SZ") + code)
    members = pd.DataFrame({"instrument": instruments, "code": instruments.map(to_futu_code),
                            "name": data.iloc[:, 5], "publisher_asof": asof.dt.strftime("%Y-%m-%d")})
    if members.instrument.nunique() != 300:
        raise ValueError("Constituent symbols are duplicated")
    members.to_csv(directory / "members.csv", index=False, encoding="utf-8-sig")
    (directory / "members_source.xls").write_bytes(response.content)
    (directory / "members_source.json").write_text(json.dumps({
        "url": URL, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "publisher_asof": str(asof.max().date()), "members": 300,
    }, indent=2), encoding="utf-8")
    print(f"Current CSI300 universe: {len(members)}; publisher asof {asof.max().date()}", flush=True)


if __name__ == "__main__":
    main()
