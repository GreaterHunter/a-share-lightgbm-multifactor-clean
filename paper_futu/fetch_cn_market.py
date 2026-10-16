"""Read current CSI300 stock quotes and corporate actions from local OpenD."""

import argparse
import json
import time

import pandas as pd

from paper_futu.cn_common import STATE_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--repair-history", action="store_true")
    args = parser.parse_args()
    directory = STATE_DIR / args.date
    members = pd.read_csv(directory / "members.csv", dtype={"code": str})
    if len(members) != 300 or members.code.nunique() != 300:
        raise ValueError("CSI300 list must contain exactly 300 stocks")
    from futu import AuType, KLType, OpenQuoteContext, RET_OK
    quote = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        if args.repair_history:
            gaps = json.loads((directory / "history_gaps.json").read_text(encoding="utf-8"))
            for code in gaps:
                status, history, next_page = quote.request_history_kline(
                    code, start=gaps[code], end=args.date, ktype=KLType.K_DAY,
                    autype=AuType.NONE, max_count=500,
                )
                if status != RET_OK or next_page is not None:
                    raise RuntimeError(f"Missing history repair failed {code}: {history}")
                (directory / "history").mkdir(exist_ok=True)
                history.to_csv(directory / "history" / f"{code}.csv", index=False)
                print(f"Repaired {code}: {len(history)} unadjusted historical bars", flush=True)
            return
        snapshots = []
        for begin in range(0, 300, 100):
            status, frame = quote.get_market_snapshot(members.code.iloc[begin:begin + 100].tolist())
            if status != RET_OK:
                raise RuntimeError(f"行情快照失败：{frame}")
            snapshots.append(frame)
        snapshots = pd.concat(snapshots, ignore_index=True)
        if snapshots.code.nunique() != 300:
            raise ValueError("Snapshot does not cover all CSI300 members")
        snapshots.to_csv(directory / "snapshots.csv", index=False)
        # Individual calls are paced below the official 60 requests/30 seconds cap.
        actions = []
        for i, code in enumerate(members.code):
            cache = directory / "rehab" / f"{code}.json"
            if cache.exists():
                records = json.loads(cache.read_text(encoding="utf-8"))
            else:
                status, frame = quote.get_rehab(code)
                if status != RET_OK:
                    raise RuntimeError(f"复权信息失败 {code}：{frame}")
                records = json.loads(frame.to_json(orient="records"))
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(records), encoding="utf-8")
                time.sleep(0.55)
            for record in records:
                record["code"] = code
                actions.append(record)
            if (i + 1) % 50 == 0:
                print(f"Corporate actions checked {i + 1}/300", flush=True)
        pd.DataFrame(actions).to_csv(directory / "corporate_actions.csv", index=False)
        print(f"Saved live A-share market data: {directory}", flush=True)
    finally:
        quote.close()


if __name__ == "__main__":
    main()
