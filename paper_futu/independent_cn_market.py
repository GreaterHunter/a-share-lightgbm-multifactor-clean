"""Public daily bars, cross-checked without importing Futu or contacting OpenD.

Sina supplies raw share volume and dated cumulative adjustment factors. Tencent
independently checks OHLC/volume. Adjustment factors are anchored to the existing
verified Qlib units, never copied across providers as absolute values.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
SINA_URL = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
FACTOR_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"
OHLC = ["open", "close", "high", "low"]


def verified_cash_deduction(instrument,day,action):
    """Only issuer-evidenced diluted ex-dividend deductions; never blanket tolerance."""
    from paper_futu.cn_common import ROOT
    path=ROOT/'configs/verified_cn_cash_actions.json'
    if not path.exists():return None
    entries=json.loads(path.read_text(encoding='utf-8'))['events']
    found=[r for r in entries if r['instrument']==instrument and r['ex_date']==str(pd.Timestamp(day).date())]
    if not found:return None
    if len(found)!=1:raise ValueError('Duplicate verified cash action')
    event=found[0]
    declared=event['declared_cash_per_10']
    source_declared=declared
    if 'source_cash_decimal_places' in event:
        from decimal import Decimal, ROUND_DOWN
        places=event['source_cash_decimal_places']
        if type(places) is not int or not 0 <= places <= 8:
            raise ValueError('Invalid verified source dividend precision')
        source_declared=float(Decimal(str(declared)).quantize(Decimal(10)**-places,rounding=ROUND_DOWN))
    if (event['announcement_date']>event['ex_date'] or
        sha(ROOT/event['evidence_path'])!=event['evidence_sha256'] or
        not (np.isclose(float(action['fh_sh']),declared,rtol=0,atol=1e-10)
             or np.isclose(float(action['fh_sh']),source_declared,rtol=0,atol=1e-10))):
        raise ValueError('Verified cash action identity/evidence mismatch')
    diluted=event['announced_total_cash']/event['total_shares']
    computed=event['participating_shares']*(event['declared_cash_per_10']/10)
    if (abs(computed-event['announced_total_cash'])>.02 or
        abs(diluted-event['effective_ex_dividend_per_share'])>1e-7):
        raise ValueError('Issuer diluted dividend arithmetic mismatch')
    return float(event['effective_ex_dividend_per_share'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parse_factor(text):
    # JSON only: the remote JavaScript is data, never eval/execute it.
    start = text.find("=") + 1
    value, _ = json.JSONDecoder().raw_decode(text[start:].lstrip())
    frame = pd.DataFrame(value["data"]).rename(columns={"d": "date", "f": "factor"})
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d")
    frame["factor"] = pd.to_numeric(frame.factor, errors="raise")
    if frame.date.duplicated().any() or not np.isfinite(frame.factor).all() or frame.factor.le(0).any():
        raise ValueError("Invalid dated adjustment factors")
    return frame.set_index("date").sort_index().factor


def factor_on(factors, day):
    known = factors.loc[factors.index <= pd.Timestamp(day)]
    if known.empty:
        raise ValueError("No adjustment factor effective on requested date")
    return float(known.iloc[-1])


def parse_sina(payload):
    frame = pd.DataFrame(payload).rename(columns={"day": "date"})
    if frame.empty or not {"date", *OHLC, "volume"}.issubset(frame.columns):
        raise ValueError("Missing Sina daily schema")
    frame.date = pd.to_datetime(frame.date, format="%Y-%m-%d")
    for name in OHLC + ["volume"]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if frame.date.duplicated().any():
        raise ValueError("Duplicate Sina stock dates")
    return frame.set_index("date").sort_index()[OHLC + ["volume"]]


def parse_tencent(payload, instrument):
    symbol = instrument.lower()
    if payload.get("code") != 0:
        raise ValueError("Tencent returned an error")
    data = payload["data"][symbol]
    rows = data.get("day", [])
    frame = pd.DataFrame([row[:6] for row in rows], columns=["date", *OHLC, "volume"])
    frame.date = pd.to_datetime(frame.date, format="%Y-%m-%d")
    for name in OHLC + ["volume"]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    # Mainland stock k-lines: STAR board uses shares; ordinary SH/SZ use hands.
    # Each stock is checked against Sina shares, so a unit/schema change fails.
    if not instrument.startswith("SH688"):
        frame.volume *= 100
    if frame.date.duplicated().any():
        raise ValueError("Duplicate Tencent stock dates")
    quote = data.get("qt", {}).get(symbol, [])
    frame = frame.set_index("date").sort_index()
    frame.attrs["actions"] = {row[0]:row[6] for row in rows if len(row) > 6 and row[6]}
    return frame, quote


def compare_raw(left, right, label):
    for name in OHLC:
        a, b = float(left[name]), float(right[name])
        if not np.isfinite([a, b]).all() or min(a, b) <= 0 or abs(a-b) > max(.011, abs(a)*1e-5):
            raise ValueError(f"OHLC cross-source mismatch: {label}/{name}: {a} vs {b}")
    a, b = float(left["volume"]), float(right["volume"])
    if not np.isfinite([a, b]).all() or min(a, b) <= 0 or abs(a-b) > max(100., abs(a)*1e-4):
        raise ValueError(f"Share-volume cross-source mismatch: {label}: {a} vs {b}")
    if left["low"] > min(left["open"], left["close"]) or left["high"] < max(left["open"], left["close"]):
        raise ValueError("Inconsistent daily OHLC range")


def raw_from_adjusted(row):
    factor = float(row.factor)
    return {**{key: float(row[key])/factor for key in OHLC}, "volume": float(row.volume)*100*factor}


def normalize_stock(instrument, previous, last_valid, sina, tencent, factors, quote, dates):
    """Return consecutive appended bars; a missing quote is not a suspension."""
    result, suspensions = [], []
    previous_date = pd.Timestamp(previous.name[1])
    anchor_date = pd.Timestamp(last_valid.name[1])
    if anchor_date not in sina.index or anchor_date not in tencent.index:
        raise ValueError(f"Missing normalization anchor: {instrument}/{anchor_date.date()}")
    compare_raw(sina.loc[anchor_date], tencent.loc[anchor_date], instrument + "/anchor")
    compare_raw(raw_from_adjusted(last_valid), sina.loc[anchor_date], instrument + "/saved-anchor")
    scale = float(last_valid.factor)/factor_on(factors, anchor_date)
    for day in dates:
        day = pd.Timestamp(day)
        if day <= previous_date:
            raise ValueError("New bars overlap their saved anchor")
        in_sina, in_tx = day in sina.index, day in tencent.index
        if in_sina != in_tx:
            raise ValueError(f"Incomplete/stale daily feed: {instrument}/{day.date()}")
        if not in_sina:
            # Only accept a current explicitly zero-open/zero-volume quote plus
            # the last verified suspended session. Historical holes fail closed.
            quote_day = str(quote[30])[:8] if len(quote) > 30 else ""
            was_suspended = not np.isfinite(previous["close"])
            zero_quote = len(quote) > 30 and float(quote[5]) == 0 and float(quote[6]) == 0
            if not (was_suspended and zero_quote and quote_day >= day.strftime("%Y%m%d")
                    and sina.index.max() < day and tencent.index.max() < day):
                raise ValueError(f"Unverified suspension or missing daily bar: {instrument}/{day.date()}")
            values = {**{key: np.nan for key in OHLC + ["volume"]}, "factor": float(previous.factor)}
            suspensions.append(str(day.date()))
        else:
            raw = sina.loc[day]
            compare_raw(raw, tencent.loc[day], instrument + "/" + str(day.date()))
            # Independently confirm every adjustment change, not merely price
            # continuity. Cash distributions use precise per-10-share amounts;
            # unsupported rights/merger/bonus-share schemas require review.
            prior = sina.loc[sina.index < day]
            prior_day, prior_close = prior.index[-1], float(prior.iloc[-1].close)
            ratio = factor_on(factors, day)/factor_on(factors, prior_day)
            action = tencent.attrs.get("actions", {}).get(str(day.date()), {})
            expected = 1.
            rounded_expected = 1.
            if action:
                import re
                if (action.get("cqr") != str(day.date()) or "fh_sh" not in action
                        or re.search("送|转|配|合|拆", str(action.get("FHcontent", "")))
                        or set(action)-{"nd", "fh_sh", "djr", "cqr", "FHcontent"}):
                    raise ValueError(f"Unsupported corporate action; review required: {instrument}/{day.date()}")
                verified=verified_cash_deduction(instrument,day,action)
                dividend = float(action["fh_sh"])/10 if verified is None else verified
                if not 0 <= dividend < prior_close:
                    raise ValueError("Invalid cash dividend")
                expected = prior_close/(prior_close-dividend)
                # Providers may apply cash ex-rights to the exchange's cent-
                # rounded reference price (e.g. 5.04 - .08008 -> 4.96). Accept
                # precisely these two declared-event interpretations, not a
                # wider blanket tolerance for arbitrary factor mismatches.
                from decimal import Decimal, ROUND_HALF_UP
                reference = (Decimal(str(prior_close))-Decimal(str(dividend))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP)
                if reference <= 0:
                    raise ValueError("Invalid rounded cash ex-rights price")
                rounded_expected = prior_close/float(reference)
            if not (np.isclose(ratio, expected, rtol=1e-6, atol=1e-9)
                    or np.isclose(ratio, rounded_expected, rtol=1e-6, atol=1e-9)):
                raise ValueError(f"Adjustment factor cross-source mismatch: {instrument}/{day.date()}")
            factor = scale * factor_on(factors, day)
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError("Invalid normalized adjustment factor")
            values = {**{key: float(raw[key])*factor for key in OHLC},
                      "volume": float(raw.volume)/100/factor, "factor": factor}
        result.append({"instrument": instrument, "datetime": day, **values})
        previous = pd.Series(values, name=(instrument, day))
        previous_date = day
    return result, suspensions


class PublicClient:
    def __init__(self, cache, pause=.2):
        self.cache = Path(cache)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
        self.pause = pause

    def get(self, symbol, name, url, params=None):
        path = self.cache/symbol/(name + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        for attempt in range(3):
            try:
                time.sleep(self.pause)
                # HTTPS-only alternate endpoint; never disable certificate verification.
                endpoint=url
                if name=='sina' and attempt>0:
                    endpoint='https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
                response = self.session.get(endpoint, params=params, timeout=(10, 25))
                response.raise_for_status()
                text = response.text
                value = text if name == "factor" else response.json()
                write_json(path, value)
                write_json(path.with_suffix(".source.json"), dict(url=response.url,
                    retrieved_at=datetime.now(timezone.utc).isoformat(), sha256=sha(path)))
                return value
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def stock(self, instrument, start, end):
        symbol = instrument.lower()
        sina = parse_sina(self.get(symbol, "sina", SINA_URL,
            dict(symbol=symbol, scale="240", ma="no", datalen="640")))
        tx, quote = parse_tencent(self.get(symbol, "tencent", TX_URL,
            dict(param=f"{symbol},day,{start},{end},640,")), instrument)
        factors = parse_factor(self.get(symbol, "factor", FACTOR_URL.format(symbol=symbol)))
        return sina, tx, factors, quote

    def invalidate(self, instrument):
        # A temporarily delayed daily bar must not poison the next scheduled
        # retry. Remove only these three cached HTTP responses, never live data.
        for name in ["sina", "tencent", "factor"]:
            for suffix in [".json", ".source.json"]:
                (self.cache/instrument.lower()/(name+suffix)).unlink(missing_ok=True)
