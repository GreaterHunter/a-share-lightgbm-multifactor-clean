from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_futu import independent_cn_market as market
from paper_futu import update_cn_data as updater
from paper_futu.cn_common import CHINA_TZ


def tx(symbol, volume):
    return {"code": 0, "data": {symbol.lower(): {"day": [["2026-10-16", "10", "11", "12", "9", str(volume)]]}}}


@pytest.mark.parametrize("instrument,volume", [("SZ000001", 100), ("SH600000", 100), ("SH688981", 10000)])
def test_tencent_board_specific_volume_is_cross_checked(instrument, volume):
    frame, _ = market.parse_tencent(tx(instrument, volume), instrument)
    assert frame.iloc[0].volume == 10000
    market.compare_raw(frame.iloc[0], dict(open=10, close=11, high=12, low=9, volume=10000), "unit")
    with pytest.raises(ValueError, match="volume"):
        market.compare_raw(frame.iloc[0], dict(open=10, close=11, high=12, low=9, volume=1000000), "unit")


def test_factor_data_is_json_not_executable_code():
    f = market.parse_factor('var sh600000hfq={"data":[{"d":"2026-10-15","f":"1.5"},{"d":"1900-01-01","f":"1"}]};/*comment*/')
    assert market.factor_on(f, "2026-10-14") == 1
    assert market.factor_on(f, "2026-10-15") == 1.5
    with pytest.raises(ValueError):
        market.parse_factor('var x=__import__("os").system("not executed")')
    with pytest.raises(ValueError, match="Invalid"):
        market.parse_factor('var x={"data":[{"d":"1900-01-01","f":"0"}]}')


def normalization_fixture():
    dates = pd.to_datetime(["2026-10-15", "2026-10-16"])
    source = pd.DataFrame(dict(open=[10., 9.], close=[10., 9.], high=[10., 9.], low=[10., 9.],
                               volume=[10000., 20000.]), index=dates)
    old = pd.Series(dict(open=5., close=5., high=5., low=5., volume=200., factor=.5), name=("SH600000", dates[0]))
    factors = pd.Series([1., 10/9], index=dates)
    source.attrs["actions"] = {"2026-10-16":dict(cqr="2026-10-16", fh_sh="10", FHcontent="10派10元")}
    return old, source, factors


def test_factor_ratio_preserves_price_and_volume_units_without_restatement():
    old, source, factors = normalization_fixture()
    original = old.copy()
    rows, halts = market.normalize_stock("SH600000", old, old, source, source, factors, [], ["2026-10-16"])
    assert halts == []
    assert rows[0]["close"] == pytest.approx(5.)
    assert rows[0]["factor"] == pytest.approx(5/9)
    assert rows[0]["volume"] == pytest.approx(360.)
    pd.testing.assert_series_equal(old, original)
    factors.loc[pd.Timestamp("2026-10-20")] = 999.
    later, _ = market.normalize_stock("SH600000", old, old, source, source, factors, [], ["2026-10-16"])
    assert later == rows  # Future corporate actions cannot alter an earlier bar.


def test_bad_anchor_and_missing_feed_block_publication():
    old, source, factors = normalization_fixture()
    bad = old.copy(); bad["volume"] *= 100
    with pytest.raises(ValueError, match="volume"):
        market.normalize_stock("SH600000", bad, bad, source, source, factors, [], ["2026-10-16"])
    with pytest.raises(ValueError, match="stale"):
        market.normalize_stock("SH600000", old, old, source, source.iloc[:1], factors, [], ["2026-10-16"])
    with pytest.raises(ValueError, match="Unverified suspension"):
        market.normalize_stock("SH600000", old, old, source.iloc[:1], source.iloc[:1], factors, [], ["2026-10-16"])
    unverified = source.copy(); unverified.attrs = {}
    with pytest.raises(ValueError, match="Adjustment factor"):
        market.normalize_stock("SH600000", old, old, source, unverified, factors, [], ["2026-10-16"])


def test_suspension_is_not_filled_with_zero_return_or_zero_volume():
    valid, source, factors = normalization_fixture()
    previous = valid.copy(); previous.loc[market.OHLC + ["volume"]] = np.nan
    quote = [""]*31; quote[5] = "0"; quote[6] = "0"; quote[30] = "20261016090000"
    rows, halts = market.normalize_stock("SH600000", previous, valid, source.iloc[:1], source.iloc[:1], factors,
                                         quote, ["2026-10-16"])
    assert halts == ["2026-10-16"]
    assert np.isnan(rows[0]["close"]) and np.isnan(rows[0]["volume"])
    assert rows[0]["factor"] == .5


def test_declared_cash_dividend_may_use_cent_rounded_reference():
    old, source, factors = normalization_fixture()
    source.loc[:,["open","close","high","low"]]=[[5.04]*4,[4.93]*4]
    old.loc[["open","close","high","low"]]=2.52
    source.attrs["actions"]={"2026-10-16":dict(cqr="2026-10-16",fh_sh="0.8008",FHcontent="10派0.8008元")}
    factors.iloc[1]=5.04/4.96
    rows,_=market.normalize_stock("SH600000",old,old,source,source,factors,[],["2026-10-16"])
    assert rows[0]["factor"]==pytest.approx(.5*5.04/4.96)
    factors.iloc[1]=1.02  # Neither exact nor cent-rounded declared reference.
    with pytest.raises(ValueError,match="Adjustment factor"):
        market.normalize_stock("SH600000",old,old,source,source,factors,[],["2026-10-16"])


@pytest.mark.parametrize("stamp,expected", [("2026-10-19T15:34:59", "2026-10-16"),
    ("2026-10-19T15:35:00", "2026-10-19"), ("2026-10-20T08:50:00", "2026-10-19"),
    ("2026-10-18T20:00:00", "2026-10-16")])
def test_completed_session_cutoff(stamp, expected):
    dates = pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20"])
    assert updater.completed_date(datetime.fromisoformat(stamp).replace(tzinfo=CHINA_TZ), dates) == expected


def test_intraday_request_rejected_before_any_network(monkeypatch):
    monkeypatch.setattr(updater, "china_now", lambda: datetime(2026, 10, 19, 10, tzinfo=CHINA_TZ))
    monkeypatch.setattr(updater, "calendar", lambda: pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20"]))
    with pytest.raises(ValueError, match="intraday"):
        updater.update("2026-10-19")


def test_prebuilt_package_tampering_rejected_before_qlib_or_broker(tmp_path, monkeypatch):
    from paper_futu import build_cn_signal as builder
    (tmp_path/"bars.parquet").write_bytes(b"not used")
    (tmp_path/"market_manifest.json").write_text(json.dumps(dict(status="validated", validation_schema=2, date="2026-10-16",
        artifacts={"bars.parquet": "incorrect"})))
    monkeypatch.setattr(builder.sys if hasattr(builder, "sys") else __import__("sys"), "argv",
        ["build", "--date", "2026-10-16", "--prebuilt-bars", str(tmp_path/"bars.parquet")])
    with pytest.raises(ValueError, match="artifact changed"):
        builder.main()


def test_data_protocol_revision_preserves_old_seal_and_blocks_unapproved_changes(tmp_path):
    from paper_futu import amend_s6_data_protocol as amendment
    root = tmp_path
    old = root/"reports/paper_futu/s6_forward_v1"
    bindings = {}
    for name in amendment.CHANGED + amendment.ADDED + ["models/frozen/model.txt"]:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("original")
        if name in amendment.CHANGED:
            archive = old/"source_before_independent_data"/name
            archive.parent.mkdir(parents=True, exist_ok=True); archive.write_text("original")
        if name not in amendment.ADDED:
            bindings[str(path)] = market.sha(path)
    market.write_json(old/"protocol.json", dict(version=1, immutable_bindings=bindings,
        first_execution_date="2026-10-20", signal_anchor="2026-10-19", strategy={"model_sha256":"unchanged"}))
    seal = market.sha(old/"protocol.json")
    market.write_json(old/"seal.json", dict(protocol_sha256=seal))
    for name in amendment.CHANGED:
        (root/name).write_text("new data branch")
    model = root/"models/frozen/model.txt"
    model.write_text("unauthorized model update")
    with pytest.raises(ValueError, match="Unapproved"):
        amendment.amend(root, datetime(2026, 10, 19, 10, tzinfo=CHINA_TZ))
    model.write_text("original")
    new = amendment.amend(root, datetime(2026, 10, 19, 10, tzinfo=CHINA_TZ))
    assert new["strategy"]["model_sha256"] == "unchanged"
    assert new["version"] == 2
    assert market.sha(old/"protocol.json") == seal
    assert new["data_source_amendment"]["parent_protocol_sha256"] == seal
