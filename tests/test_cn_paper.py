"""Execution constraints must hold before a CN paper order can be submitted."""

import numpy as np
import pandas as pd
import pytest

from paper_futu.check_connection import stock_paper_accounts
from paper_futu.cn_common import select_holdings, sellable_quantity, target_share_quantity, tiered_weights, validate_execution_session
from paper_futu.build_cn_signal import adjusted_bar, read_saved_bars
from paper_futu.cn_common import validate_plan_deployment
import paper_futu.cn_common as common
import hashlib
import json
from src.forward_labels import label_contract


def test_cn_filter_never_uses_live_hkcc_or_us():
    accounts = pd.DataFrame({"acc_id": [1, 2, 3], "trd_env": ["SIMULATE", "REAL", "SIMULATE"],
        "sim_acc_type": ["STOCK", "STOCK", "STOCK_AND_OPTION"],
        "trdmarket_auth": [["CN"], ["HKCC"], ["US"]], "acc_status": ["ACTIVE"] * 3})
    assert stock_paper_accounts(accounts, "CN").acc_id.tolist() == [1]


def test_rounding_never_uses_money_above_the_target():
    assert target_share_quantity("SH.600000", 0.038, 1000000, 9.67) == 3900
    assert target_share_quantity("SZ.300308", 0.038, 1000000, 724.47) == 0
    assert target_share_quantity("SH.688506", 0.038, 1000000, 228) == 0
    assert target_share_quantity("SH.688396", 0.038, 1000000, 50.15) == 757


def test_t_plus_one_checks_both_platform_and_today_purchases():
    assert sellable_quantity(500, 500, 500) == 0
    assert sellable_quantity(500, 100, 200) == 100
    assert sellable_quantity(500, 500, 200) == 300


@pytest.mark.parametrize("today,sh,sz", [
    ("2026-10-12", "MORNING", "MORNING"),
    ("2026-10-14", "MORNING", "MORNING"),
    ("2026-10-13", "CLOSED", "CLOSED"),
    ("2026-10-13", "REST", "REST"),
    ("2026-10-13", "MORNING", "CLOSED"),
])
def test_execution_rejects_stale_future_closed_and_lunch_sessions(today, sh, sz):
    with pytest.raises(ValueError):
        validate_execution_session(today, "2026-10-13", 2026, sh, sz)


def test_valid_next_day_session():
    validate_execution_session("2026-10-13", "2026-10-13", 2026, "MORNING", "MORNING")


def test_weights_keep_five_percent_cash_and_tier_exposure():
    codes = [f"SH.{600000 + i}" for i in range(50)]
    weights = tiered_weights(codes)
    assert sum(weights.values()) == pytest.approx(0.95)
    assert sum(weights[code] for code in codes[:10]) == pytest.approx(0.38)
    assert sum(weights[code] for code in codes[10:]) == pytest.approx(0.57)


def test_rebalance_replaces_only_the_worst_five_when_already_full():
    predictions = {str(i): float(i) for i in range(100)}
    current = [str(i) for i in range(50)]
    selected = select_holdings(predictions, list(predictions), current)
    assert len(selected) == 50
    assert set(current).difference(selected) == set(map(str, range(5)))
    assert set(selected).difference(current) == set(map(str, range(95, 100)))


def test_top10_weights_and_three_replacements():
    codes = [str(i) for i in range(10)]
    weights = tiered_weights(codes, top_k=10, top_tier_size=2, top_tier_weight=0.4)
    assert sum(weights.values()) == pytest.approx(0.95)
    assert weights[codes[0]] == pytest.approx(0.19)
    assert weights[codes[2]] == pytest.approx(0.07125)
    scores = {str(i): float(i) for i in range(20)}
    selected = select_holdings(scores, list(scores), codes, top_k=10, n_drop=3)
    assert len(selected) == 10
    assert set(codes) - set(selected) == {"0", "1", "2"}
    assert set(selected) - set(codes) == {"17", "18", "19"}


def test_initial_top10_opens_ten_candidates_not_three():
    scores = {str(i): float(i) for i in range(20)}
    selected = select_holdings(scores, list(scores), [], top_k=10, n_drop=3)
    assert len(selected) == 10
    assert set(selected) == {str(i) for i in range(10, 20)}


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    model_dir = tmp_path / "models/top10"
    model_dir.mkdir(parents=True)
    model = b"test fitted trees"
    (model_dir / "model.txt").write_bytes(model)
    metadata = {**label_contract(10), "model_sha256": hashlib.sha256(model).hexdigest(), "model_year": 2026,
                "label_horizon": 10, "portfolio": {"top_k": 10, "n_drop": 3,
                "top_tier_size": 2, "top_tier_weight": 0.4, "risk_degree": 0.95, "rebalance_every": 10}}
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
    config_path = tmp_path / "deployment.json"
    config = {"model_dir": "models/top10", "environment": "SIMULATE", "market": "CN", "account_suffix": "0000",
              "label_price_basis": "open_to_open"}
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(common, "DEPLOYMENT_PATH", config_path)
    return config, config_path, metadata


def test_deployed_portfolio_is_checked_even_for_identical_tree_hashes(deployment):
    _, _, metadata = deployment
    model_dir, actual = common.load_deployment()
    plan = {"model_dir": "models/top10", "model_sha256": metadata["model_sha256"],
            "model_year": 2026, **label_contract(10), **metadata["portfolio"]}
    validate_plan_deployment(plan, model_dir, actual)
    plan["n_drop"] = 5
    with pytest.raises(ValueError, match="portfolio mismatch"):
        validate_plan_deployment(plan, model_dir, actual)


@pytest.mark.parametrize("field,value", [("environment", "REAL"), ("market", "US"), ("account_suffix", "9999")])
def test_deployment_cannot_select_live_or_another_account(deployment, field, value):
    config, config_path, _ = deployment
    config[field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="CN simulated account"):
        common.load_deployment()


def test_deployed_model_integrity_check(deployment):
    (common.ROOT / "models/top10/model.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        common.load_deployment()


def test_cash_dividend_updates_price_factor_and_volume_units():
    previous = pd.Series({"close": 20.0, "factor": 2.0}, name=("SH600000", pd.Timestamp("2026-10-09")))
    snapshot = {"code": "SH.600000", "prev_close_price": 9.0, "suspension": False,
        "open_price": 9.1, "last_price": 9.2, "high_price": 9.3, "low_price": 9.0, "volume": 1000}
    actions = pd.DataFrame({"code": ["SH.600000"], "ex_div_date": ["2026-10-12"],
                            "forward_adj_factorA": [1.0], "forward_adj_factorB": [-1.0]})
    bar = adjusted_bar(previous, snapshot, actions, pd.Timestamp("2026-10-12"))
    assert bar["factor"] == pytest.approx(20 / 9)
    assert bar["close"] / previous["close"] - 1 == pytest.approx(9.2 / 9 - 1)
    assert bar["volume"] * bar["factor"] == pytest.approx(10)


def test_pre_fix_saved_bar_is_read_in_hands_without_rewriting(tmp_path):
    directory = tmp_path / "2026-10-12"
    directory.mkdir()
    path = directory / "bars.parquet"
    pd.DataFrame({"volume": [1000.0]}).to_parquet(path)
    assert read_saved_bars(path).volume.iloc[0] == pytest.approx(10.0)
    assert pd.read_parquet(path).volume.iloc[0] == pytest.approx(1000.0)
    (directory / "bars_units.json").write_text(
        json.dumps({"volume": "hands_divided_by_adjustment_factor"}), encoding="utf-8"
    )
    assert read_saved_bars(path).volume.iloc[0] == pytest.approx(1000.0)


def test_bad_previous_reference_aborts_instead_of_fabricating_a_return():
    previous = pd.Series({"close": 20.0, "factor": 2.0}, name=("SH600000", pd.Timestamp("2026-10-09")))
    snapshot = {"code": "SH.600000", "prev_close_price": 15, "suspension": False}
    actions = pd.DataFrame(columns=["code", "ex_div_date", "forward_adj_factorA", "forward_adj_factorB"])
    with pytest.raises(ValueError):
        adjusted_bar(previous, snapshot, actions, pd.Timestamp("2026-10-12"))
