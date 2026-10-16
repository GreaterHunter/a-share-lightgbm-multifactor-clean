"""Maturity and deployment consistency, without contacting any broker."""

from datetime import datetime
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from paper_futu import cn_common as common
from paper_futu import cn_deployment as deployment
from src.forward_labels import label_contract, label_end_dates, select_mature_period
from src.train_walk_forward_cv import select_training_period


@pytest.mark.parametrize("horizon", [5, 10, 15, 20, 30, 40, 60])
def test_all_active_labels_use_open_to_open(horizon):
    spec = label_contract(horizon)
    assert spec["label_expression"] == f"Ref($open, -{horizon + 1}) / Ref($open, -1) - 1"
    assert spec["label_end_offset"] == horizon + 1


def test_main_folds_remove_only_outcomes_beyond_actual_boundary():
    calendar = pd.bdate_range("2020-12-01", "2021-01-29")
    index = pd.MultiIndex.from_product([["SH600000", "SZ000001"], calendar], names=["instrument", "datetime"])
    frame = pd.DataFrame({"factor": 1., "label_end_date": label_end_dates(index, calendar, 10)}, index=index)
    retained = select_training_period(frame, "2010-01-01", "2020-12-31", 10, "future_open_return_10d")
    assert retained.label_end_date.max() == pd.Timestamp("2020-12-31")
    assert retained.index.get_level_values("datetime").max() == pd.Timestamp("2020-12-16")
    pd.testing.assert_frame_equal(retained, select_mature_period(retained, "2010-01-01", "2020-12-31"))
    frame.loc[frame.label_end_date.gt("2020-12-31"), "factor"] = 999.
    pd.testing.assert_frame_equal(retained, select_mature_period(frame, "2010-01-01", "2020-12-31"))


def test_open_label_fold_rejects_old_dataset_even_if_column_is_renamed():
    index = pd.MultiIndex.from_tuples([("SH600000", pd.Timestamp("2020-12-01"))], names=["instrument", "datetime"])
    with pytest.raises(ValueError, match="actual label_end_date"):
        select_training_period(pd.DataFrame({"future_open_return_10d": [0.1]}, index=index),
                               "2010-01-01", "2020-12-31", 10, "future_open_return_10d")


@pytest.fixture
def transition(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(common, "STATE_DIR", state)
    config_path = tmp_path / "deployment.json"
    monkeypatch.setattr(common, "DEPLOYMENT_PATH", config_path)
    monkeypatch.setattr(common, "china_now", lambda: datetime(2026, 10, 13, tzinfo=common.CHINA_TZ))
    portfolio = {"top_k": 10, "n_drop": 3, "top_tier_size": 2, "top_tier_weight": .4,
                 "risk_degree": .95, "rebalance_every": 10}
    models = []
    for name, spec in [("old", {}), ("new", label_contract(10))]:
        directory = tmp_path / "models" / name
        directory.mkdir(parents=True)
        binary = name.encode()
        (directory / "model.txt").write_bytes(binary)
        metadata = {**spec, "model_sha256": hashlib.sha256(binary).hexdigest(), "label_horizon": 10,
                    "model_year": 2026, "portfolio": portfolio}
        (directory / "metadata.json").write_text(json.dumps(metadata))
        models.append(metadata)
    old, new = models
    old_plan = {"model_dir": "models/old", "model_sha256": old["model_sha256"], "model_year": 2026,
                "label_horizon": 10, **portfolio, "signal_date": "2026-10-12", "execution_date": "2026-10-13",
                "next_rebalance_signal_date": "2026-10-26"}
    deployment.write_json(state / "active_plan.json", old_plan)
    catchup = {"execution_date": "2026-10-14", "code": "SZ.300316", "model_sha256": old["model_sha256"],
               "source_execution_date": "2026-10-13", "max_qty": 2100}
    deployment.write_json(state / "catchup_plan.json", catchup)
    config = {"environment": "SIMULATE", "market": "CN", "account_suffix": "0000", "model_dir": "models/old",
              "label_price_basis": "legacy_open_to_close", "pending_deployment": {
                  "transition_id": "test_transition", "model_dir": "models/new", "model_sha256": new["model_sha256"],
                  "previous_model_sha256": old["model_sha256"], "effective_signal_date": "2026-10-26",
                  "source_plan_sha256": deployment.digest(state / "active_plan.json")}}
    deployment.write_json(config_path, config)
    new_plan = {**old_plan, **label_contract(10), "model_dir": "models/new", "model_sha256": new["model_sha256"],
                "signal_date": "2026-10-26", "execution_date": "2026-10-27", "next_rebalance_signal_date": "2026-11-09"}
    return state, config, new_plan


def test_prepared_model_does_not_replace_original_recovery_plan(transition):
    state, _, _ = transition
    before = deployment.digest(state / "active_plan.json"), deployment.digest(state / "catchup_plan.json")
    assert deployment.model_for_signal("2026-10-13")[0].name == "old"
    assert deployment.model_for_signal("2026-10-26")[0].name == "new"
    assert before == (deployment.digest(state / "active_plan.json"), deployment.digest(state / "catchup_plan.json"))


def test_unresolved_old_catchup_blocks_switch(transition):
    _, _, plan = transition
    with pytest.raises(ValueError, match="catchup is unresolved"):
        deployment.activate_pending(plan)


def test_explicitly_cancelled_catchup_does_not_block_later_scheduled_switch(transition):
    state, _, plan = transition
    catchup_path = state / "catchup_plan.json"
    catchup = json.loads(catchup_path.read_text(encoding="utf-8"))
    deployment.write_json(catchup_path, {**catchup, "status": "cancelled"})
    assert deployment.activate_pending(plan)
    assert not (state / "catchup_complete_2026-10-14.json").exists()


def test_successful_transition_preserves_timing_and_does_not_rebind_old_order(transition):
    state, _, plan = transition
    old_catchup_hash = deployment.digest(state / "catchup_plan.json")
    deployment.write_json(state / "catchup_complete_2026-10-14.json", {
        "status": "filled_all", "code": "SZ.300316", "qty": 2100,
        "source_execution_date": "2026-10-13", "execution_date": "2026-10-14"})
    assert deployment.activate_pending(plan)
    model_dir, metadata = common.load_deployment()
    assert model_dir.name == "new"
    common.validate_plan_deployment(json.loads((state / "active_plan.json").read_text()), model_dir, metadata)
    assert old_catchup_hash == deployment.digest(state / "catchup_plan.json")
    assert not deployment.activate_pending(plan)


def test_transition_rejects_off_cycle_plan_and_modified_candidate(transition):
    _, _, plan = transition
    plan["signal_date"] = "2026-10-25"
    assert not deployment.activate_pending(plan)
    (common.ROOT / "models/new/model.txt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash changed"):
        deployment.model_for_signal("2026-10-26")


def test_open_model_rejects_legacy_label_plan(transition):
    _, _, plan = transition
    directory, metadata = deployment.model_for_signal("2026-10-26")
    plan["label_expression"] = "Ref($close, -10) / Ref($open, -1) - 1"
    with pytest.raises(ValueError, match="label contract mismatch"):
        common.validate_plan_deployment(plan, directory, metadata)


def test_unmarked_legacy_deployment_is_rejected(transition):
    _, config, _ = transition
    config.pop("pending_deployment")
    deployment.write_json(common.DEPLOYMENT_PATH, config)
    with pytest.raises(ValueError, match="unexpired"):
        common.load_deployment()


def test_corrupt_catchup_completion_does_not_enable_switch(transition):
    state, _, plan = transition
    deployment.write_json(state / "catchup_complete_2026-10-14.json", {"status": "failed"})
    with pytest.raises(ValueError, match="completion is invalid"):
        deployment.activate_pending(plan)


def test_open_backtest_uses_open_limits_and_cash_warmup(monkeypatch):
    from src import analyze_portfolio_robustness as backtest
    signal_date, trade_date = pd.Timestamp("2022-01-04"), pd.Timestamp("2022-01-05")
    index = pd.MultiIndex.from_tuples([("SH600000", signal_date)], names=["instrument", "datetime"])
    frame = pd.DataFrame({"prediction": [.1]}, index=index)
    monkeypatch.setattr(backtest, "find_first_trade_date", lambda **kwargs: trade_date)
    monkeypatch.setattr(backtest, "TieredTopKWeightStrategy", lambda **kwargs: object())
    monkeypatch.setattr(backtest, "SimulatorExecutor", lambda **kwargs: object())
    calls = {}

    def simulate(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({"account": [1_000_000., 1_000_000.], "turnover": [0., .9],
                             "bench": [.05, .03]}, index=[signal_date, trade_date]), {}

    monkeypatch.setattr(backtest, "backtest_daily", simulate)
    quote_index = pd.MultiIndex.from_tuples([("SH000300", trade_date)], names=["instrument", "datetime"])
    monkeypatch.setattr(backtest, "D", SimpleNamespace(features=lambda *args, **kwargs:
                        pd.DataFrame({"$open": [100.], "$close": [102.]}, index=quote_index)))
    config = {"label_price_basis": "open_to_open", "n_drop": 3, "risk_degree": .95,
              "initial_cash": 1_000_000, "benchmark": "SH000300", "limit_threshold": .095,
              "buy_cost": .0005, "sell_cost": .0015, "minimum_cost": 5.}
    result = backtest.run_tiered_backtest(frame, config, 10, top_k=10, top_tier_size=2, period_end="2022-01-05")
    assert calls["start_time"] == signal_date
    assert calls["exchange_kwargs"]["deal_price"] == "open"
    assert calls["exchange_kwargs"]["limit_threshold"] == (
        "($open / Ref($close, 1) - 1) >= 0.095", "($open / Ref($close, 1) - 1) <= -0.095")
    assert result.index.tolist() == [trade_date]
    assert result.bench.iloc[0] == pytest.approx(.02)
