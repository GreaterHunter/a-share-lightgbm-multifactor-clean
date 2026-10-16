"""S6 deployment invariants; never connect to OpenD or submit an order."""
from datetime import datetime
import json

import numpy as np
import pandas as pd
import pytest

from paper_futu import cn_common as common
from paper_futu import cn_deployment as transition
from paper_futu import deploy_s6_quarterly as deploy
from paper_futu.build_cn_signal import model_inputs, model_financials
from src.factor_library import ENHANCED_FEATURE_NAMES
from src.forward_labels import label_contract


def test_raw_model_is_never_fed_ranked_values():
    index = pd.MultiIndex.from_product([["SH600000","SZ000001"],[pd.Timestamp("2026-10-15")]],
        names=["instrument","datetime"])
    panel = pd.DataFrame({name:[10.,20.] for name in ENHANCED_FEATURE_NAMES},index=index)
    raw = model_inputs(panel,{"input_kind":"raw"})
    assert raw.iloc[0,0] == 10.
    assert model_inputs(panel,{"input_kind":"rank"}).iloc[0,0] == 0.
    assert raw.dtypes.eq("float32").all()
    with pytest.raises(ValueError): model_inputs(panel,{"input_kind":"unknown"})


def test_s6_weights_are_fractions_of_the_equity_sleeve():
    codes = list(map(str,range(20)))
    weights = common.tiered_weights(codes,top_k=20,top_tier_size=6,top_tier_weight=.4)
    assert sum(weights.values()) == pytest.approx(.95)
    assert sum(weights[c] for c in codes[:6]) == pytest.approx(.38)
    assert weights["0"] == pytest.approx(.06333333333333333)
    assert weights["6"] == pytest.approx(.040714285714285716)


def test_fill_empty_slots_is_not_counted_as_replacing_six_old_names():
    scores = {str(i):float(i) for i in range(40)}
    current = list(map(str,range(9)))
    selected = common.select_audited_holdings(scores,list(scores),current,20,6,current)
    assert len(selected) == 20
    assert len(set(current)-set(selected)) == 6
    assert len(set(selected)-set(current)) == 17


def test_preserve_unsellable_and_missing_score_holdings():
    scores = {str(i):float(i) for i in range(1,40)}
    current = list(map(str,range(20)))
    selected = common.select_audited_holdings(scores,list(scores),current,20,6,current[1:])
    assert "0" in selected
    assert len(set(current)-set(selected)) == 6
    with pytest.raises(ValueError):
        common.select_audited_holdings(scores,list(scores),list(map(str,range(21))),20,6,current)


def test_name_selection_matches_audited_research_for_varied_accounts():
    from paper_futu.final_ablation_account import select_holdings
    rng = np.random.default_rng(42)
    codes = list(map(str,range(40)))
    for _ in range(30):
        score = pd.Series(rng.normal(size=40),index=codes)
        score.iloc[rng.choice(40,3,replace=False)] = np.nan
        held = rng.choice(codes,int(rng.integers(0,21)),replace=False).tolist()
        buyable = [c for c in codes if pd.notna(score[c]) and rng.random() > .1]
        sellable = [c for c in held if rng.random() > .1]
        assert common.select_audited_holdings(score.to_dict(),buyable,held,20,6,sellable) == select_holdings(score,held,20,6,buyable,sellable)


def test_new_five_day_cadence_remains_anchored_to_original_signal():
    calendar = pd.bdate_range("2026-10-12","2026-11-06").strftime("%Y-%m-%d").tolist()
    plan = {"signal_date":"2026-10-12"}
    assert deploy.next_signal(plan,calendar,"2026-10-16") == "2026-10-19"
    assert deploy.next_signal(plan,calendar,"2026-10-20") == "2026-10-26"


def test_serving_quarter_expires_without_a_new_validated_model():
    metadata = {"retrain_months":3,"information_cutoff":"2026-09-30","serving_signal_end":"2026-12-31"}
    common.validate_serving_date(metadata,"2026-10-19")
    for day in ["2026-09-29","2027-01-04"]:
        with pytest.raises(ValueError,match="quarter"):
            common.validate_serving_date(metadata,day)


def test_financial_event_snapshot_is_pinned_and_causal(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path)
    import paper_futu.build_cn_signal as signal
    monkeypatch.setattr(signal,"ROOT",tmp_path)
    path = tmp_path / "events.parquet"
    pd.DataFrame({"available_date":pd.to_datetime(["2026-09-01","2026-10-20"])}).to_parquet(path)
    metadata = dict(financial_events_path="events.parquet",financial_events_sha256=transition.digest(path))
    assert len(model_financials(tmp_path,pd.Timestamp("2026-10-19"),metadata)) == 1
    path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="snapshot changed"):
        model_financials(tmp_path,pd.Timestamp("2026-10-19"),metadata)


def test_versioned_strategy_change_requires_both_explicit_binding_and_authorization(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path)
    model = tmp_path / "models/new"
    model.mkdir(parents=True)
    (model / "model.txt").write_bytes(b"model")
    current = dict(model_sha256="previous",portfolio={"top_k":10})
    metadata = {**label_contract(10),"strategy_id":deploy.STRATEGY,"portfolio":deploy.PORTFOLIO,
        "model_sha256":transition.digest(model / "model.txt")}
    transition.write_json(model / "metadata.json",metadata)
    pending = dict(kind="authorized_strategy_revision",previous_model_sha256="previous",
        model_dir="models/new",model_sha256=metadata["model_sha256"],
        metadata_sha256=transition.digest(model / "metadata.json"),target_strategy_id=deploy.STRATEGY,
        target_portfolio=deploy.PORTFOLIO,user_authorization="User selected this strategy")
    assert transition.pending_model({"pending_deployment":pending},current)[1] == metadata
    pending.pop("user_authorization")
    with pytest.raises(ValueError,match="binding changed"):
        transition.pending_model({"pending_deployment":pending},current)
    pending.pop("kind")
    with pytest.raises(ValueError,match="must not change"):
        transition.pending_model({"pending_deployment":pending},current)


def test_signal_metadata_cannot_silently_revert_to_ranked_features(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path)
    model = tmp_path / "models/new"
    model.mkdir(parents=True)
    metadata = {**label_contract(10),"strategy_id":deploy.STRATEGY,"model_sha256":"hash","model_year":2026,
        "portfolio":deploy.PORTFOLIO,"input_kind":"raw","retrain_months":3,
        "information_cutoff":"2026-09-30","serving_signal_end":"2026-12-31",
        "selection_rule":"audited_limited_replacement_v1"}
    plan = {**metadata,**deploy.PORTFOLIO,"model_dir":"models/new","signal_date":"2026-10-19"}
    common.validate_plan_deployment(plan,model,metadata)
    plan["input_kind"] = "rank"
    with pytest.raises(ValueError,match="input_kind"):
        common.validate_plan_deployment(plan,model,metadata)


def test_authorized_switch_activates_full_portfolio_contract_and_preserves_recovery(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path)
    state = tmp_path / "state"; state.mkdir()
    monkeypatch.setattr(common,"STATE_DIR",state)
    config_path = tmp_path / "deployment.json"
    monkeypatch.setattr(common,"DEPLOYMENT_PATH",config_path)
    monkeypatch.setattr(common,"china_now",lambda:datetime(2026,10,19,tzinfo=common.CHINA_TZ))
    old_portfolio = {**deploy.PORTFOLIO,"top_k":10,"n_drop":3,"top_tier_size":2,"rebalance_every":10}
    data = []
    for name,portfolio in [("old",old_portfolio),("new",deploy.PORTFOLIO)]:
        directory = tmp_path / "models" / name; directory.mkdir(parents=True)
        (directory / "model.txt").write_bytes(name.encode())
        meta = {**label_contract(10),"model_year":2026,"portfolio":portfolio,
            "model_sha256":transition.digest(directory / "model.txt")}
        if name == "new":
            meta.update(strategy_id=deploy.STRATEGY,input_kind="raw",retrain_months=3,
                information_cutoff="2026-09-30",serving_signal_end="2026-12-31",
                selection_rule="audited_limited_replacement_v1")
        transition.write_json(directory / "metadata.json",meta);data.append(meta)
    old,new = data
    active = {**old,**old_portfolio,"model_dir":"models/old","signal_date":"2026-10-12",
        "execution_date":"2026-10-13","next_rebalance_signal_date":"2026-10-19"}
    transition.write_json(state / "active_plan.json",active)
    transition.write_json(state / "catchup_plan.json",{"execution_date":"2026-10-14","status":"cancelled"})
    recovery_hash = transition.digest(state / "catchup_plan.json")
    pending = dict(kind="authorized_strategy_revision",transition_id="authorized-test",
        previous_model_sha256=old["model_sha256"],model_dir="models/new",model_sha256=new["model_sha256"],
        metadata_sha256=transition.digest(tmp_path / "models/new/metadata.json"),
        target_strategy_id=deploy.STRATEGY,target_portfolio=deploy.PORTFOLIO,user_authorization="selected by user",
        effective_signal_date="2026-10-19",source_plan_sha256=transition.digest(state / "active_plan.json"))
    transition.write_json(config_path,dict(environment="SIMULATE",market="CN",account_suffix="0000",
        model_dir="models/old",label_price_basis="open_to_open",pending_deployment=pending))
    new_plan = {**new,**deploy.PORTFOLIO,"model_dir":"models/new","signal_date":"2026-10-19",
        "execution_date":"2026-10-20","next_rebalance_signal_date":"2026-10-26"}
    assert transition.model_for_signal("2026-10-16")[0].name == "old"
    assert transition.model_for_signal("2026-10-19")[0].name == "new"
    assert transition.activate_pending(new_plan)
    directory,metadata = common.load_deployment()
    common.validate_plan_deployment(read_json(state / "active_plan.json"),directory,metadata)
    assert metadata["input_kind"] == "raw"
    assert recovery_hash == transition.digest(state / "catchup_plan.json")
    assert read_json(config_path)["model_metadata_sha256"] == pending["metadata_sha256"]
    metadata["input_kind"] = "rank"
    transition.write_json(directory / "metadata.json",metadata)
    with pytest.raises(ValueError,match="metadata changed"):common.load_deployment()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
