"""Decision-complete protocol and causal execution regression tests."""
import numpy as np
import pandas as pd
import pytest

from paper_futu.final_ablation_core import (
    GRID, SCHEDULES, choose_candidates, fee, identity, initial_specs, mature, offsets,
    parameter_configs, portfolios, serving_periods, window_start,
)
from paper_futu.final_ablation_account import AuditedExchange, AuditedOrderGenerator, select_holdings
from qlib.backtest.decision import Order
from qlib.backtest.position import Position
from paper_futu.run_final_ablation import FINE_TOP_PAIRS, choose_final, fine_pairs, search_counts


def test_frozen_search_counts_and_candidates():
    configs = parameter_configs()
    assert len(configs) == 24
    assert configs == parameter_configs()
    assert sum(map(len,SCHEDULES.values())) == 20
    assert len(initial_specs()) == 12
    assert len(portfolios()) == 88
    base = configs[0]["parameters"]
    assert all(sum(p["parameters"][k]!=base[k] for k in GRID)==1 for p in configs[1:16])
    assert all(sum(p["parameters"][k]!=base[k] for k in GRID)>1 for p in configs[16:])


def test_reduced_parameters_are_the_exact_original_first_24():
    # Recorded from the expanded protocol before the authorized scope change.
    assert identity(parameter_configs()) == "686eba256103a11b6dedbe4f41d656f2515c829dbd0467624ba3f3189fc3b0c5"
    assert [c["id"] for c in parameter_configs()] == [f"p{i:02d}" for i in range(24)]


def test_reduced_schedules_exclude_h60_and_fast_rebalances():
    assert SCHEDULES == {5:[5,10],10:[5,10,20,30],20:[5,10,20,30],
                         30:[5,10,15,20,30],40:[5,10,20,30,40]}
    assert 60 not in SCHEDULES
    assert not {1,3} & {r for values in SCHEDULES.values() for r in values}
    counts = search_counts()
    assert counts["parameter_configs"] == 24
    assert counts["horizon_rebalance_pairs"] == 20
    assert counts["coarse_replays_max"] == 240
    assert counts["fine_replays_max"] == 2112
    assert counts["h20_validation_fits_max"] == 504
    assert counts["portfolios_per_pair"] == 88


def test_fine_search_keeps_two_qualified_pairs_and_two_fixed_anchors():
    records = [dict(case_id=f"c{i}",period="all",horizon=h,rebalance=r,
                    annualized_return=.3-i*.01,max_drawdown=-.2,total_fees_cny=100.)
               for i,(h,r) in enumerate([(5,5),(5,5),(40,40),(30,30),(20,20)])]
    records.insert(0,dict(case_id="fails_drawdown",period="all",horizon=30,rebalance=15,
                          annualized_return=.9,max_drawdown=-.31,total_fees_cny=100.))
    assert FINE_TOP_PAIRS == 2
    assert fine_pairs(pd.DataFrame(records)) == {(5,5),(40,40),(10,10),(20,10)}
    assert fine_pairs(pd.DataFrame(records).iloc[:0]) == {(10,10),(20,10)}


def test_calendar_frequency_window_and_maturity():
    calendar = pd.bdate_range("2010-01-01","2026-10-13")
    assert [len(serving_periods(calendar,m)) for m in [12,6,3]] == [5,10,20]
    q2 = next(p for p in serving_periods(calendar,3) if p["period_start"]==pd.Timestamp("2026-04-01"))
    assert q2["cutoff"]==pd.Timestamp("2026-03-31")
    assert window_start(q2["period_start"],5)==pd.Timestamp("2021-04-01")
    index = pd.MultiIndex.from_product([["A"],pd.to_datetime(["2020-12-01","2020-12-02"])], names=["instrument","datetime"])
    frame = pd.DataFrame(dict(label=[.1,.2],target=[.1,.2],label_end_date=pd.to_datetime(["2020-12-31","2021-01-01"])),index=index)
    assert len(mature(frame,"2020-01-01","2020-12-31"))==1
    frame.loc[("A",pd.Timestamp("2020-12-02")),"label"] = 999
    assert len(mature(frame,"2020-01-01","2020-12-31"))==1


def test_fold_joint_selection_and_finite_three_folds():
    records=[]
    for p,scores in [("p00",[.2,0,0]),("p01",[.08,.08,.08])]:
        records.extend(dict(parameter_id=p,trees=20,num_leaves=31,validation_year=2019+i,mean_rank_ic=s) for i,s in enumerate(scores))
    assert choose_candidates(pd.DataFrame(records)).iloc[0].parameter_id=="p01"
    with pytest.raises(ValueError):
        choose_candidates(pd.DataFrame(records[:-1]))


def test_fee_floor_and_phase_deduplication():
    assert fee(1000,1,.002)==pytest.approx(7)
    assert fee(100000,0,.002)==pytest.approx(350)
    assert fee(0,1,.002)==0
    assert offsets(1)==[0]
    assert offsets(3)==[0,1,2]
    assert offsets(10)==[0,2,4,6,8]


def test_missing_scores_and_untradeable_names_cannot_disappear():
    score=pd.Series(dict(A=1.,B=2.,C=3.,D=4.))
    selected=select_holdings(score,["A","B","MISSING"],3,1,list(score.index),["A","B"])
    assert "MISSING" in selected
    assert len(set(["A","B","MISSING"])-set(selected))<=1
    selected=select_holdings(score,["A","B","MISSING"],3,1,list(score.index),["A","B","MISSING"])
    assert "MISSING" not in selected
    assert len(selected)==3


def test_final_rule_uses_all_phases_and_stress_not_best_phase():
    configs=[dict(candidate_id="a",rebalance=5,months=12),dict(candidate_id="b",rebalance=5,months=3)]
    records=[]
    for c in configs:
        for friction in [0.,.002]:
            for offset in offsets(5):
                records.append(dict(candidate_id=c["candidate_id"],period="all",friction=friction,
                    annualized_return=.2 if c["candidate_id"]=="a" else .19,
                    max_drawdown=-.31 if c["candidate_id"]=="a" and offset==4 else -.2,
                    average_daily_turnover=.1,total_fees_cny=1000))
    ranking,order=choose_final(configs,pd.DataFrame(records))
    assert order==["b"]
    assert not ranking.set_index("candidate_id").loc["a","passed"]


def fake_exchange():
    exchange = object.__new__(AuditedExchange)
    exchange.reset_audit(.002)
    exchange.get_quote_info = lambda *args: 10.
    exchange.get_factor = lambda *args: 1.
    exchange.trade_w_adj_price, exchange.trade_unit = False, 100
    return exchange


class FakePosition:
    def __init__(self, cash=10000., amount=1000.):
        self.cash, self.amount = cash, amount

    def check_stock(self, code):
        return True

    def get_stock_amount(self, code):
        return self.amount

    def get_cash(self):
        return self.cash


def test_t1_newly_bought_units_cannot_be_sold_that_day():
    exchange = fake_exchange()
    day = pd.Timestamp("2024-01-03")
    exchange.today_bought[(day,"A")] = 200
    order = Order("A",1000,Order.SELL,day,day)
    price, value, cost = exchange._calc_trade_info_by_order(order,FakePosition(),{})
    assert order.deal_amount==800
    assert cost==pytest.approx(fee(value,Order.SELL,.002))


def test_pressure_fee_affordability_and_raw_round_lots():
    exchange = fake_exchange()
    day = pd.Timestamp("2024-01-03")
    order = Order("A",10000,Order.BUY,day,day)
    price, value, cost = exchange._calc_trade_info_by_order(order,FakePosition(cash=10010),{})
    assert value+cost<=10010
    assert order.deal_amount%100==0
    assert order.deal_amount==900


def test_calendar_long_holiday_maps_first_execution_to_previous_session():
    calendar = pd.bdate_range("2010-01-01","2026-10-13")
    calendar = calendar[~calendar.isin(pd.date_range("2026-10-01","2026-10-07"))]
    q4 = serving_periods(calendar,3)[-1]
    assert q4["period_start"]==pd.Timestamp("2026-10-01")
    assert q4["execution_start"]==pd.Timestamp("2026-10-08")
    assert q4["cutoff"]==pd.Timestamp("2026-09-30")


def test_no_qualified_candidate_produces_a_readable_empty_selection():
    ranking, order = choose_final([],pd.DataFrame())
    assert order==[]
    assert list(ranking.columns)==["candidate_id","passed","median_cagr","worst_drawdown",
                                  "stress_median_cagr","median_turnover","median_fees","months","reason"]


def test_cached_outputs_are_verified_and_feature_order_changes_identity(tmp_path,monkeypatch):
    from paper_futu import final_ablation_core as core
    monkeypatch.setattr(core,"REPORT",tmp_path)
    store = object.__new__(core.Store)
    store.code,store.data = "code1","data1"
    config = dict(features=["a","b"],cutoff="2020-12-31")
    reversed_config = dict(features=["b","a"],cutoff="2020-12-31")
    assert store.key("fit",config)!=store.key("fit",reversed_config)
    output = tmp_path/"model.txt"
    output.write_text("original",encoding="utf-8")
    key = store.key("fit",config)
    core.save_json(tmp_path/"tasks"/f"{key}.json",dict(status="complete",result={"ok":True},
                   artifacts={str(output):core.digest(output)}))
    assert store.cached("fit",config)=={"ok":True}
    assert store.cached("fit",reversed_config) is None
    output.write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="artifact changed"):
        store.cached("fit",config)


def execution_exchange(top_k=1):
    import logging
    exchange = fake_exchange()
    exchange.top_k = top_k
    exchange.logger = logging.getLogger("test_final_ablation")
    exchange.get_close = lambda *args: 10.
    exchange.is_stock_tradable = lambda *args, **kwargs: True
    return exchange


def test_limit_directions_are_consistent_from_sizing_through_execution():
    exchange = execution_exchange()
    # At limit up A can be sold; at limit down B can be bought.
    exchange.is_stock_tradable = lambda stock, start, end, direction=None: (
        (stock == "A" and direction == Order.SELL)
        or (stock == "B" and direction == Order.BUY))
    day = pd.Timestamp("2024-01-03")
    previous = day-pd.Timedelta(days=1)
    position = Position(cash=100., position_dict={"A":dict(amount=1000.,price=10.)})
    orders = AuditedOrderGenerator().generate_order_list_from_target_weight_position(
        position, exchange, {"B":1.}, .95, previous, previous, day, day)
    assert [(o.stock_id,o.direction) for o in orders] == [("A",Order.SELL),("B",Order.BUY)]
    for order in orders:
        exchange.deal_order(order, position=position)
    assert position.get_stock_list() == ["B"]
    assert position.get_cash() >= 0
    assert [r["holdings_after"] for r in exchange.journal] == [0,1]
    assert all(r["raw_quantity"] % 100 == 0 for r in exchange.journal)
    cash = 100.+exchange.journal[0]["notional"]-exchange.journal[0]["fee"]
    cash -= exchange.journal[1]["notional"]+exchange.journal[1]["fee"]
    assert cash == pytest.approx(position.get_cash())
    # Opposite directions remain blocked in order generation.
    assert exchange.generate_order_for_target_amount_position(
        {"A":2000.}, {"A":1000.,"B":1000.}, day, day) == []


@pytest.mark.parametrize("failure", ["rejected", "t1_partial"])
def test_only_an_actual_full_sale_releases_a_new_name_slot(failure):
    exchange = execution_exchange()
    day = pd.Timestamp("2024-01-03")
    position = Position(cash=20000., position_dict={"A":dict(amount=1000.,price=10.)})
    orders = exchange.generate_order_for_target_amount_position({"B":1000.},{"A":1000.},day,day)
    # A rejection or partial T+1 sale happens after the plan was generated.
    if failure == "rejected":
        exchange.is_stock_tradable = lambda stock, *args, **kwargs: stock != "A"
    else:
        exchange.today_bought[(day,"A")] = 200.
    for order in orders:
        exchange.deal_order(order, position=position)
    assert position.get_stock_list() == ["A"]
    assert position.get_stock_amount("A") == (1000. if failure == "rejected" else 200.)
    assert exchange.journal[-1]["reason"] == "no_slot_after_actual_sells"
    assert exchange.journal[-1]["notional"] == 0
    assert position.get_cash() >= 0


def test_suspended_and_missing_score_holdings_keep_their_slots():
    exchange = execution_exchange(top_k=2)
    day = pd.Timestamp("2024-01-03")
    exchange.is_stock_tradable = lambda stock, *args, **kwargs: stock != "MISSING"
    selected = select_holdings(pd.Series(dict(A=1.,B=2.)),["A","MISSING"],2,1,["A","B"],["A"])
    assert set(selected) == {"B","MISSING"}
    position = Position(cash=100.,position_dict={
        "A":dict(amount=1000.,price=10.),"MISSING":dict(amount=1000.,price=10.)})
    previous = day-pd.Timedelta(days=1)
    orders = AuditedOrderGenerator().generate_order_list_from_target_weight_position(
        position,exchange,{s:.5 for s in selected},.95,previous,previous,day,day)
    for order in orders:
        exchange.deal_order(order,position=position)
    assert set(position.get_stock_list()) == {"B","MISSING"}
    assert position.get_stock_amount("MISSING") == 1000.


def test_preexisting_over_k_still_raises_with_account_diagnostics():
    exchange = execution_exchange()
    position = Position(cash=100.,position_dict={
        "A":dict(amount=1000.,price=10.),"B":dict(amount=1000.,price=10.)})
    with pytest.raises(ValueError,match=r"K=1, count=2, holdings=\['A', 'B'\]"):
        exchange.deal_order(Order("A",1000.,Order.SELL,pd.Timestamp("2024-01-03"),
                                  pd.Timestamp("2024-01-03")),position=position)


@pytest.mark.parametrize("amount,factor", [(1050.,1.),(420.,2.5)])
def test_full_liquidation_sells_the_odd_lot_and_frees_the_slot(amount,factor):
    exchange = execution_exchange()
    exchange.get_factor = lambda *args: factor
    exchange.trade_w_adj_price, exchange.trade_unit = False, 100
    day = pd.Timestamp("2024-01-03")
    position = Position(cash=20000.,position_dict={"A":dict(amount=amount,price=10.)})
    orders = exchange.generate_order_for_target_amount_position({"B":1000.},{"A":amount},day,day)
    assert orders[0].amount == amount
    for order in orders:
        exchange.deal_order(order,position=position)
    assert position.get_stock_list() == ["B"]
    assert exchange.journal[0]["raw_quantity"] == pytest.approx(1050.)
    assert exchange.journal[0]["holdings_after"] == 0
    assert exchange.journal[1]["raw_quantity"] % 100 == 0


@pytest.mark.parametrize("factor", [1.,2.5])
@pytest.mark.parametrize("friction", [0.,.001,.002])
@pytest.mark.parametrize("boundary_raw,expected_raw", [(100.,0.),(1300.,1200.)])
def test_real_qlib_rounding_cannot_cross_cash_and_fee_boundary(factor,friction,boundary_raw,expected_raw):
    exchange = execution_exchange()
    exchange.friction = friction
    # Fixed raw price; adjusted price and adjusted lot size both reflect the factor.
    exchange.get_quote_info = lambda *args: 10.*factor
    exchange.get_factor = lambda *args: factor
    boundary_value = boundary_raw*10.
    cash = boundary_value+fee(boundary_value,Order.BUY,friction)-.5
    position = Position(cash=cash,position_dict={})
    day = pd.Timestamp("2024-01-03")
    order = Order("A",boundary_raw*10./factor,Order.BUY,day,day)
    exchange.deal_order(order,position=position)
    assert order.deal_amount*factor == pytest.approx(expected_raw)
    assert position.get_cash() >= 0.
    assert exchange.journal[-1]["notional"]+exchange.journal[-1]["fee"] <= cash
    if factor == 1. and friction == 0. and boundary_raw == 100.:
        assert cash == 1004.5  # Previously Qlib's +0.1 made this balance -0.5.


@pytest.mark.parametrize("factor", [1.,2.5])
@pytest.mark.parametrize("friction", [0.,.001,.002])
@pytest.mark.parametrize("raw_quantity", [100.,1100.])
def test_affordable_orders_keep_their_quantity_and_real_fee_branch(factor,friction,raw_quantity):
    exchange = execution_exchange()
    exchange.friction = friction
    exchange.get_quote_info = lambda *args: 10.*factor
    exchange.get_factor = lambda *args: factor
    value = raw_quantity*10.
    expected_fee = (5. if raw_quantity == 100. else value*.0005)+friction*value
    cash = value+expected_fee
    position = Position(cash=cash,position_dict={})
    day = pd.Timestamp("2024-01-03")
    order = Order("A",raw_quantity/factor,Order.BUY,day,day)
    exchange.deal_order(order,position=position)
    assert order.deal_amount*factor == pytest.approx(raw_quantity)
    assert exchange.journal[-1]["fee"] == pytest.approx(expected_fee)
    assert position.get_cash() >= 0.
