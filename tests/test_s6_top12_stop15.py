import pytest

from paper_futu import research_s6_stop_loss as old
from paper_futu.research_s6_top12_stop15 import output_dir, strategy_parameters


@pytest.mark.parametrize("weight", [.4, .5, .6])
def test_only_requested_portfolio_and_stop_settings_change(weight):
    baseline = strategy_parameters(weight, "top12")
    stop = strategy_parameters(weight, "top12_stop15")
    assert baseline["threshold"] is None
    assert stop["threshold"] == .15
    assert {k:v for k,v in baseline.items() if k != "threshold"} == {
        k:v for k,v in stop.items() if k != "threshold"}
    assert baseline["top_k"] == 12
    assert baseline["top_tier_size"] == 6
    assert baseline["top_tier_weight"] == weight
    assert baseline["risk_degree"] == .95
    assert baseline["n_drop"] == 6
    assert output_dir(weight) not in (old.OUT, old.SOURCE)


def test_weight_choices_cannot_collide_or_overwrite_prior_experiment():
    assert len({output_dir(w) for w in (.4,.5,.6)}) == 3
    with pytest.raises(ValueError): output_dir(.41)


def test_fifteen_percent_stop_includes_buy_fee_and_is_not_twenty_percent():
    book = old.CostBook()
    book.fill("X", 1, 100, 1000, 5)
    loss = book.return_at("X", 8.5425)
    assert loss == pytest.approx(-.15)
    assert loss <= -strategy_parameters(.4,"top12_stop15")["threshold"]+1e-12
    assert loss > -.20
