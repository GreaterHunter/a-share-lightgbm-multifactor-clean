import numpy as np
import pandas as pd

from examples.offline_demo import make_panel, run_demo
from src.forward_labels import select_mature_period


def test_demo_uses_future_opens_and_keeps_immature_features():
    panel, calendar = make_panel()
    stock = panel.xs("SYNTH000")
    np.testing.assert_allclose(stock.raw_label.iloc[25], stock.open.iloc[36] / stock.open.iloc[26] - 1)
    assert stock.raw_label.tail(11).isna().all()
    assert stock.mom_5d.tail(11).notna().all()
    train = select_mature_period(panel, str(calendar[0]), str(calendar[119]))
    assert train.label_end_date.max() <= calendar[119]
    assert train.index.get_level_values("datetime").max() == calendar[108]
    assert not panel.index.has_duplicates


def test_demo_is_deterministic_and_explicitly_synthetic():
    first, _ = make_panel()
    second, _ = make_panel()
    pd.testing.assert_frame_equal(first, second)
    result = run_demo()
    assert result["max_train_label_end"] <= result["train_cutoff"]
    assert result["train_rows"] > 0 and result["valid_ic_days"] > 0
    assert "SYNTHETIC" in result["data"]
    assert -1 <= result["mean_rank_ic"] <= 1
