"""Portfolio construction strategies used by the robustness experiments."""

from __future__ import annotations

import sys
import types

import pandas as pd


def load_weight_strategy_base():
    """Load Qlib's weight strategy without importing its unused optimizer stack."""
    module_name = "qlib.contrib.strategy.optimizer"
    if module_name not in sys.modules:
        optimizer_stub = types.ModuleType(module_name)

        class EnhancedIndexingOptimizer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("This project does not use EnhancedIndexingOptimizer")

        optimizer_stub.EnhancedIndexingOptimizer = EnhancedIndexingOptimizer
        sys.modules[module_name] = optimizer_stub

    from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

    return WeightStrategyBase


WeightStrategyBase = load_weight_strategy_base()


class TieredTopKWeightStrategy(WeightStrategyBase):
    """Hold Top K stocks and give a fixed larger allocation to the top tier.

    The strategy creates target portfolio weights. Qlib's order generator then
    trades only the difference between current and target positions.
    """

    def __init__(
        self,
        *,
        top_k: int = 50,
        n_drop: int = 5,
        top_tier_size: int = 10,
        top_tier_weight: float = 0.40,
        only_tradable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 < n_drop <= top_k:
            raise ValueError("n_drop must be in [1, top_k]")
        if not 0 < top_tier_size <= top_k:
            raise ValueError("top_tier_size must be in [1, top_k]")
        if not 0 < top_tier_weight < 1:
            raise ValueError("top_tier_weight must be between 0 and 1")

        self.top_k = top_k
        self.n_drop = n_drop
        self.top_tier_size = top_tier_size
        self.top_tier_weight = top_tier_weight
        self.only_tradable = only_tradable

    def generate_target_weight_position(
        self,
        score: pd.Series | pd.DataFrame,
        current,
        trade_start_time,
        trade_end_time,
    ) -> dict[str, float]:
        """Convert the latest prediction ranking into normalized target weights."""
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]

        score = score.dropna().sort_values(ascending=False)
        if self.only_tradable:
            tradable = [
                stock
                for stock in score.index
                if self.trade_exchange.is_stock_tradable(
                    stock_id=stock,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            ]
            buyable_score = score.reindex(tradable)
        else:
            buyable_score = score

        current_stocks = pd.Index(current.get_stock_list())
        current_ranked = (
            score.reindex(current_stocks)
            .dropna()
            .sort_values(ascending=False)
        )
        new_candidates = buyable_score.loc[
            ~buyable_score.index.isin(current_stocks)
        ]

        candidate_count = self.n_drop + max(
            self.top_k - len(current_stocks),
            0,
        )
        proposed_buys = new_candidates.head(candidate_count).index
        combined = (
            score.reindex(current_ranked.index.union(proposed_buys))
            .dropna()
            .sort_values(ascending=False)
        )
        sell_candidates = combined.tail(self.n_drop).index
        sells = current_stocks[current_stocks.isin(sell_candidates)]

        survivors = current_stocks[~current_stocks.isin(sells)]
        buy_count = len(sells) + max(self.top_k - len(current_stocks), 0)
        buys = proposed_buys[:buy_count]
        selected_index = survivors.append(pd.Index(buys))

        # Defensive fill for missing scores or temporarily undersized positions.
        if len(selected_index) < self.top_k:
            fill = buyable_score.loc[
                ~buyable_score.index.isin(selected_index)
            ].head(self.top_k - len(selected_index)).index
            selected_index = selected_index.append(fill)

        selected = list(
            score.reindex(selected_index)
            .dropna()
            .sort_values(ascending=False)
            .head(self.top_k)
            .index
        )
        if not selected:
            return {}

        top_count = min(self.top_tier_size, len(selected))
        lower_count = len(selected) - top_count

        if lower_count == 0:
            equal_weight = 1.0 / top_count
            return {stock: equal_weight for stock in selected}

        top_weight = self.top_tier_weight / top_count
        lower_weight = (1.0 - self.top_tier_weight) / lower_count

        target_weights = {
            stock: top_weight
            for stock in selected[:top_count]
        }
        target_weights.update(
            {
                stock: lower_weight
                for stock in selected[top_count:]
            }
        )
        return target_weights
