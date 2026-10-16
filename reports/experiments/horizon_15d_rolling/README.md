# 15-day return-horizon experiment

## Purpose

This experiment tests whether a longer prediction target is more suitable than
the original 5-day target.  It keeps the feature set, LightGBM configuration,
portfolio construction, transaction costs, and execution convention unchanged.

## Target and timing

- Signal date: after market close on day `T`.
- Raw evaluation label: `T+1` open to `T+15` close return.
- Model target: the daily cross-sectional rank of that raw return, centered to
  approximately `[-0.5, 0.5]`.
- Execution: rebalance at the next tradable open; no same-day round trip.

## Temporal validation

The model is retrained annually:

| Fold | Train | Validation | Test |
|---|---|---|---|
| 1 | 2010-2020 | 2021 | 2022 |
| 2 | 2011-2021 | 2022 | 2023 |
| 3 | 2012-2022 | 2023 | 2024 |

Fifteen signal days are purged from the end of each train/validation period so
that their future-return labels cannot cross into the following period.

## Validation Rank IC

| Validation year | Mean Rank IC |
|---|---:|
| 2021 | 2.91% |
| 2022 | 10.57% |
| 2023 | 8.11% |

## Backtest comparison (2022-2024)

| Rebalance | Annual return | Annual excess return | Sharpe | Information ratio | Max drawdown | Average daily turnover |
|---:|---:|---:|---:|---:|---:|---:|
| 5 days | 11.72% | 20.72% | 0.735 | 2.312 | -18.42% | 7.71% |
| 10 days | 8.28% | 17.01% | 0.551 | 1.969 | -20.68% | 4.42% |
| 15 days | 5.20% | 13.68% | 0.379 | 1.615 | -18.98% | 3.28% |

The 15-day target performs best with a 5-day refresh in this historical sample.
The target horizon and rebalance interval are different concepts: the target
defines what the model learns, while the rebalance interval determines how often
new information replaces the portfolio.  These 2022-2024 results have now been
observed and therefore form a retrospective comparison, not a fresh final test.
