# Four-strategy overfitting and style-dependence study

This study diagnoses S2/S4/S5/S6 on the immutable 2026-10-13 snapshot. It does not select a new champion or change the Futu deployment.

Run from the project root with the `quant-lightgbm` Python interpreter:

```powershell
python -m paper_futu.research_overfit_four --stage all
python -m paper_futu.research_overfit_four --stage status
```

Separate resumable stages: `audit`, `train`, `backtest`, `diagnose`, `report`. Training requires a successful audit; subsequent stages require the preceding stage's completed artifacts. A controller lock prevents overlapping runs. Ordinary errors stop scheduling new jobs and preserve completed cache entries. One confirmed memory-error retry is permitted with serial execution. No process is forcibly killed by the controller.

For the already-audited study, the scheduling-only continuation below promptly fills up to two free worker slots. Do not launch it alongside another controller:

```powershell
python -m scripts.resume_overfit_four_parallel
```

It uses the original worker and scientific task keys, verifies cached artifacts, and preserves the scientific implementation binding. Its own SHA is recorded separately in `parallel_scheduler_binding.json`, runtime history and the final report manifest. Memory failures are recognized from UTF-8 task records even if Windows stderr uses a different code page; a persistent `memory_retry_budget.json` permits only one serial retry of the affected task, including across restarts. After prediction assembly, the controller returns unused Arrow buffers before applying the launch gate; it does not close other applications or lower the threshold. Each stage may fall back to one worker when actual overlapping memory use triggers protection; a two-worker limit is not a promise that available RAM can sustain two workers.

Outputs are in `reports/experiments/overfit_four_asof_20261013`; new models are in the same-named folder under `models/experiments`. The frozen original experiment is read-only. Before the first fit, audit independently recomputes opening-to-opening labels and rank targets, verifies original model reload predictions and the 20 source accounts, binds strategy IDs to full configurations, and hashes the snapshot and protected files.

The two source prediction families are raw15/H10/fixed and rank15/H20/annual-tuned. V0 preserves the original stream. V1/V2/V3 use rolling 3/5/10-year windows, V4 uses normalized sample weights with a 756-session half-life, V5 raises L1 to `max(1,10*original)`, V6 multiplies L2 by five, V7 combines those two changes, and V8 halves leaves and triples minimum leaf samples. Other annual parameters remain the originals. Each new variant reselects tree count using the preceding three mature annual validation folds; no new hyperparameter grid is searched.

The primary seed is 42. V0/V2/V7 also use 2024/2025 for final fits only, retaining the seed-42 tree selections. Source forecasts are stitched without changing their stock/date sample. S2/S6 share H10 forecasts; S4/S5 share H20 forecasts.

The exact scope is 18 main streams plus 12 seed-check streams; 180 main, 120 seed and 40 V0 friction accounts, including 20 existing base accounts. Independent validation jobs are shared across execution years whenever their scientific configuration matches. The hard limit is 380 fits. At most two workers, each with four threads, run concurrently. Following the user's explicit 2026-10-15 update, workers require 3.0 GiB free before each launch (previously 3.6 GiB), while the 2 GiB per-task check remains unchanged. Account batches contain at most ten replays. A continuous 30-minute memory shortage stops the controller and requires an explicit resume after memory is available.

The final report and machine-readable outputs include:

- `research_report.html`: fixed V0–V8 order, four strategy tables and methodological limits.
- `strategy_summary.csv`, `paired_strategy_differences.csv`: annual/full-period, seed, friction, five-phase return/risk/turnover/cost/holdings and style summaries.
- `signal_ic_summary.csv`, `generalization.csv`: yearly/quarterly IC, prior validation folds, in-sample last-252-session IC, tree counts, prediction-rank agreement and TopK overlap.
- `account_metrics_exposure.csv` and per-job stock contribution files: individual phases, bank weights, style percentiles, account profit attribution and missing-style coverage.
- `paired_block_bootstrap.csv`: 2,000 paired moving-block draws, block lengths 60/120, fixed seed 20261015, for variant-minus-baseline IC and mean-phase CAGR.
- `protocol.json`, `report_manifest.json`, `completion.json`: scientific identities, hashes, source protections and acceptance completion.

Report figures are descriptive on a repeatedly used historical interval, not new independent tests. Residual rank correlation controls only the bank indicator and five predeclared styles; it is not full industry neutrality or proof of tradable alpha. Removing years is a discontinuous dependence diagnostic. Bootstrap intervals do not undo strategy-selection bias. The daily replay uses ideal opens, a uniform 9.5% limit filter and the original corporate-action approximation, not verified 09:31 fills or full point-in-time financial revisions.
