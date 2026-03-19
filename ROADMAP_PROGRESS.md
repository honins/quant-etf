# Roadmap Progress Snapshot

Last updated: 2026-03-19
Branch: `feature/research-roadmap`

## Current Status

This branch has moved beyond the original single-model daily signal prototype and now has a working research-platform baseline with a fallback portfolio construction path that can be backtested end-to-end.

The main limitation is still environment and dependency availability: the current recent-window backtest is running through the `Rules` fallback path because `xgboost` is unavailable in the local runtime. The fallback path is no longer a trivial rule engine; it now includes regime features, cross-sectional filtering, and portfolio variants.

## Implemented Roadmap Work

### Research Layer

- Added `src/research/experiment_runner.py`
- Added `src/research/metrics.py`
- Added `src/research/validation.py`
- Added `src/research/labeler.py`
- Added unified experiment output directories under `reports/experiments/`
- Added benchmark suite outputs, champion/challenger selection, and benchmark history artifacts

### Feature Layer

- Split technical feature logic toward roadmap structure:
  - `src/features/price_features.py`
  - `src/features/cross_section_features.py`
  - `src/features/regime_features.py`
- Kept `src/features/technical.py` as the compatibility entrypoint
- Added regime feature injection to both training and recent-backtest/live scoring paths

### Label / Validation Layer

- Added multi-task label support:
  - `future_ret_1d/3d/5d/10d/20d`
  - `best_holding_days`
  - `hit_take_profit_first`
  - `hit_stop_loss_first`
  - `meta_target`
- Preserved old baseline labels for comparison
- Added rolling / anchored / purged validation support
- Added regime split utilities
- Added coverage and leakage diagnostics artifacts

### Backtest / Portfolio Layer

- Standardized `src/backtest/backtester.py` output structure
- Added `daily_pnl`, `position_history`, `annual_return`, `calmar`, `turnover`, `avg_holding_days`
- Added `src/portfolio/optimizer.py`
- Added `src/portfolio/replay.py`
- Added allocation replay and portfolio-level backtest summary output
- Added fallback portfolio variants:
  - `aggressive`
  - `balanced`
  - `quality`
  - `elite`

### Model Layer

- Upgraded `src/models/xgb_model.py` into a safer baseline with dependency-aware loading
- Added `src/models/logistic_model.py`
- Added `src/models/model_registry.py`
- Added benchmark suite orchestration over multiple models and validation modes
- Added champion persistence flow in `train_and_backtest.py` for the best available trained model

### Dashboard / Output Layer

- Dashboard payload now carries portfolio-level backtest summaries
- Added benchmark selection/history payloads
- Added portfolio replay metrics and portfolio variant comparison display support

## Current Best Fallback Results

Recent backtest command:

```bash
python backtest_recent.py
```

Current local environment result is using `Rules` fallback because `xgboost` is unavailable.

### Single-ETF Aggregate View

- Average Return: `-0.37%`
- Overall Win Rate: `16.32%`
- Average Drawdown: `1.13%`
- Trades: `26`

This view is no longer considered the primary decision metric.

### Portfolio-Level Variant Comparison

- `aggressive`: ROI `5.93%`, MaxDD `-2.27%`, Vol `9.72%`
- `balanced`: ROI `6.02%`, MaxDD `-2.37%`, Vol `9.88%`
- `quality`: ROI `7.38%`, MaxDD `-2.58%`, Vol `10.61%`
- `elite`: ROI `8.46%`, MaxDD `-2.80%`, Vol `10.98%`

Current default fallback preference: `elite`

## Important Caveats

1. The current positive portfolio ROI is from the fallback portfolio construction path, not from the intended champion model path.
2. `xgboost` is unavailable in the local runtime, so the recent backtest is not yet validating the benchmark-selected trained model.
3. The fallback path is now useful as a resilient baseline, but it should not be treated as the final target architecture.

### Trained-Model Path Status

- `train_and_backtest.py` now runs the benchmark suite and attempts to persist the selected champion model.
- The benchmark training path is runnable in the current environment.
- Current blocker: `xgboost` dependency is unavailable locally, so champion persistence falls back with:
  - `{'saved': False, 'reason': 'xgboost training failed'}`
- The final all-data XGBoost save step now degrades gracefully instead of crashing.
- This means the repository can complete the benchmark orchestration flow, but cannot yet activate the intended trained XGBoost path until the local runtime has the dependency.

## What Is Stable Right Now

- Research / experiment directory structure
- Multi-model registry and benchmark outputs
- Rolling / anchored / purged validation framework
- Regime and cross-sectional feature pipeline
- Portfolio replay and portfolio-level ROI reporting
- Fallback portfolio variants with recent-window comparison

## Highest-Value Next Steps

1. Restore the trained model execution path in the local environment so benchmark-selected models can run in recent backtests.
2. Make dashboard fully consume portfolio variants and portfolio-level summaries as first-class outputs.
3. Continue replacing fallback heuristics with champion-model-driven portfolio construction.
