# Roadmap Progress Snapshot

Last updated: 2026-03-19 (post-trained-model recovery)
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

Current local environment now has a runnable trained-model path through `logistic`, while `xgboost` remains unavailable.

### Trained Logistic Recent Backtest

- Average Return: `2.33%`
- Overall Win Rate: `27.24%`
- Average Drawdown: `3.02%`
- Trades: `53`

This view is still not the primary deployment metric, but it confirms the trained-model path is now active in local recent backtests.

### Portfolio-Level Variant Comparison

- `aggressive`: ROI `10.92%`, MaxDD `-14.62%`, Vol `34.43%`
- `balanced`: ROI `10.61%`, MaxDD `-14.96%`, Vol `34.83%`
- `quality`: ROI `3.21%`, MaxDD `-12.67%`, Vol `25.59%`
- `elite`: ROI `1.82%`, MaxDD `-13.80%`, Vol `24.81%`
- `defensive`: ROI `1.14%`, MaxDD `-11.41%`, Vol `21.51%`
- `risk_adjusted`: ROI `1.13%`, MaxDD `-11.28%`, Vol `21.37%`
- `trained_selective`: ROI `5.42%`, MaxDD `-11.46%`, Vol `24.80%`

Current recent-backtest preferred variant (score-based selection): `aggressive`

## Important Caveats

1. `xgboost` is still unavailable in the local runtime.
2. The trained-model path is now working through `logistic`, but its portfolio variants currently carry much higher drawdown and volatility than the older fallback baseline.
3. The new optimization target is no longer "make trained models runnable"; it is now "make trained-model portfolio construction competitive on risk-adjusted terms."

### Trained-Model Path Status

- `train_and_backtest.py` now runs the benchmark suite and attempts to persist the selected champion model.
- The benchmark training path is runnable in the current environment.
- Champion selection now ignores zero-result models.
- Current local outcome:
  - logistic becomes the valid champion
  - persistence succeeds with `{'saved': True, 'model_name': 'logistic'}`
- The final all-data XGBoost save step now degrades gracefully instead of crashing.
- This means the repository can now complete benchmark orchestration and activate a trained local model path, even though the intended XGBoost path is still blocked by missing dependency support.

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
