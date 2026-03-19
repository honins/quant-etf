import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.backtester import Backtester
from src.features.regime_features import attach_regime_features, build_regime_feature_frame
from src.models.model_registry import ModelRegistry
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.replay import build_portfolio_backtest_report, replay_portfolio_allocations
from src.research.experiment_runner import ExperimentRunner
from src.research.labeler import MultiTaskLabeler
from src.research.validation import (
    attach_regime_label,
    generate_purged_walk_forward_windows,
    generate_walk_forward_windows,
    split_windows_by_regime,
)


class DummyModel:
    def __init__(self):
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        self.is_trained = not df.empty

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.65, dtype=float)

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"feature": "bias_5", "importance": 1.0},
            {"feature": "rsi_14", "importance": 0.5},
        ])


def _make_frame(code: str, start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="D")
    base = np.linspace(100, 130, periods)
    return pd.DataFrame(
        {
            "ts_code": code,
            "trade_date": dates.strftime("%Y%m%d"),
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.3,
            "vol": np.linspace(1000, 2000, periods),
            "atr": np.full(periods, 1.2),
            "bias_5": np.linspace(0.01, 0.03, periods),
            "bias_20": np.linspace(0.01, 0.03, periods),
            "bias_60": np.linspace(0.01, 0.03, periods),
            "rsi_6": np.linspace(40, 60, periods),
            "rsi_14": np.linspace(45, 62, periods),
            "atr_pct": np.full(periods, 0.01),
            "vol_ratio": np.full(periods, 1.1),
            "vol_ratio_20": np.full(periods, 1.05),
            "macd_norm": np.full(periods, 0.01),
            "macdsignal_norm": np.full(periods, 0.009),
            "macdhist_norm": np.full(periods, 0.001),
            "bb_pos": np.full(periods, 0.6),
            "rs_20d": np.full(periods, 0.02),
            "rel_vol": np.full(periods, 1.0),
            "ret_5": np.full(periods, 0.01),
            "ret_10": np.full(periods, 0.02),
            "ret_20": np.full(periods, 0.03),
            "trend_gap": np.full(periods, 0.01),
            "ma20_slope_5": np.full(periods, 0.01),
            "ma60_slope_10": np.full(periods, 0.01),
            "breakout_20": np.full(periods, 0.01),
            "drawdown_20": np.full(periods, -0.01),
            "drawdown_60": np.full(periods, -0.02),
            "rebound_20": np.full(periods, 0.02),
            "rsi_spread": np.full(periods, 0.05),
            "close_to_ma20": np.full(periods, 0.01),
            "close_to_ma60": np.full(periods, 0.02),
            "intraday_range": np.full(periods, 0.02),
        }
    )


class ResearchWorkflowTest(unittest.TestCase):
    def test_multitask_labeler_adds_expected_columns(self):
        frame = _make_frame("510300.SH", "2024-01-01", 40)
        labeled = MultiTaskLabeler().add_labels(frame)
        self.assertIn("future_ret_1d", labeled.columns)
        self.assertIn("future_ret_20d", labeled.columns)
        self.assertIn("best_holding_days", labeled.columns)
        self.assertIn("meta_target", labeled.columns)
        self.assertIn("target", labeled.columns)

    def test_experiment_runner_writes_artifacts(self):
        dataset = {
            "510300.SH": MultiTaskLabeler().add_labels(_make_frame("510300.SH", "2023-01-01", 120)).dropna().reset_index(drop=True),
            "159915.SZ": MultiTaskLabeler().add_labels(_make_frame("159915.SZ", "2023-01-01", 120)).dropna().reset_index(drop=True),
        }
        windows = generate_walk_forward_windows(
            earliest_date="20230125",
            latest_date="20230430",
            train_window_days=40,
            test_window_days=20,
            step_days=20,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(output_root=Path(tmpdir))
            result = runner.run_walk_forward(
                dataset=dataset,
                windows=windows,
                model_factory=DummyModel,
                backtester=Backtester(),
                threshold=0.6,
                ticker_names={"510300.SH": "沪深300ETF", "159915.SZ": "创业板ETF"},
                run_name="unit_test",
            )
            output_dir = Path(result["output_dir"])
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "trades.csv").exists())
            self.assertTrue((output_dir / "equity_curve.csv").exists())
            self.assertTrue((output_dir / "feature_importance.csv").exists())
            self.assertTrue((output_dir / "feature_coverage.json").exists())
            self.assertTrue((output_dir / "leakage_check.md").exists())

    def test_portfolio_optimizer_returns_weighted_selection(self):
        candidates = pd.DataFrame(
            [
                {"code": "510300.SH", "expected_return": 0.08, "downside_risk": 0.03, "confidence": 0.80, "avg_correlation": 0.25, "liquidity": 0.90},
                {"code": "159915.SZ", "expected_return": 0.10, "downside_risk": 0.05, "confidence": 0.75, "avg_correlation": 0.35, "liquidity": 0.70},
                {"code": "512480.SH", "expected_return": 0.06, "downside_risk": 0.02, "confidence": 0.65, "avg_correlation": 0.15, "liquidity": 0.60},
            ]
        )
        optimizer = PortfolioOptimizer()
        return_matrix = pd.DataFrame(
            {
                "510300.SH": [0.01, 0.00, 0.02, -0.01],
                "159915.SZ": [0.02, -0.01, 0.03, -0.02],
                "512480.SH": [0.005, 0.004, 0.006, 0.003],
            }
        )
        result = optimizer.optimize(
            candidates,
            previous_weights={"510300.SH": 0.2},
            return_matrix=return_matrix,
            trading_costs={"510300.SH": 0.002, "159915.SZ": 0.004, "512480.SH": 0.001},
        )
        self.assertTrue(result["weights"])
        self.assertLessEqual(sum(result["weights"].values()), 1.000001)
        self.assertGreater(result["expected_turnover"], 0.0)
        self.assertTrue(result["top_k"])
        self.assertIn("avg_correlation", result["diagnostics"])

    def test_regime_validation_helpers_group_windows(self):
        windows = generate_purged_walk_forward_windows(
            earliest_date="20230101",
            latest_date="20230430",
            train_window_days=30,
            test_window_days=15,
            step_days=15,
            purge_days=3,
        )
        market_status_map = {window.test_end: "Bull Market" if idx % 2 == 0 else "Volatile Market" for idx, window in enumerate(windows)}
        grouped = split_windows_by_regime(windows, market_status_map)
        self.assertTrue(grouped["Bull Market"])
        self.assertTrue(grouped["Volatile Market"])

        sample = pd.DataFrame({"trade_date": [window.test_end for window in windows]})
        labeled = attach_regime_label(sample, market_status_map)
        self.assertIn("regime", labeled.columns)

    def test_regime_features_attach_market_context(self):
        index_df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=80, freq="D").strftime("%Y%m%d"),
                "close": np.linspace(100, 120, 80),
                "ma20": np.linspace(99, 118, 80),
                "ma60": np.linspace(97, 112, 80),
                "vol": np.linspace(1000, 1400, 80),
            }
        )
        regime_frame = build_regime_feature_frame(index_df)
        asset_df = pd.DataFrame(
            {
                "trade_date": index_df["trade_date"].tail(20).tolist(),
                "close": np.linspace(50, 55, 20),
            }
        )
        merged = attach_regime_features(asset_df, regime_frame)
        self.assertIn("market_trend_strength", merged.columns)
        self.assertIn("regime_label", merged.columns)

    def test_model_registry_contains_multiple_models(self):
        registry = ModelRegistry()
        model_names = registry.list_models()
        self.assertIn("xgboost", model_names)
        self.assertIn("logistic", model_names)
        self.assertIn("rules", model_names)

    def test_benchmark_suite_writes_comparison_artifacts(self):
        dataset = {
            "510300.SH": MultiTaskLabeler().add_labels(_make_frame("510300.SH", "2023-01-01", 120)).dropna().reset_index(drop=True),
            "159915.SZ": MultiTaskLabeler().add_labels(_make_frame("159915.SZ", "2023-01-01", 120)).dropna().reset_index(drop=True),
        }
        windows = generate_walk_forward_windows(
            earliest_date="20230125",
            latest_date="20230430",
            train_window_days=40,
            test_window_days=20,
            step_days=20,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(output_root=Path(tmpdir))
            result = runner.run_benchmark_suite(
                dataset=dataset,
                model_specs=[("dummy_a", DummyModel), ("dummy_b", DummyModel)],
                validation_specs=[("rolling", windows)],
                backtester=Backtester(),
                ticker_names={"510300.SH": "沪深300ETF", "159915.SZ": "创业板ETF"},
                suite_name="unit_suite",
            )
            suite_dir = Path(result["suite_dir"])
            self.assertTrue((suite_dir / "benchmark_summary.csv").exists())
            self.assertTrue((suite_dir / "benchmark_summary.json").exists())
            self.assertTrue((suite_dir / "benchmark_summary.md").exists())
            self.assertTrue((suite_dir / "benchmark_selection.json").exists())
            self.assertTrue((suite_dir / "benchmark_history.json").exists())
            self.assertIn("champion", result["selection"])

    def test_portfolio_replay_builds_equity_series(self):
        plan = {"weights": {"510300.SH": 0.6, "159915.SZ": 0.4}}
        datasets = {
            "510300.SH": _make_frame("510300.SH", "2024-01-01", 20),
            "159915.SZ": _make_frame("159915.SZ", "2024-01-01", 20),
        }
        replay = replay_portfolio_allocations(plan, datasets, lookback_days=10)
        self.assertTrue(replay["series"])
        self.assertIn("total_return", replay["summary"])
        report = build_portfolio_backtest_report(replay)
        self.assertIn("max_drawdown", report)
        self.assertIn("ending_equity", report)


if __name__ == "__main__":
    unittest.main()
