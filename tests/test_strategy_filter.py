import unittest

import pandas as pd

from config.settings import settings
from src.strategy.logic import StrategyFilter


def _index_df_for_bull() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "20260307", "close": 100.0, "ma20": 99.0, "ma60": 98.0},
            {"trade_date": "20260310", "close": 101.0, "ma20": 99.5, "ma60": 98.5},
            {"trade_date": "20260311", "close": 102.0, "ma20": 100.0, "ma60": 99.0},
        ]
    )


def _index_df_for_bear() -> pd.DataFrame:
    confirm_days = settings.MARKET_STATE_CONFIRM_DAYS
    rows = []
    for i in range(confirm_days):
        rows.append(
            {
                "trade_date": f"202603{10 + i:02d}",
                "close": 95.0 - i,
                "ma20": 98.0,
                "ma60": 99.0,
            }
        )
    return pd.DataFrame(rows)


def _index_df_for_volatile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "20260309", "close": 100.0, "ma20": 100.0, "ma60": 99.0},
            {"trade_date": "20260310", "close": 99.8, "ma20": 99.7, "ma60": 99.6},
            {"trade_date": "20260311", "close": 99.7, "ma20": 99.5, "ma60": 99.6},
        ]
    )


class StrategyFilterRegressionTest(unittest.TestCase):
    def setUp(self):
        self.filter = StrategyFilter()

    def test_bull_market_uses_base_threshold_for_normal_ticker(self):
        code = "510300.SH"
        below = settings.BULL_BASE_THRESHOLD - 0.01
        at = settings.BULL_BASE_THRESHOLD

        self.assertEqual(
            self.filter.filter_signal(below, _index_df_for_bull(), code=code),
            (False, "Bull Market"),
        )
        self.assertEqual(
            self.filter.filter_signal(at, _index_df_for_bull(), code=code),
            (True, "Bull Market"),
        )

    def test_bull_market_uses_aggressive_threshold_for_aggressive_ticker(self):
        code = next(code for code in settings.AGGRESSIVE_TICKERS if code not in settings.TICKER_BULL_THRESHOLDS)
        below = settings.BULL_AGGRESSIVE_THRESHOLD - 0.01
        at = settings.BULL_AGGRESSIVE_THRESHOLD

        self.assertEqual(
            self.filter.filter_signal(below, _index_df_for_bull(), code=code),
            (False, "Bull Market (Aggressive)"),
        )
        self.assertEqual(
            self.filter.filter_signal(at, _index_df_for_bull(), code=code),
            (True, "Bull Market (Aggressive)"),
        )

    def test_bull_market_prefers_ticker_specific_override_over_aggressive_threshold(self):
        code = next(iter(settings.TICKER_BULL_THRESHOLDS))
        threshold = settings.TICKER_BULL_THRESHOLDS[code]

        self.assertEqual(
            self.filter.filter_signal(threshold - 0.01, _index_df_for_bull(), code=code),
            (False, "Bull Market (Aggressive)"),
        )
        self.assertEqual(
            self.filter.filter_signal(threshold, _index_df_for_bull(), code=code),
            (True, "Bull Market (Aggressive)"),
        )

    def test_bull_market_respects_dynamic_threshold_when_provided(self):
        code = "510300.SH"
        dynamic_threshold = 0.68

        self.assertEqual(
            self.filter.filter_signal(0.67, _index_df_for_bull(), code=code, dynamic_threshold=dynamic_threshold),
            (False, "Bull Market"),
        )
        self.assertEqual(
            self.filter.filter_signal(0.68, _index_df_for_bull(), code=code, dynamic_threshold=dynamic_threshold),
            (True, "Bull Market"),
        )

    def test_bear_market_uses_bear_threshold(self):
        below = settings.BEAR_THRESHOLD - 0.01
        at = settings.BEAR_THRESHOLD

        self.assertEqual(
            self.filter.filter_signal(below, _index_df_for_bear(), code="510300.SH"),
            (False, "Bear Market"),
        )
        self.assertEqual(
            self.filter.filter_signal(at, _index_df_for_bear(), code="510300.SH"),
            (True, "Bear Market"),
        )

    def test_volatile_market_uses_volatile_threshold(self):
        below = settings.VOLATILE_THRESHOLD - 0.01
        at = settings.VOLATILE_THRESHOLD

        self.assertEqual(
            self.filter.filter_signal(below, _index_df_for_volatile(), code="510300.SH"),
            (False, "Volatile Market"),
        )
        self.assertEqual(
            self.filter.filter_signal(at, _index_df_for_volatile(), code="510300.SH"),
            (True, "Volatile Market"),
        )


if __name__ == "__main__":
    unittest.main()
