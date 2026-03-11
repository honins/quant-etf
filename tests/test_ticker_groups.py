import unittest

from config import tickers


class TickerGroupsTest(unittest.TestCase):
    def test_tradable_and_observe_are_disjoint(self):
        self.assertTrue(set(tickers.get_tradable_ticker_list()).isdisjoint(tickers.get_observe_ticker_list()))

    def test_all_tickers_are_classified(self):
        for code in tickers.TICKERS:
            self.assertIn(tickers.get_ticker_category(code), {"core", "satellite", "observe"})

    def test_default_ticker_list_includes_observe_and_tradable_list_does_not(self):
        all_codes = set(tickers.get_ticker_list())
        tradable_codes = set(tickers.get_tradable_ticker_list())
        observe_codes = set(tickers.get_observe_ticker_list())

        self.assertTrue(observe_codes.issubset(all_codes))
        self.assertTrue(observe_codes.isdisjoint(tradable_codes))
        self.assertEqual(all_codes, tradable_codes | observe_codes)

    def test_duplicate_aliases_are_not_in_active_lists(self):
        active_codes = set(tickers.get_ticker_list())
        tradable_codes = set(tickers.get_tradable_ticker_list())
        for alias in tickers.DUPLICATE_TICKER_ALIASES:
            self.assertNotIn(alias, active_codes)
            self.assertNotIn(alias, tradable_codes)

    def test_duplicate_aliases_resolve_to_representatives(self):
        for alias, representative in tickers.DUPLICATE_TICKER_ALIASES.items():
            self.assertEqual(tickers.normalize_ticker(alias), representative)
            self.assertEqual(tickers.get_ticker_category(alias), tickers.get_ticker_category(representative))
            self.assertIn(alias, tickers.TICKERS)
            self.assertIn(representative, tickers.TICKERS)


if __name__ == "__main__":
    unittest.main()
