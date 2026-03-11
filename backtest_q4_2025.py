from src.backtest.backtester import Backtester
from src.backtest.hybrid_runner import (
    build_data_cache,
    prepare_index_data,
    run_backtest_for_cache,
    summarize_results,
)
from src.data_loader.data_manager import DataManager
from src.data_loader.tushare_loader import TushareLoader
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter
from config import tickers


def main():
    start_date = "20251001"
    end_date = "20251231"

    print(f"📅 Starting Backtest for Q4 2025 ({start_date} - {end_date})...")

    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    backtester = Backtester(initial_capital=100000.0)
    strat_filter = StrategyFilter()

    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("❌ Model not found. Please train the model first.")
        return

    print(f"📊 Analyzing {len(tickers.get_ticker_list())} ETFs...")
    index_df, market_status_map = prepare_index_data(data_manager, feature_eng, strat_filter, index_code="000300.SH")
    data_cache = build_data_cache(
        tickers.get_ticker_list(),
        data_manager,
        feature_eng,
        index_df,
        model,
        start_date,
        end_date,
    )
    results = run_backtest_for_cache(data_cache, backtester, market_status_map)
    results = [r for r in results if r["num_trades"] > 0]

    if not results:
        print("\n⚠️ No trades triggered during this period.")
        return

    results_sorted = sorted(results, key=lambda x: x["total_return"], reverse=True)

    print("\n" + "=" * 80)
    print(f"📈 Q4 2025 Backtest Summary ({start_date} - {end_date})")
    print("=" * 80)
    print(f"{'代码':<10} {'名称':<12} {'收益':<10} {'胜率':<10} {'交易':<8} {'最大回撤':<8} {'波动':<8} {'夏普':<8} {'熊市天数'}")
    print("-" * 80)
    for res in results_sorted:
        print(
            f"{res['code']:<10} {res['name']:<12} "
            f"{res['total_return'] * 100:6.2f}%    "
            f"{res['win_rate'] * 100:5.1f}%      "
            f"{res['num_trades']:<8} "
            f"{res['max_drawdown'] * 100:6.2f}% "
            f"{res['volatility'] * 100:7.2f}% "
            f"{res['sharpe']:<8.2f} "
            f"{res['bear_days']}"
        )

    print("=" * 80)
    summary = summarize_results(results)
    print(f"整体平均收益: {summary['avg_return'] * 100:.2f}%")
    print(f"整体胜率:     {summary['overall_win_rate'] * 100:.2f}%")
    print(f"总交易次数:   {summary['total_trades']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
