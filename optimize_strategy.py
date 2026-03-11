import random
from dataclasses import asdict
from datetime import datetime, timedelta

from config import tickers
from src.backtest.backtester import Backtester
from src.backtest.hybrid_runner import (
    build_data_cache,
    objective_score,
    prepare_index_data,
    run_backtest_for_cache,
    summarize_results,
)
from src.backtest.strategy_config import StrategyConfig
from src.data_loader.data_manager import DataManager
from src.data_loader.tushare_loader import TushareLoader
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter


def sample_configs(seed: int = 42, trials: int = 120) -> list[StrategyConfig]:
    rng = random.Random(seed)
    configs = [StrategyConfig.from_settings()]

    bull_base_choices = [0.58, 0.60, 0.62, 0.65]
    bull_aggressive_choices = [0.45, 0.48, 0.50, 0.53, 0.55]
    volatile_choices = [0.62, 0.65, 0.68, 0.70]
    bear_choices = [0.72, 0.75, 0.80, 0.85]
    exit_score_choices = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    use_dynamic_choices = [False, True]
    dynamic_quantile_choices = [0.70, 0.80, 0.85, 0.90]
    dynamic_min_choices = [0.45, 0.50, 0.55]
    dynamic_max_choices = [0.65, 0.70, 0.75]
    atr_choices = [1.0, 1.2, 1.3, 1.5]
    atr_aggressive_choices = [2.0, 2.5, 3.0, 3.5]
    exit_lookback_choices = [15, 22, 30]
    max_dd_choices = [0.05, 0.07, 0.10]

    while len(configs) < trials + 1:
        bull_base = rng.choice(bull_base_choices)
        bull_aggressive = rng.choice([x for x in bull_aggressive_choices if x <= bull_base])
        volatile = rng.choice([x for x in volatile_choices if x >= bull_base])
        bear = rng.choice([x for x in bear_choices if x >= volatile])
        dynamic_min = rng.choice(dynamic_min_choices)
        dynamic_max = rng.choice([x for x in dynamic_max_choices if x >= dynamic_min])

        config = StrategyConfig(
            bull_base_threshold=bull_base,
            bull_aggressive_threshold=bull_aggressive,
            volatile_threshold=volatile,
            bear_threshold=bear,
            signal_exit_threshold=rng.choice(exit_score_choices),
            use_dynamic_threshold=rng.choice(use_dynamic_choices),
            dynamic_threshold_lookback=rng.choice([30, 45, 60, 90]),
            dynamic_threshold_quantile=rng.choice(dynamic_quantile_choices),
            dynamic_threshold_min=dynamic_min,
            dynamic_threshold_max=dynamic_max,
            atr_multiplier=rng.choice(atr_choices),
            atr_multiplier_aggressive=rng.choice(atr_aggressive_choices),
            exit_lookback_period=rng.choice(exit_lookback_choices),
            max_drawdown_stop=rng.choice(max_dd_choices),
        )
        if config not in configs:
            configs.append(config)

    return configs


def evaluate_config(
    config: StrategyConfig,
    data_cache_90: dict,
    data_cache_180: dict,
    market_status_map: dict[str, str],
    backtester: Backtester,
) -> dict:
    results_90 = run_backtest_for_cache(data_cache_90, backtester, market_status_map, config=config)
    results_180 = run_backtest_for_cache(data_cache_180, backtester, market_status_map, config=config)

    summary_90 = summarize_results(results_90)
    summary_180 = summarize_results(results_180)
    score_90 = objective_score(summary_90)
    score_180 = objective_score(summary_180)
    combined_score = 0.65 * score_90 + 0.35 * score_180

    return {
        "config": config,
        "score_90": score_90,
        "score_180": score_180,
        "combined_score": combined_score,
        "summary_90": summary_90,
        "summary_180": summary_180,
    }


def main():
    print("Optimizing strategy configuration...")

    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    strat_filter = StrategyFilter()
    backtester = Backtester()

    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("Model not found. Train the model first.")
        return

    index_df, market_status_map = prepare_index_data(data_manager, feature_eng, strat_filter, index_code="000300.SH")
    ticker_list = tickers.get_tradable_ticker_list()

    end_date = datetime.now()
    start_90 = (end_date - timedelta(days=90)).strftime("%Y%m%d")
    start_180 = (end_date - timedelta(days=180)).strftime("%Y%m%d")

    data_cache_90 = build_data_cache(ticker_list, data_manager, feature_eng, index_df, model, start_90)
    data_cache_180 = build_data_cache(ticker_list, data_manager, feature_eng, index_df, model, start_180)

    trials = sample_configs()
    evaluations = []
    for idx, config in enumerate(trials, start=1):
        evaluation = evaluate_config(config, data_cache_90, data_cache_180, market_status_map, backtester)
        evaluations.append(evaluation)
        if idx % 20 == 0:
            print(f"Evaluated {idx}/{len(trials)} candidates...")

    evaluations.sort(key=lambda x: x["combined_score"], reverse=True)
    baseline = evaluations[0]
    default_config = StrategyConfig.from_settings()
    for item in evaluations:
        if item["config"] == default_config:
            baseline = item
            break

    print("\nTop strategy candidates:")
    for rank, item in enumerate(evaluations[:10], start=1):
        cfg = item["config"]
        print(
            f"{rank:>2}. score={item['combined_score']:.2f} "
            f"90d={item['summary_90']['avg_return']*100:.2f}% "
            f"180d={item['summary_180']['avg_return']*100:.2f}% "
            f"exit={cfg.signal_exit_threshold:.2f} "
            f"bull={cfg.bull_base_threshold:.2f}/{cfg.bull_aggressive_threshold:.2f} "
            f"vol={cfg.volatile_threshold:.2f} bear={cfg.bear_threshold:.2f} "
            f"atr={cfg.atr_multiplier:.2f}/{cfg.atr_multiplier_aggressive:.2f} "
            f"lookback={cfg.exit_lookback_period} dynamic={cfg.use_dynamic_threshold}"
        )

    best = evaluations[0]
    print("\nBaseline config:")
    print(asdict(baseline["config"]))
    print(baseline["summary_90"])
    print(baseline["summary_180"])

    print("\nBest config:")
    print(asdict(best["config"]))
    print(best["summary_90"])
    print(best["summary_180"])


if __name__ == "__main__":
    main()
