from __future__ import annotations

import numpy as np
import pandas as pd

from config import tickers
from config.settings import settings
from src.backtest.backtester import Backtester
from src.backtest.strategy_config import StrategyConfig
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter


def prepare_index_data(
    data_manager: DataManager,
    feature_eng: FeatureEngineer,
    strat_filter: StrategyFilter,
    index_code: str = "000300.SH",
) -> tuple[pd.DataFrame, dict[str, str]]:
    index_df = data_manager.update_and_get_data(index_code, is_index=True)
    index_df = feature_eng.calculate_technical_indicators(index_df)

    market_status_map: dict[str, str] = {}
    for i in range(len(index_df)):
        trade_date = str(index_df.iloc[i]["trade_date"])
        market_status_map[trade_date] = strat_filter._detect_market_regime(index_df.iloc[: i + 1])

    return index_df, market_status_map


def prepare_ticker_dataset(
    code: str,
    data_manager: DataManager,
    feature_eng: FeatureEngineer,
    index_df: pd.DataFrame,
    model: XGBoostModel,
    start_date: str,
    end_date: str | None = None,
) -> dict | None:
    df = data_manager.update_and_get_data(code)
    if df.empty:
        return None

    df = feature_eng.calculate_technical_indicators(df)
    df = feature_eng.add_relative_strength(df, index_df, period=20)
    df = df.dropna()

    mask = df["trade_date"].astype(str) >= start_date
    if end_date is not None:
        mask &= df["trade_date"].astype(str) <= end_date
    test_df = df[mask].copy()

    if len(test_df) < 10:
        return None

    probs = model.predict_batch(test_df)
    return {"test_df": test_df, "probs": probs}


def use_dynamic_for_code(
    code: str,
    threshold_overrides: dict[str, float] | None,
    config: StrategyConfig,
) -> bool:
    if threshold_overrides is not None:
        return False
    if code in tickers.WIDE_INDEX_TICKERS:
        return False
    if code in tickers.SECTOR_TICKERS:
        return config.use_dynamic_threshold
    return config.use_dynamic_threshold


def _resolve_bull_threshold(
    probs: np.ndarray,
    i: int,
    code: str,
    use_dynamic: bool,
    threshold_overrides: dict[str, float] | None,
    config: StrategyConfig,
) -> float:
    threshold = None
    if use_dynamic:
        window_start = max(0, i - config.dynamic_threshold_lookback + 1)
        threshold = float(np.quantile(probs[window_start : i + 1], config.dynamic_threshold_quantile))
        threshold = max(config.dynamic_threshold_min, min(config.dynamic_threshold_max, threshold))
    if threshold_overrides is not None:
        threshold = threshold_overrides.get(code, threshold)
    if threshold is None:
        threshold = settings.TICKER_BULL_THRESHOLDS.get(code)
    if threshold is None:
        threshold = config.bull_aggressive_threshold if code in settings.AGGRESSIVE_TICKERS else config.bull_base_threshold
    return round(float(threshold), 4)


def build_adjusted_probs(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    market_status_map: dict[str, str],
    code: str,
    use_dynamic: bool,
    threshold_overrides: dict[str, float] | None = None,
    config: StrategyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    config = config or StrategyConfig.from_settings()
    entry_probs = []
    exit_probs = []
    bear_days = 0

    for i, raw_prob in enumerate(probs):
        trade_date = str(test_df.iloc[i]["trade_date"])
        current_status = market_status_map.get(trade_date, "Volatile Market")

        if current_status == "Bear Market":
            threshold = config.bear_threshold
            if raw_prob < threshold:
                bear_days += 1
        elif current_status == "Volatile Market":
            threshold = config.volatile_threshold
        else:
            threshold = _resolve_bull_threshold(
                probs,
                i,
                code,
                use_dynamic,
                threshold_overrides,
                config,
            )

        entry_probs.append(raw_prob if raw_prob >= threshold else 0.0)

        exit_prob = raw_prob
        if current_status == "Bear Market" and raw_prob < config.bear_threshold:
            exit_prob = 0.0
        exit_probs.append(exit_prob)

    return np.array(entry_probs), np.array(exit_probs), bear_days


def run_backtest_for_cache(
    data_cache: dict[str, dict],
    backtester: Backtester,
    market_status_map: dict[str, str],
    threshold_overrides: dict[str, float] | None = None,
    config: StrategyConfig | None = None,
) -> list[dict]:
    config = config or StrategyConfig.from_settings()
    results = []

    for code, payload in data_cache.items():
        test_df = payload["test_df"]
        probs = payload["probs"]
        use_dynamic = use_dynamic_for_code(code, threshold_overrides, config)
        entry_probs, exit_probs, bear_days = build_adjusted_probs(
            test_df,
            probs,
            market_status_map,
            code,
            use_dynamic,
            threshold_overrides,
            config,
        )
        result = backtester.run(
            test_df,
            entry_probs,
            threshold=0.0,
            code=code,
            exit_probs=exit_probs,
            config=config,
        )
        result["code"] = code
        result["name"] = tickers.TICKERS[code]
        result["bear_days"] = bear_days
        result["mode"] = "dynamic" if use_dynamic else "fixed"
        results.append(result)

    return results


def summarize_results(results: list[dict]) -> dict[str, float]:
    total_trades = sum(r["num_trades"] for r in results)
    winning_trades = sum(int(round(r["win_rate"] * r["num_trades"])) for r in results)
    avg_return = float(np.mean([r["total_return"] for r in results])) if results else 0.0
    avg_max_drawdown = float(np.mean([r["max_drawdown"] for r in results])) if results else 0.0
    avg_volatility = float(np.mean([r["volatility"] for r in results])) if results else 0.0
    positive_ratio = float(np.mean([r["total_return"] > 0 for r in results])) if results else 0.0
    overall_win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    return {
        "avg_return": avg_return,
        "avg_max_drawdown": avg_max_drawdown,
        "avg_volatility": avg_volatility,
        "positive_ratio": positive_ratio,
        "overall_win_rate": overall_win_rate,
        "total_trades": total_trades,
    }


def objective_score(summary: dict[str, float]) -> float:
    return (
        100.0 * summary["avg_return"]
        - 35.0 * summary["avg_max_drawdown"]
        - 12.0 * summary["avg_volatility"]
        + 8.0 * summary["positive_ratio"]
        + 5.0 * summary["overall_win_rate"]
    )


def build_data_cache(
    codes: list[str],
    data_manager: DataManager,
    feature_eng: FeatureEngineer,
    index_df: pd.DataFrame,
    model: XGBoostModel,
    start_date: str,
    end_date: str | None = None,
) -> dict[str, dict]:
    data_cache: dict[str, dict] = {}
    for code in codes:
        dataset = prepare_ticker_dataset(
            code,
            data_manager,
            feature_eng,
            index_df,
            model,
            start_date,
            end_date,
        )
        if dataset is not None:
            data_cache[code] = dataset
    return data_cache
