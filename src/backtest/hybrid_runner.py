from __future__ import annotations

import numpy as np
import pandas as pd

from config import tickers
from config.settings import settings
from src.backtest.backtester import Backtester
from src.backtest.strategy_config import StrategyConfig
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.research.metrics import summarize_experiment_results
from src.models.scoring_model import BaseModel
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
    model: BaseModel,
    start_date: str,
    end_date: str | None = None,
) -> dict | None:
    df = data_manager.update_and_get_data(code)
    if df.empty:
        return None

    df = feature_eng.calculate_technical_indicators(df)
    df = feature_eng.add_relative_strength(df, index_df, period=20)
    df = feature_eng.add_regime_features(df, index_df)
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
    category = tickers.get_ticker_category(code)
    if category == "core":
        return False
    if category == "satellite":
        return config.use_dynamic_threshold
    if category == "observe":
        return False
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


def apply_cross_sectional_filter(
    data_cache: dict[str, dict],
    market_status_map: dict[str, str],
    config: StrategyConfig,
) -> dict[str, dict]:
    if not data_cache:
        return data_cache

    top_k = max(int(settings.CROSS_SECTION_TOP_K), 1)
    min_score = float(settings.CROSS_SECTION_MIN_SCORE)
    date_to_scores: dict[str, list[tuple[str, float]]] = {}

    for code, payload in data_cache.items():
        test_df = payload.get("test_df")
        probs = payload.get("probs")
        if test_df is None or probs is None:
            continue
        for i, raw_prob in enumerate(probs):
            trade_date = str(test_df.iloc[i]["trade_date"])
            current_status = market_status_map.get(trade_date, "Volatile Market")
            status_bonus = 0.02 if current_status == "Bull Market" else (-0.04 if current_status == "Bear Market" else 0.0)
            adjusted = float(raw_prob) + status_bonus
            date_to_scores.setdefault(trade_date, []).append((code, adjusted))

    survivors_by_date: dict[str, set[str]] = {}
    for trade_date, pairs in date_to_scores.items():
        ranked = sorted(pairs, key=lambda item: item[1], reverse=True)
        core_pairs = [item for item in ranked if tickers.get_ticker_category(item[0]) == "core"]
        satellite_pairs = [item for item in ranked if tickers.get_ticker_category(item[0]) == "satellite"]
        unknown_pairs = [item for item in ranked if tickers.get_ticker_category(item[0]) not in {"core", "satellite"}]

        core_top_k = max(int(settings.CROSS_SECTION_CORE_TOP_K), 0)
        satellite_top_k = max(int(settings.CROSS_SECTION_SATELLITE_TOP_K), 0)

        survivors = {
            code
            for code, score in core_pairs[:core_top_k] + satellite_pairs[:satellite_top_k]
            if score >= min_score
        }

        if len(survivors) < top_k:
            remaining_slots = top_k - len(survivors)
            filler = [item for item in ranked if item[0] not in survivors]
            survivors.update({code for code, score in filler[:remaining_slots] if score >= min_score})

        if len(survivors) < top_k and unknown_pairs:
            remaining_slots = top_k - len(survivors)
            survivors.update({code for code, score in unknown_pairs[:remaining_slots] if score >= min_score})

        survivors_by_date[trade_date] = survivors

    filtered_cache: dict[str, dict] = {}
    for code, payload in data_cache.items():
        test_df = payload.get("test_df")
        probs = payload.get("probs")
        if test_df is None or probs is None:
            continue
        filtered_probs = []
        for i, raw_prob in enumerate(probs):
            trade_date = str(test_df.iloc[i]["trade_date"])
            survivors = survivors_by_date.get(trade_date, set())
            filtered_probs.append(float(raw_prob) if code in survivors else 0.0)
        new_payload = dict(payload)
        new_payload["probs"] = np.asarray(filtered_probs, dtype=float)
        filtered_cache[code] = new_payload

    return filtered_cache


def run_backtest_for_cache(
    data_cache: dict[str, dict],
    backtester: Backtester,
    market_status_map: dict[str, str],
    threshold_overrides: dict[str, float] | None = None,
    config: StrategyConfig | None = None,
) -> list[dict]:
    config = config or StrategyConfig.from_settings()
    results = []
    filtered_cache = apply_cross_sectional_filter(data_cache, market_status_map, config)

    for code, payload in filtered_cache.items():
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
    summary = summarize_experiment_results(results)
    return {
        "avg_return": float(summary["avg_total_return"]),
        "avg_annual_return": float(summary["avg_annual_return"]),
        "avg_max_drawdown": float(summary["avg_max_drawdown"]),
        "avg_volatility": float(summary["avg_volatility"]),
        "avg_sharpe": float(summary["avg_sharpe"]),
        "avg_calmar": float(summary["avg_calmar"]),
        "avg_turnover": float(summary["avg_turnover"]),
        "avg_holding_days": float(summary["avg_holding_days"]),
        "positive_ratio": float(summary["positive_ratio"]),
        "overall_win_rate": float(summary["overall_win_rate"]),
        "total_trades": int(summary["total_trades"]),
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
    model: BaseModel,
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
