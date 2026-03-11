from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings


@dataclass(frozen=True)
class StrategyConfig:
    bull_base_threshold: float
    bull_aggressive_threshold: float
    volatile_threshold: float
    bear_threshold: float
    signal_exit_threshold: float
    use_dynamic_threshold: bool
    dynamic_threshold_lookback: int
    dynamic_threshold_quantile: float
    dynamic_threshold_min: float
    dynamic_threshold_max: float
    atr_multiplier: float
    atr_multiplier_aggressive: float
    exit_lookback_period: int
    max_drawdown_stop: float

    @classmethod
    def from_settings(cls) -> "StrategyConfig":
        return cls(
            bull_base_threshold=settings.BULL_BASE_THRESHOLD,
            bull_aggressive_threshold=settings.BULL_AGGRESSIVE_THRESHOLD,
            volatile_threshold=settings.VOLATILE_THRESHOLD,
            bear_threshold=settings.BEAR_THRESHOLD,
            signal_exit_threshold=settings.SIGNAL_EXIT_THRESHOLD,
            use_dynamic_threshold=settings.USE_DYNAMIC_THRESHOLD,
            dynamic_threshold_lookback=settings.DYNAMIC_THRESHOLD_LOOKBACK,
            dynamic_threshold_quantile=settings.DYNAMIC_THRESHOLD_QUANTILE,
            dynamic_threshold_min=settings.DYNAMIC_THRESHOLD_MIN,
            dynamic_threshold_max=settings.DYNAMIC_THRESHOLD_MAX,
            atr_multiplier=settings.ATR_MULTIPLIER,
            atr_multiplier_aggressive=settings.ATR_MULTIPLIER_AGGRESSIVE,
            exit_lookback_period=settings.EXIT_LOOKBACK_PERIOD,
            max_drawdown_stop=settings.MAX_DRAWDOWN_STOP,
        )
