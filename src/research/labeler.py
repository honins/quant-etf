from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import settings


@dataclass(slots=True)
class LabelConfig:
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    take_profit: float = 0.03
    stop_loss: float = 0.02
    baseline_horizon: int = settings.TRAIN_LABEL_HORIZON
    baseline_threshold: float = settings.TRAIN_LABEL_THRESHOLD
    baseline_end_weight: float = settings.TRAIN_LABEL_END_WEIGHT
    baseline_drawdown_penalty: float = settings.TRAIN_LABEL_DRAWDOWN_PENALTY


def add_legacy_training_labels(
    df: pd.DataFrame,
    horizon: int | None = None,
    threshold: float | None = None,
    end_weight: float | None = None,
    drawdown_penalty: float | None = None,
) -> pd.DataFrame:
    horizon = horizon or settings.TRAIN_LABEL_HORIZON
    threshold = threshold or settings.TRAIN_LABEL_THRESHOLD
    end_weight = end_weight if end_weight is not None else settings.TRAIN_LABEL_END_WEIGHT
    drawdown_penalty = (
        drawdown_penalty if drawdown_penalty is not None else settings.TRAIN_LABEL_DRAWDOWN_PENALTY
    )

    labeled = df.copy()
    future_close = labeled["close"].shift(-1)
    future_max_close = future_close.rolling(window=horizon, min_periods=1).max().shift(-(horizon - 1))
    future_min_close = future_close.rolling(window=horizon, min_periods=1).min().shift(-(horizon - 1))
    future_end_close = labeled["close"].shift(-horizon)

    labeled[f"future_max_ret_{horizon}d"] = future_max_close / labeled["close"] - 1
    labeled[f"future_min_ret_{horizon}d"] = future_min_close / labeled["close"] - 1
    labeled[f"future_end_ret_{horizon}d"] = future_end_close / labeled["close"] - 1

    upside = labeled[f"future_max_ret_{horizon}d"].fillna(0.0)
    downside = labeled[f"future_min_ret_{horizon}d"].clip(upper=0).abs().fillna(0.0)
    end_ret = labeled[f"future_end_ret_{horizon}d"].fillna(0.0)

    quality = upside + end_weight * end_ret - drawdown_penalty * downside
    labeled[f"quality_{horizon}d"] = quality
    labeled[f"ret_{horizon}d"] = upside
    labeled["target"] = ((upside > threshold) & (quality > 0)).astype(int)
    labeled["sample_weight"] = (
        1.0
        + upside.clip(lower=0, upper=0.12) * 12
        + downside.clip(lower=0, upper=0.10) * 8
    )
    return labeled


class MultiTaskLabeler:
    def __init__(self, config: LabelConfig | None = None):
        self.config = config or LabelConfig()

    def add_labels(self, df: pd.DataFrame, include_legacy: bool = True) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        labeled = df.copy().sort_values("trade_date").reset_index(drop=True)
        close = labeled["close"].astype(float)
        max_horizon = max(self.config.horizons)

        future_returns = pd.DataFrame(index=labeled.index)
        for horizon in self.config.horizons:
            future_close = close.shift(-horizon)
            labeled[f"future_ret_{horizon}d"] = future_close / close - 1
            future_returns[horizon] = labeled[f"future_ret_{horizon}d"]

        future_steps = pd.DataFrame(
            {
                step: close.shift(-step) / close - 1
                for step in range(1, max_horizon + 1)
            },
            index=labeled.index,
        )

        labeled[f"future_max_ret_{max_horizon}d"] = future_steps.max(axis=1)
        labeled[f"future_min_ret_{max_horizon}d"] = future_steps.min(axis=1)

        best_holding_days: list[float] = []
        take_profit_first: list[float] = []
        stop_loss_first: list[float] = []

        for row in future_steps.itertuples(index=False, name=None):
            values = [value for value in row if pd.notna(value)]
            if not values:
                best_holding_days.append(np.nan)
                take_profit_first.append(np.nan)
                stop_loss_first.append(np.nan)
                continue

            best_idx = int(np.argmax(values)) + 1
            best_holding_days.append(float(best_idx))

            first_tp = next(
                (idx for idx, value in enumerate(values, start=1) if value >= self.config.take_profit),
                None,
            )
            first_sl = next(
                (idx for idx, value in enumerate(values, start=1) if value <= -self.config.stop_loss),
                None,
            )

            take_profit_first.append(
                float(first_tp is not None and (first_sl is None or first_tp <= first_sl))
            )
            stop_loss_first.append(
                float(first_sl is not None and (first_tp is None or first_sl < first_tp))
            )

        labeled["best_holding_days"] = best_holding_days
        labeled["hit_take_profit_first"] = take_profit_first
        labeled["hit_stop_loss_first"] = stop_loss_first
        labeled["meta_target"] = (
            (labeled["hit_take_profit_first"] == 1.0)
            & (labeled["hit_stop_loss_first"].fillna(0.0) == 0.0)
        ).astype(int)

        if include_legacy:
            labeled = add_legacy_training_labels(
                labeled,
                horizon=self.config.baseline_horizon,
                threshold=self.config.baseline_threshold,
                end_weight=self.config.baseline_end_weight,
                drawdown_penalty=self.config.baseline_drawdown_penalty,
            )

        return labeled
