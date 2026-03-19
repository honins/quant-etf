from __future__ import annotations

import numpy as np
import pandas as pd


def build_regime_feature_frame(index_df: pd.DataFrame) -> pd.DataFrame:
    if index_df.empty:
        return index_df.copy()

    frame = index_df.copy().sort_values("trade_date").reset_index(drop=True)
    close = pd.Series(frame["close"], dtype=float)
    ma20 = pd.Series(frame.get("ma20", close.rolling(20).mean()), dtype=float)
    ma60 = pd.Series(frame.get("ma60", close.rolling(60).mean()), dtype=float)
    volume = pd.Series(frame.get("vol", pd.Series(np.nan, index=frame.index)), dtype=float)

    frame["market_trend_strength"] = (ma20 - ma60) / ma60.replace(0, np.nan)
    frame["market_above_ma20"] = (close > ma20).astype(float)
    frame["market_above_ma60"] = (close > ma60).astype(float)
    frame["market_return_5d"] = close.pct_change(5)
    frame["market_return_20d"] = close.pct_change(20)
    frame["market_volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    frame["market_drawdown_60d"] = close / close.rolling(60).max() - 1
    frame["market_breadth_proxy"] = (frame["market_return_5d"] > 0).rolling(10).mean()

    if not volume.isna().all():
        volume_ma20 = volume.rolling(20).mean()
        frame["market_volume_ratio"] = volume / volume_ma20.replace(0, np.nan)
    else:
        frame["market_volume_ratio"] = np.nan

    frame["regime_label"] = np.select(
        [
            (frame["market_trend_strength"] > 0) & (frame["market_above_ma20"] == 1.0),
            (frame["market_trend_strength"] < 0) & (frame["market_above_ma60"] == 0.0),
        ],
        ["Bull Market", "Bear Market"],
        default="Volatile Market",
    )
    return frame


def attach_regime_features(asset_df: pd.DataFrame, index_feature_df: pd.DataFrame) -> pd.DataFrame:
    if asset_df.empty or index_feature_df.empty:
        return asset_df.copy()

    regime_cols = [
        "trade_date",
        "market_trend_strength",
        "market_above_ma20",
        "market_above_ma60",
        "market_return_5d",
        "market_return_20d",
        "market_volatility_20d",
        "market_drawdown_60d",
        "market_breadth_proxy",
        "market_volume_ratio",
        "regime_label",
    ]
    available_cols = [col for col in regime_cols if col in index_feature_df.columns]
    merged = asset_df.merge(index_feature_df[available_cols], on="trade_date", how="left")
    return merged
