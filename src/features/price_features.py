from __future__ import annotations

import numpy as np
import pandas as pd


def _calculate_rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_price_features(df: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = df.copy().sort_values("trade_date")
    close = pd.Series(result["close"], dtype=float)
    high = pd.Series(result["high"], dtype=float)
    low = pd.Series(result["low"], dtype=float)
    vol = pd.Series(result["vol"], dtype=float)

    result["ma5"] = close.rolling(window=5).mean()
    result["ma20"] = close.rolling(window=20).mean()
    result["ma60"] = close.rolling(window=60).mean()

    result["bias_5"] = (close - result["ma5"]) / result["ma5"]
    result["bias_20"] = (close - result["ma20"]) / result["ma20"]
    result["bias_60"] = (close - result["ma60"]) / result["ma60"]

    delta = close.diff()
    gain = pd.Series(delta.where(delta > 0, 0.0), dtype=float)
    loss = pd.Series(-delta.where(delta < 0, 0.0), dtype=float)

    avg_gain_14 = _calculate_rma(gain, 14)
    avg_loss_14 = _calculate_rma(loss, 14)
    rs_14 = avg_gain_14 / avg_loss_14
    result["rsi_14"] = 100 - (100 / (1 + rs_14))

    avg_gain_6 = _calculate_rma(gain, 6)
    avg_loss_6 = _calculate_rma(loss, 6)
    rs_6 = avg_gain_6 / avg_loss_6
    result["rsi_6"] = 100 - (100 / (1 + rs_6))

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result["atr"] = _calculate_rma(pd.Series(tr, dtype=float), atr_period)
    result["atr_pct"] = result["atr"] / close

    result["vol_ma5"] = vol.rolling(window=5).mean()
    result["vol_ma20"] = vol.rolling(window=20).mean()
    result["vol_ratio"] = vol / result["vol_ma5"]
    result["vol_ratio_20"] = vol / result["vol_ma20"]

    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    result["macd"] = exp12 - exp26
    result["macdsignal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macdhist"] = result["macd"] - result["macdsignal"]
    result["macd_norm"] = result["macd"] / close
    result["macdsignal_norm"] = result["macdsignal"] / close
    result["macdhist_norm"] = result["macdhist"] / close

    result["middle"] = result["ma20"]
    std = close.rolling(window=20).std()
    result["upper"] = result["middle"] + (std * 2)
    result["lower"] = result["middle"] - (std * 2)
    bb_range = (result["upper"] - result["lower"]).replace(0, np.nan)
    result["bb_pos"] = (close - result["lower"]) / bb_range

    rolling_high_20 = high.rolling(window=20).max()
    rolling_high_60 = high.rolling(window=60).max()
    rolling_low_20 = low.rolling(window=20).min()

    result["ret_5"] = close.pct_change(5)
    result["ret_10"] = close.pct_change(10)
    result["ret_20"] = close.pct_change(20)
    result["trend_gap"] = (result["ma20"] - result["ma60"]) / result["ma60"]
    result["ma20_slope_5"] = result["ma20"].pct_change(5)
    result["ma60_slope_10"] = result["ma60"].pct_change(10)
    result["breakout_20"] = close / rolling_high_20.shift(1) - 1
    result["drawdown_20"] = close / rolling_high_20 - 1
    result["drawdown_60"] = close / rolling_high_60 - 1
    result["rebound_20"] = close / rolling_low_20 - 1
    result["rsi_spread"] = (result["rsi_6"] - result["rsi_14"]) / 100
    result["close_to_ma20"] = close / result["ma20"] - 1
    result["close_to_ma60"] = close / result["ma60"] - 1
    result["intraday_range"] = (high - low) / close

    return result
