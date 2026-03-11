import numpy as np
import pandas as pd

from config.settings import settings


class FeatureEngineer:
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df = df.sort_values("trade_date")
        return FeatureEngineer._calc_with_pandas(df)

    @staticmethod
    def _calc_with_pandas(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["vol"]

        df["ma5"] = close.rolling(window=5).mean()
        df["ma20"] = close.rolling(window=20).mean()
        df["ma60"] = close.rolling(window=60).mean()

        df["bias_5"] = (close - df["ma5"]) / df["ma5"]
        df["bias_20"] = (close - df["ma20"]) / df["ma20"]
        df["bias_60"] = (close - df["ma60"]) / df["ma60"]

        delta = close.diff()

        def calculate_rma(series: pd.Series, period: int) -> pd.Series:
            return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain_14 = calculate_rma(gain, 14)
        avg_loss_14 = calculate_rma(loss, 14)
        rs_14 = avg_gain_14 / avg_loss_14
        df["rsi_14"] = 100 - (100 / (1 + rs_14))

        avg_gain_6 = calculate_rma(gain, 6)
        avg_loss_6 = calculate_rma(loss, 6)
        rs_6 = avg_gain_6 / avg_loss_6
        df["rsi_6"] = 100 - (100 / (1 + rs_6))

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = calculate_rma(tr, settings.ATR_PERIOD)
        df["atr_pct"] = df["atr"] / close

        df["vol_ma5"] = vol.rolling(window=5).mean()
        df["vol_ma20"] = vol.rolling(window=20).mean()
        df["vol_ratio"] = vol / df["vol_ma5"]
        df["vol_ratio_20"] = vol / df["vol_ma20"]

        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = exp12 - exp26
        df["macdsignal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macdhist"] = df["macd"] - df["macdsignal"]
        df["macd_norm"] = df["macd"] / close
        df["macdsignal_norm"] = df["macdsignal"] / close
        df["macdhist_norm"] = df["macdhist"] / close

        df["middle"] = df["ma20"]
        std = close.rolling(window=20).std()
        df["upper"] = df["middle"] + (std * 2)
        df["lower"] = df["middle"] - (std * 2)
        bb_range = (df["upper"] - df["lower"]).replace(0, np.nan)
        df["bb_pos"] = (close - df["lower"]) / bb_range

        rolling_high_20 = high.rolling(window=20).max()
        rolling_high_60 = high.rolling(window=60).max()
        rolling_low_20 = low.rolling(window=20).min()

        df["ret_5"] = close.pct_change(5)
        df["ret_10"] = close.pct_change(10)
        df["ret_20"] = close.pct_change(20)
        df["trend_gap"] = (df["ma20"] - df["ma60"]) / df["ma60"]
        df["ma20_slope_5"] = df["ma20"].pct_change(5)
        df["ma60_slope_10"] = df["ma60"].pct_change(10)
        df["breakout_20"] = close / rolling_high_20.shift(1) - 1
        df["drawdown_20"] = close / rolling_high_20 - 1
        df["drawdown_60"] = close / rolling_high_60 - 1
        df["rebound_20"] = close / rolling_low_20 - 1
        df["rsi_spread"] = (df["rsi_6"] - df["rsi_14"]) / 100
        df["close_to_ma20"] = close / df["ma20"] - 1
        df["close_to_ma60"] = close / df["ma60"] - 1
        df["intraday_range"] = (high - low) / close

        return df

    @staticmethod
    def add_relative_strength(etf_df: pd.DataFrame, index_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        if etf_df.empty or index_df.empty:
            return etf_df

        etf_df = etf_df.copy()
        etf_ret = etf_df["close"].pct_change(period)
        index_ret = index_df.set_index("trade_date")["close"].pct_change(period)
        aligned_index_ret = etf_df["trade_date"].map(index_ret)
        etf_df[f"rs_{period}d"] = etf_ret - aligned_index_ret

        if "atr_pct" in etf_df.columns and "atr_pct" in index_df.columns:
            index_atr_pct = index_df.set_index("trade_date")["atr_pct"]
            aligned_index_atr_pct = etf_df["trade_date"].map(index_atr_pct)
            etf_df["rel_vol"] = etf_df["atr_pct"] / aligned_index_atr_pct.replace(0, np.nan)

        return etf_df

    @staticmethod
    def add_labels(
        df: pd.DataFrame,
        horizon: int | None = None,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        horizon = horizon or settings.TRAIN_LABEL_HORIZON
        threshold = threshold or settings.TRAIN_LABEL_THRESHOLD

        future_window = horizon
        future_close = df["close"].shift(-1)
        future_max_close = future_close.rolling(window=future_window, min_periods=1).max().shift(-(future_window - 1))
        future_min_close = future_close.rolling(window=future_window, min_periods=1).min().shift(-(future_window - 1))
        future_end_close = df["close"].shift(-future_window)

        df[f"future_max_ret_{horizon}d"] = future_max_close / df["close"] - 1
        df[f"future_min_ret_{horizon}d"] = future_min_close / df["close"] - 1
        df[f"future_end_ret_{horizon}d"] = future_end_close / df["close"] - 1

        upside = df[f"future_max_ret_{horizon}d"].fillna(0.0)
        downside = df[f"future_min_ret_{horizon}d"].clip(upper=0).abs().fillna(0.0)
        end_ret = df[f"future_end_ret_{horizon}d"].fillna(0.0)

        quality = upside + settings.TRAIN_LABEL_END_WEIGHT * end_ret - settings.TRAIN_LABEL_DRAWDOWN_PENALTY * downside
        df[f"quality_{horizon}d"] = quality

        df[f"ret_{horizon}d"] = upside
        df["target"] = ((upside > threshold) & (quality > 0)).astype(int)

        sample_weight = 1.0 + upside.clip(lower=0, upper=0.12) * 12 + downside.clip(lower=0, upper=0.10) * 8
        df["sample_weight"] = sample_weight

        return df
