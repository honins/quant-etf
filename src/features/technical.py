import pandas as pd
import numpy as np
from config.settings import settings

class FeatureEngineer:
    """
    计算技术指标 (Pure Pandas Implementation)
    """
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df = df.copy()
        # 确保按日期升序
        df = df.sort_values('trade_date')
        
        return FeatureEngineer._calc_with_pandas(df)

    @staticmethod
    def _calc_with_pandas(df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        
        # 1. 均线 (MA)
        df['ma5'] = close.rolling(window=5).mean()
        df['ma20'] = close.rolling(window=20).mean()
        df['ma60'] = close.rolling(window=60).mean()
        
        # 2. RSI (Wilder's Smoothing)
        delta = close.diff()
        
        # Helper for Wilder's Smoothing (RMA)
        def calculate_rma(series, period):
            return series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # RSI 14
        avg_gain_14 = calculate_rma(gain, 14)
        avg_loss_14 = calculate_rma(loss, 14)
        rs_14 = avg_gain_14 / avg_loss_14
        df['rsi_14'] = 100 - (100 / (1 + rs_14))
        
        # RSI 6
        avg_gain_6 = calculate_rma(gain, 6)
        avg_loss_6 = calculate_rma(loss, 6)
        rs_6 = avg_gain_6 / avg_loss_6
        df['rsi_6'] = 100 - (100 / (1 + rs_6))

        # 3. ATR (简化版)
        # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=settings.ATR_PERIOD).mean()

        # 4. OBV
        # if close > prev_close: vol else -vol
        # cumsum
        obv_vol = np.where(close > prev_close, df['vol'], np.where(close < prev_close, -df['vol'], 0))
        df['obv'] = obv_vol.cumsum()
        
        # 5. MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
        
        # 6. BBANDS
        df['middle'] = df['ma20']
        std = close.rolling(window=20).std()
        df['upper'] = df['middle'] + (std * 2)
        df['lower'] = df['middle'] - (std * 2)
        
        return df

    @staticmethod
    def add_labels(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.01) -> pd.DataFrame:
        """
        添加训练标签: 未来 N 天是否上涨超过 M%
        """
        # 未来 N 天收益率
        df[f'ret_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # 标签: 1 if return > threshold else 0
        df['target'] = (df[f'ret_{horizon}d'] > threshold).astype(int)
        
        return df
