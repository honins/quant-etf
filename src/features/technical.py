import pandas as pd
import numpy as np

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("Warning: TA-Lib not found. Using Pandas implementation.")

class FeatureEngineer:
    """
    计算技术指标
    """
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df = df.copy()
        # 确保按日期升序
        df = df.sort_values('trade_date')
        
        if HAS_TALIB:
            return FeatureEngineer._calc_with_talib(df)
        else:
            return FeatureEngineer._calc_with_pandas(df)

    @staticmethod
    def _calc_with_talib(df: pd.DataFrame) -> pd.DataFrame:
        # TA-Lib 需要 numpy array (float)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        vol = df['vol'].values

        # 1. 均线 (MA)
        df['ma5'] = talib.SMA(close, timeperiod=5)
        df['ma20'] = talib.SMA(close, timeperiod=20)
        df['ma60'] = talib.SMA(close, timeperiod=60)
        
        # 2. 相对强弱指标 (RSI)
        df['rsi_6'] = talib.RSI(close, timeperiod=6)
        df['rsi_14'] = talib.RSI(close, timeperiod=14)
        
        # 3. 波动率 (ATR) - 用于止损
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # 4. 能量潮 (OBV) - 资金流向
        df['obv'] = talib.OBV(close, vol)
        
        # 5. MACD
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        # 6. 布林带 (Bollinger Bands)
        df['upper'], df['middle'], df['lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        return df

    @staticmethod
    def _calc_with_pandas(df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        
        # 1. 均线 (MA)
        df['ma5'] = close.rolling(window=5).mean()
        df['ma20'] = close.rolling(window=20).mean()
        df['ma60'] = close.rolling(window=60).mean()
        
        # 2. RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean() # 简化版RSI算法，非Wilder平滑
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # RSI 6
        gain6 = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss6 = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs6 = gain6 / loss6
        df['rsi_6'] = 100 - (100 / (1 + rs6))

        # 3. ATR (简化版)
        # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

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
