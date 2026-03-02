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
        
        # 1. 均线 (MA) & 乖离率 (Bias)
        df['ma5'] = close.rolling(window=5).mean()
        df['ma20'] = close.rolling(window=20).mean()
        df['ma60'] = close.rolling(window=60).mean()
        
        # 归一化: 乖离率 (Bias) = (Price - MA) / MA
        df['bias_5'] = (close - df['ma5']) / df['ma5']
        df['bias_20'] = (close - df['ma20']) / df['ma20']
        df['bias_60'] = (close - df['ma60']) / df['ma60']
        
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

        # 3. ATR — 使用 Wilder's RMA (标准定义)
        # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # 修复：改用 RMA (Wilder's Smoothing) 而非 SMA
        df['atr'] = calculate_rma(tr, settings.ATR_PERIOD)
        
        # 归一化: 波动率占比
        df['atr_pct'] = df['atr'] / close

        # 4. Volume Ratio (量比)
        # 替代 OBV (OBV 是累积值，非平稳)
        df['vol_ma5'] = df['vol'].rolling(window=5).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma5']
        
        # 5. MACD (Normalized)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
        
        # 归一化: 除以价格
        df['macd_norm'] = df['macd'] / close
        df['macdsignal_norm'] = df['macdsignal'] / close
        df['macdhist_norm'] = df['macdhist'] / close
        
        # 6. BBANDS & Position
        df['middle'] = df['ma20']
        std = close.rolling(window=20).std()
        df['upper'] = df['middle'] + (std * 2)
        df['lower'] = df['middle'] - (std * 2)
        
        # 归一化: 布林带相对位置 (0=Lower, 1=Upper)
        # 修复: 防止除以零 (例如价格长期不变/停牌的ETF)
        bb_range = (df['upper'] - df['lower']).replace(0, np.nan)
        df['bb_pos'] = (close - df['lower']) / bb_range
        
        return df

    @staticmethod
    def add_relative_strength(etf_df: pd.DataFrame, index_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        【优化2】 添加相对大盘强弱特征 (Cross-Sectional Features)
        
        通过将 ETF 自身的涨幅与基准指数对比，捕捉板块轮动的 Alpha。
        - rs_{period}d: ETF N日涨幅 - 指数N日涨幅 (正值代表跑赢大盘)
        - rel_vol: ETF ATR占比 / 指数ATR占比 (>1代表ETF波动比大盘更活跃)
        
        注意: 该方法需要在 calculate_technical_indicators 之后调用。
        """
        if etf_df.empty or index_df.empty:
            return etf_df
        
        etf_df = etf_df.copy()
        
        # ETF N日涨幅
        etf_ret = etf_df['close'].pct_change(period)
        
        # 指数 N日涨幅 (按日期对齐)
        index_ret = index_df.set_index('trade_date')['close'].pct_change(period)
        aligned_index_ret = etf_df['trade_date'].map(index_ret)
        
        # 相对强度 = ETF涨幅 - 指数涨幅
        etf_df[f'rs_{period}d'] = etf_ret - aligned_index_ret
        
        # 相对波动率 = ETF ATR占比 / 指数ATR占比
        if 'atr_pct' in etf_df.columns and 'atr_pct' in index_df.columns:
            index_atr_pct = index_df.set_index('trade_date')['atr_pct']
            aligned_index_atr_pct = etf_df['trade_date'].map(index_atr_pct)
            etf_df['rel_vol'] = etf_df['atr_pct'] / aligned_index_atr_pct.replace(0, np.nan)
        
        return etf_df

    @staticmethod
    def add_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> pd.DataFrame:
        """
        添加训练标签: 未来 N 天 (收盘价) 最高涨幅是否超过 M%
        
        【优化1】 标签改造：改用未来N天内的最高收盘价，而非最高价(High)。
        避免"偷价"：即仅靠日内瞬间刺穿阈值但无法真实成交的情形被错误标记为正样本。
        """
        # 使用未来 N 天内的最高收盘价（不使用 High，避免偷价）
        future_max_close = df['close'].shift(-1).rolling(window=horizon, min_periods=1).max().shift(-(horizon - 1))
        df[f'ret_{horizon}d'] = future_max_close / df['close'] - 1
        
        # 标签: 1 if return > threshold else 0
        df['target'] = (df[f'ret_{horizon}d'] > threshold).astype(int)
        
        return df
