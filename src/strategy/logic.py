import pandas as pd
import numpy as np
from config.settings import settings

class StrategyFilter:
    """
    策略过滤器：结合大盘趋势调整买入标准
    """
    @staticmethod
    def dynamic_threshold(scores) -> float | None:
        if scores is None or len(scores) == 0:
            return None
        q = float(np.quantile(scores, settings.DYNAMIC_THRESHOLD_QUANTILE))
        thr = max(settings.DYNAMIC_THRESHOLD_MIN, min(settings.DYNAMIC_THRESHOLD_MAX, q))
        return round(thr, 2)

    @staticmethod
    def _detect_market_regime(index_df: pd.DataFrame) -> str:
        """
        【优化3】 大盘牛熊判定：双均线 + 滞后确认，消除频繁假信号 (Whipsaw)。

        规则（优先级从高到低）：
        1. 确认牛市: MA20 > MA60 且 当前价格 > MA20
        2. 确认熊市: MA20 < MA60 且 当前价格持续在 MA60 下方 >= CONFIRM_DAYS 天
        3. 震荡区: 其余情况（均线粘合或价格夹在均线之间），维持上一状态（默认震荡中性）

        注意: 熊市确认需要收盘价在 MA60 下方连续 CONFIRM_DAYS 天才切换，防止假突破。
        """
        confirm_days = settings.MARKET_STATE_CONFIRM_DAYS
        current = index_df.iloc[-1]

        ma20 = current.get('ma20', None)
        ma60 = current.get('ma60', None)

        if ma20 is None or ma60 is None or pd.isna(ma20) or pd.isna(ma60):
            return "Unknown Market"

        # 条件1：双均线全部多头，且价格在趋势上方 → 牛市
        if ma20 > ma60 and current['close'] > ma20:
            return "Bull Market"

        # 条件2：MA20已下穿MA60，且价格需要连续 confirm_days 天都在 MA60 下方
        if ma20 < ma60:
            # 取最后 confirm_days 行验证
            recent = index_df.tail(confirm_days)
            if 'ma60' in recent.columns:
                bear_confirmed = (recent['close'] < recent['ma60']).all()
            else:
                bear_confirmed = current['close'] < ma60
            if bear_confirmed:
                return "Bear Market"

        # 其他情况：震荡/中性区间（均线粘合、价格夹在均线之间）
        return "Volatile Market"

    def filter_signal(self, score: float, index_df: pd.DataFrame, code: str = "", dynamic_threshold: float | None = None) -> tuple[bool, str]:
        """
        根据大盘状态过滤个股信号
        返回: (是否建议买入, 状态描述)
        """
        if index_df.empty:
            # 默认中性
            return score > 0.6, "Unknown Market"

        market_status = self._detect_market_regime(index_df)

        if market_status == "Bull Market":
            threshold = dynamic_threshold if dynamic_threshold is not None else settings.TICKER_BULL_THRESHOLDS.get(code)
            if threshold is None:
                if code in settings.AGGRESSIVE_TICKERS:
                    threshold = settings.BULL_AGGRESSIVE_THRESHOLD
                else:
                    threshold = settings.BULL_BASE_THRESHOLD
            if code in settings.AGGRESSIVE_TICKERS or code in settings.TICKER_BULL_THRESHOLDS:
                market_status += " (Aggressive)"
            is_buy = score >= threshold

        elif market_status == "Bear Market":
            # 熊市：极致防御模式，只有极高分才允许左侧抄底
            is_buy = score >= settings.BEAR_THRESHOLD

        else:
            # 震荡市：采用中性偏保守阈值
            threshold = dynamic_threshold if dynamic_threshold is not None else settings.VOLATILE_THRESHOLD
            is_buy = score >= threshold

        return is_buy, market_status

class RiskManager:
    """
    风控模块：计算止损位
    """
    def calculate_stops(self, df: pd.DataFrame, entry_price: float = None, code: str = "") -> dict:
        """
        计算初始止损和移动止损
        """
        if df.empty:
            return {}
            
        current = df.iloc[-1]
        atr = current['atr']
        close = current['close']
        
        # 如果没有指定入场价，假设按当前收盘价买入
        entry = entry_price if entry_price else close
        
        # 确定 ATR 乘数
        multiplier = settings.ATR_MULTIPLIER
        if code in settings.AGGRESSIVE_TICKERS:
            multiplier = settings.ATR_MULTIPLIER_AGGRESSIVE
        
        # 初始止损：入场价 - N * ATR
        initial_stop = entry - (multiplier * atr)
        
        # 吊灯止损 (Chandelier Exit): 最高价 - N * ATR
        # 【优化性能 + 优化3】先取最后 N 行切片再求 max，避免对全历史做 rolling
        lookback = settings.EXIT_LOOKBACK_PERIOD
        recent_high = df['high'].tail(lookback).max()
        trailing_stop = recent_high - (multiplier * atr)
        
        return {
            "current_price": close,
            "atr": round(atr, 3),
            "initial_stop_loss": round(initial_stop, 3),
            "trailing_stop_loss": round(trailing_stop, 3),
            "risk_per_share": round(entry - initial_stop, 3)
        }
