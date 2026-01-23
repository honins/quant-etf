import pandas as pd
from config.settings import settings

class StrategyFilter:
    """
    策略过滤器：结合大盘趋势调整买入标准
    """
    def filter_signal(self, score: float, index_df: pd.DataFrame, code: str = "") -> tuple[bool, str]:
        """
        根据大盘状态过滤个股信号
        返回: (是否建议买入, 状态描述)
        """
        if index_df.empty:
            # 默认中性
            return score > 0.6, "Unknown Market"
            
        current_idx = index_df.iloc[-1]
        
        # 简单判断牛熊: 指数价格 > 60日均线
        is_bull_market = current_idx['close'] > current_idx['ma60']
        
        market_status = "Bull Market" if is_bull_market else "Bear Market"
        
        if is_bull_market:
            # 牛市：AI 进攻模式
            # 对于高弹性/激进品种，进一步放宽标准
            # 588000.SH (科创50), 159915.SZ (创业板)
            # 512480.SH (半导体), 515030.SH (新能源车)
            aggressive_tickers = [
                "588000.SH", "159915.SZ", 
                "512480.SH", "515030.SH"
            ]
            
            if code in aggressive_tickers:
                threshold = 0.45
                market_status += " (Aggressive)"
            else:
                threshold = 0.6  # 普通标的维持 0.6 以求稳
                
            is_buy = score >= threshold
        else:
            # 熊市：规则防守模式
            # 只有 AI 评分极高 (>= 0.75) 时才允许左侧抄底，否则空仓
            is_buy = score >= 0.75
            
        return is_buy, market_status

class RiskManager:
    """
    风控模块：计算止损位
    """
    def calculate_stops(self, df: pd.DataFrame, entry_price: float = None) -> dict:
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
        
        # 初始止损：入场价 - N * ATR
        initial_stop = entry - (settings.ATR_MULTIPLIER * atr)
        
        # 吊灯止损 (Chandelier Exit): 最高价 - N * ATR
        # 获取过去22天的最高价
        recent_high = df['high'].rolling(22).max().iloc[-1]
        trailing_stop = recent_high - (settings.ATR_MULTIPLIER * atr)
        
        return {
            "current_price": close,
            "atr": round(atr, 3),
            "initial_stop_loss": round(initial_stop, 3),
            "trailing_stop_loss": round(trailing_stop, 3),
            "risk_per_share": round(entry - initial_stop, 3)
        }
