import pandas as pd

class TechnicalExplainer:
    """
    负责将冷冰冰的技术指标翻译成人类可读的点评
    """
    
    @staticmethod
    def explain(df: pd.DataFrame) -> list:
        if df.empty:
            return []
            
        current = df.iloc[-1]
        reasons = []
        
        # 1. 均线形态 (Trend)
        try:
            close = current.get('close', 0)
            ma5 = current.get('ma5', 0)
            ma20 = current.get('ma20', 0)
            ma60 = current.get('ma60', 0)
            
            if ma5 and ma20 and ma60:
                if close > ma5 > ma20 > ma60:
                    reasons.append("📈 **均线多头**: 短中长期均线顺次向上，上升趋势强劲。")
                elif close > ma20 and ma20 > ma60:
                    reasons.append("↗️ **趋势向好**: 站稳20日生命线，中期趋势向上。")
                elif close < ma20:
                    reasons.append("📉 **趋势走弱**: 跌破20日均线，短期承压。")
        except Exception:
            pass
            
        # 2. RSI (Momentum)
        try:
            rsi = current.get('rsi_14', 50)
            if rsi > 80:
                reasons.append("🔥 **极度超买**: RSI>80，情绪过热，谨防回调。")
            elif rsi > 70:
                reasons.append("⚠️ **超买区**: RSI>70，上涨动能强但有回调风险。")
            elif rsi < 30:
                reasons.append("🧊 **超卖区**: RSI<30，情绪冰点，存在反弹需求。")
            elif 50 <= rsi <= 70:
                reasons.append("💪 **强势区**: RSI在50-70之间，多头主导。")
        except Exception:
            pass
            
        # 3. MACD
        try:
            macd = current.get('macd', 0)
            signal = current.get('macdsignal', 0)
            hist = current.get('macdhist', 0)
            prev_hist = df.iloc[-2]['macdhist'] if len(df) > 1 else 0
            
            if hist > 0 and hist > prev_hist:
                reasons.append("🚀 **动能增强**: MACD红柱放大，上涨加速。")
            elif hist > 0 and hist < prev_hist:
                reasons.append("🐢 **动能减弱**: MACD红柱缩短，上涨乏力。")
            elif macd > signal:
                reasons.append("✅ **金叉状态**: MACD保持金叉，多头占优。")
        except Exception:
            pass
            
        # 4. 成交量 (Volume)
        try:
            vol = current.get('vol', 0)
            # 计算简单的 ma5_vol
            ma5_vol = df['vol'].rolling(5).mean().iloc[-1] if len(df) >= 5 else vol
            
            if vol > ma5_vol * 1.5:
                reasons.append("📢 **放量**: 今日成交量明显放大(>1.5倍均量)，资金活跃。")
        except Exception:
            pass
        
        # 5. 价格位置 (Bollinger)
        try:
            close = current.get('close', 0)
            upper = current.get('upper', 0)
            lower = current.get('lower', 0)
            
            if upper and close > upper:
                reasons.append("⚡ **突破上轨**: 股价突破布林带上轨，极度强势。")
            elif lower and close < lower:
                reasons.append("💧 **跌破下轨**: 股价跌破布林带下轨，极度弱势。")
        except Exception:
            pass
            
        return reasons
