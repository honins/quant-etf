from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class BaseModel(ABC):
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> float:
        """
        输入包含技术指标的 DataFrame (最后一行是当天)
        输出 0~1 的置信度分数
        """
        pass

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据，默认直接返回。子类可覆盖。
        """
        return df

class RuleBasedModel(BaseModel):
    """
    基于规则权重的透明评分模型
    """
    def predict(self, df: pd.DataFrame) -> float:
        if df.empty or len(df) < 30:
            return 0.0
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 数据时效性检查：如果数据超过5天未更新，则视为无效
        last_date_str = str(current['trade_date']) # 假设格式 YYYYMMDD
        try:
            last_date = datetime.strptime(last_date_str, "%Y%m%d")
            if (datetime.now() - last_date).days > 5:
                print(f"Warning: Data outdated ({last_date_str}). Score set to 0.")
                return 0.0
        except Exception:
            pass # 日期解析失败忽略
        
        score = 0
        total_weight = 0
        
        # 1. RSI (权重 30) - 寻找超卖反弹
        # RSI < 30: 极度超卖 (满分)
        # 30 < RSI < 50: 弱势区间 (半分)
        w_rsi = 30
        total_weight += w_rsi
        if current['rsi_14'] < 30:
            score += w_rsi
        elif current['rsi_14'] < 50:
            score += w_rsi * 0.5
            
        # 2. 均线趋势 (权重 30) - 价格在20日均线上方
        w_trend = 30
        total_weight += w_trend
        if current['close'] > current['ma20']:
            score += w_trend
            
        # 3. MACD 金叉 (权重 20)
        w_macd = 20
        total_weight += w_macd
        # 今天 DIF > DEA 且 昨天 DIF <= DEA
        if current['macd'] > current['macdsignal'] and prev['macd'] <= prev['macdsignal']:
            score += w_macd
        elif current['macd'] > current['macdsignal']: # 保持金叉状态
            score += w_macd * 0.5
            
        # 4. 放量上涨 (权重 20)
        w_vol = 20
        total_weight += w_vol
        if current['close'] > prev['close'] and current['vol'] > current['ma5_vol']: # 需要先计算 vol ma5
             score += w_vol
             
        # 归一化到 0~1
        final_score = score / total_weight
        return round(final_score, 2)

    def prepare_data(self, df):
        # 补充一些模型特有的临时计算
        df['ma5_vol'] = df['vol'].rolling(5).mean()
        return df
