from datetime import datetime

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from config import tickers
from config.settings import settings

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

    def is_data_stale(self, trade_date: object) -> bool:
        last_date_str = str(trade_date)
        try:
            last_date = datetime.strptime(last_date_str, "%Y%m%d")
            return (datetime.now() - last_date).days > 5
        except Exception:
            return False

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_data(df.copy())
        scores = [self.predict(prepared.iloc[: i + 1]) for i in range(len(prepared))]
        return np.asarray(scores, dtype=float)

class RuleBasedModel(BaseModel):
    """
    基于规则权重的透明评分模型
    """
    def predict(self, df: pd.DataFrame) -> float:
        if df.empty or len(df) < 30:
            return 0.0
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if self.is_data_stale(current["trade_date"]):
            print(f"Warning: Data outdated ({current['trade_date']}). Score set to 0.")
            return 0.0
        
        return self._score_window(current, prev)

    def prepare_data(self, df):
        # 补充一些模型特有的临时计算
        df['ma5_vol'] = df['vol'].rolling(5).mean()
        return df

    def _score_window(self, current: pd.Series, prev: pd.Series) -> float:
        ticker = str(current.get('ts_code', '') or '')
        category = tickers.get_ticker_category(ticker) if ticker else "unknown"
        is_wide_index = ticker in set(tickers.CORE_TRADE_TICKERS[:6]) if ticker else False
        market_trend_strength = float(current.get('market_trend_strength', 0.0) or 0.0)
        market_above_ma20 = float(current.get('market_above_ma20', 0.0) or 0.0)
        market_return_20d = float(current.get('market_return_20d', 0.0) or 0.0)
        market_volatility_20d = float(current.get('market_volatility_20d', 0.0) or 0.0)
        relative_strength = float(current.get('rs_20d', 0.0) or 0.0)
        breakout_20 = float(current.get('breakout_20', 0.0) or 0.0)
        drawdown_20 = float(current.get('drawdown_20', 0.0) or 0.0)
        atr_pct = float(current.get('atr_pct', 0.0) or 0.0)
        ma20_slope_5 = float(current.get('ma20_slope_5', 0.0) or 0.0)
        ret_5 = float(current.get('ret_5', 0.0) or 0.0)

        if market_trend_strength < -0.02 and market_above_ma20 <= 0:
            return 0.0
        if market_return_20d < -0.03 and relative_strength <= 0:
            return 0.0
        if market_volatility_20d > 0.45 and breakout_20 <= 0:
            return 0.0
        if category == "satellite" and market_trend_strength < 0 and relative_strength <= 0:
            return 0.0
        if category == "satellite" and atr_pct > 0.04 and breakout_20 <= 0:
            return 0.0
        if category == "core" and drawdown_20 < -0.08 and market_return_20d < 0:
            return 0.0
        if is_wide_index and (market_trend_strength < 0 or market_return_20d < 0) and breakout_20 <= 0:
            return 0.0
        if is_wide_index and ma20_slope_5 < 0 and ret_5 < 0:
            return 0.0

        score = 0.0
        total_weight = 0.0

        w_rsi = 15.0
        total_weight += w_rsi
        if current['rsi_14'] < 35:
            score += w_rsi * 0.7
        elif 35 <= current['rsi_14'] <= 62:
            score += w_rsi
        elif current['rsi_14'] < 72:
            score += w_rsi * 0.5

        w_trend = 20.0
        total_weight += w_trend
        if current['close'] > current['ma20']:
            score += w_trend * 0.7
        if current['ma20'] > current['ma60']:
            score += w_trend * 0.3

        w_macd = 15.0
        total_weight += w_macd
        if current['macd'] > current['macdsignal'] and prev['macd'] <= prev['macdsignal']:
            score += w_macd
        elif current['macd'] > current['macdsignal']:
            score += w_macd * 0.6

        w_vol = 10.0
        total_weight += w_vol
        if current['close'] > prev['close'] and current['vol'] > current['ma5_vol']:
            score += w_vol
        elif current['vol_ratio'] > 1.0:
            score += w_vol * 0.5

        w_breakout = 10.0
        total_weight += w_breakout
        if breakout_20 > 0:
            score += w_breakout
        elif float(current.get('drawdown_20', 0.0) or 0.0) > -0.03:
            score += w_breakout * 0.4

        w_relative = 10.0
        total_weight += w_relative
        if float(current.get('rs_20d', 0.0) or 0.0) > 0:
            score += w_relative
        elif float(current.get('rel_vol', 1.0) or 1.0) < 1.1:
            score += w_relative * 0.4

        w_regime = 20.0
        total_weight += w_regime
        if market_trend_strength > 0 and market_above_ma20 > 0:
            score += w_regime * 0.8
        if market_return_20d > 0:
            score += w_regime * 0.2
        if market_volatility_20d > 0.35:
            score -= w_regime * 0.35

        if relative_strength < -0.03:
            score -= 8.0
        if breakout_20 < -0.02:
            score -= 6.0
        if drawdown_20 < -0.05:
            score -= 5.0
        if atr_pct > 0.03:
            score -= 4.0

        if category == "core":
            if market_trend_strength > 0:
                score += 4.0
            if relative_strength > -0.01:
                score += 3.0
            if ma20_slope_5 < 0:
                score -= 5.0
            if ret_5 < 0:
                score -= 4.0
        elif category == "satellite":
            if breakout_20 <= 0:
                score -= 5.0
            if market_volatility_20d > 0.30:
                score -= 5.0

        if is_wide_index and breakout_20 <= 0:
            score -= 5.0
        if is_wide_index and market_above_ma20 <= 0:
            score -= 6.0
        if is_wide_index and market_trend_strength <= 0:
            score -= 5.0

        if ticker.endswith('.SZ') and market_volatility_20d > 0.30:
            score -= 5.0

        final_score = max(0.0, min(score / total_weight, 1.0)) if total_weight > 0 else 0.0
        if category == "satellite":
            final_score *= 0.88
        elif category == "core":
            final_score *= 1.03
        if is_wide_index:
            final_score *= 0.9
        if ticker and ticker.endswith('.SZ'):
            final_score *= 0.92
        return round(final_score, 2)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_data(df.copy())
        if prepared.empty:
            return np.zeros(0, dtype=float)

        latest_trade_date = prepared.iloc[-1]["trade_date"] if "trade_date" in prepared.columns else None
        apply_stale_guard = self.is_data_stale(latest_trade_date)

        scores = []
        for i in range(len(prepared)):
            window = prepared.iloc[: i + 1]
            if not apply_stale_guard:
                current = window.iloc[-1]
                prev = window.iloc[-2] if len(window) >= 2 else current
                if len(window) < 30:
                    scores.append(0.0)
                    continue
                scores.append(self._score_window(current, prev))
                continue

            scores.append(self.predict(window))

        return np.asarray(scores, dtype=float)
