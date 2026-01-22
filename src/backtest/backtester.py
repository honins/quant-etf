import pandas as pd
import numpy as np
from src.strategy.logic import RiskManager
from config.settings import settings

class Backtester:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager()

    def run(self, df: pd.DataFrame, probs: np.ndarray, threshold=0.6) -> dict:
        """
        执行回测
        df: 包含行情数据的 DataFrame
        probs: 模型预测的概率数组 (对应每一天)
        """
        cash = self.initial_capital
        position = 0 # 持仓数量
        equity_curve = []
        
        # 交易记录
        trades = []
        
        # 移动止损价
        trailing_stop = 0.0
        entry_price = 0.0
        
        for i in range(len(df) - 1):
            # 这里的 i 对应的是收盘后的决策时间点
            # 交易将在 i+1 开盘时执行 (简化起见，这里假设以 i+1 的 open 价格成交)
            
            date = df.iloc[i]['trade_date']
            close_price = df.iloc[i]['close']
            high_price = df.iloc[i]['high']
            atr = df.iloc[i]['atr']
            score = probs[i]
            
            next_open = df.iloc[i+1]['open']
            next_date = df.iloc[i+1]['trade_date']
            
            current_equity = cash + position * close_price
            equity_curve.append({'date': date, 'equity': current_equity})
            
            # --- 卖出逻辑 (持仓时检查) ---
            if position > 0:
                # 1. 更新移动止损 (Chandelier Exit)
                # 使用过去 22 天最高价 (这里简化，用 current high 近似)
                # 实际交易中应在盘中触发，这里简化为收盘检查
                
                # 检查盘中是否跌破止损 (用 i+1 的 low)
                next_low = df.iloc[i+1]['low']
                
                is_stop_hit = next_low < trailing_stop
                
                if is_stop_hit:
                    # 止损卖出
                    sell_price = min(next_open, trailing_stop) # 如果开盘就跌破，按开盘价卖；否则按止损价卖
                    cash += position * sell_price
                    trades.append({
                        'date': next_date,
                        'action': 'SELL (Stop)',
                        'price': sell_price,
                        'pnl': (sell_price - entry_price) * position
                    })
                    position = 0
                    trailing_stop = 0.0
                    continue # 本日结束
                    
                # 2. 信号止盈/离场 (可选: 如果分数太低也卖出)
                if score < 0.4:
                    sell_price = next_open
                    cash += position * sell_price
                    trades.append({
                        'date': next_date,
                        'action': 'SELL (Signal)',
                        'price': sell_price,
                        'pnl': (sell_price - entry_price) * position
                    })
                    position = 0
                    trailing_stop = 0.0
                    continue

                # 如果没卖，更新移动止损线 (只上不下)
                # 这里简单用最高价 - 2ATR
                new_stop = high_price - (settings.ATR_MULTIPLIER * atr)
                if new_stop > trailing_stop:
                    trailing_stop = new_stop

            # --- 买入逻辑 (空仓时检查) ---
            if position == 0:
                if score >= threshold:
                    # 满仓买入 (简化)
                    buy_price = next_open
                    # 计算可买股数 (向下取整到100)
                    shares = int((cash * 0.99) / buy_price / 100) * 100
                    
                    if shares > 0:
                        position = shares
                        cash -= shares * buy_price
                        entry_price = buy_price
                        # 初始止损
                        trailing_stop = buy_price - (settings.ATR_MULTIPLIER * atr)
                        
                        trades.append({
                            'date': next_date,
                            'action': 'BUY',
                            'price': buy_price,
                            'score': score
                        })

        # 结算最后一天的净值
        final_equity = cash + position * df.iloc[-1]['close']
        
        # 计算指标
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        win_trades = [t for t in trades if t['action'].startswith('SELL') and t['pnl'] > 0]
        loss_trades = [t for t in trades if t['action'].startswith('SELL') and t['pnl'] <= 0]
        
        win_rate = len(win_trades) / (len(win_trades) + len(loss_trades)) if (len(win_trades) + len(loss_trades)) > 0 else 0
        
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "num_trades": len(win_trades) + len(loss_trades),
            "final_equity": final_equity,
            "trades": trades
        }
