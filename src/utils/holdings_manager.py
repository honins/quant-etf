import json
import pandas as pd
import os
from src.strategy.logic import RiskManager

HOLDINGS_FILE = "config/holdings.json"

class HoldingsManager:
    def __init__(self):
        self.holdings = self._load_holdings()
        self.risk_manager = RiskManager()

    def _load_holdings(self):
        if not os.path.exists(HOLDINGS_FILE):
            return []
        try:
            with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("holdings", [])
        except Exception:
            return []

    def check_holdings(self, data_manager, feature_eng):
        """
        检查所有持仓的卖出信号
        """
        results = []
        if not self.holdings:
            return results

        print(f"\n🎒 Checking Holdings ({len(self.holdings)} positions)...")
        
        for pos in self.holdings:
            code = pos['code']
            buy_price = pos['buy_price']
            
            # 获取数据
            df = data_manager.update_and_get_data(code)
            if df.empty:
                continue
                
            # 计算指标 (主要是 ATR 和 Rolling High)
            df = feature_eng.calculate_technical_indicators(df)
            current_bar = df.iloc[-1]
            current_price = current_bar['close']
            
            # 计算止盈止损位
            risk_data = self.risk_manager.calculate_stops(df, entry_price=buy_price)
            trailing_stop = risk_data['trailing_stop_loss']
            
            # 判定状态
            # 1. 跌破移动止盈线 -> 卖出
            if current_price < trailing_stop:
                status = "🔴 SELL (Stop Hit)"
                action = "卖出止盈/止损"
            else:
                status = "🟢 HOLD"
                action = "继续持有"
                
            # 计算浮动盈亏
            pnl_pct = (current_price - buy_price) / buy_price * 100
            
            results.append({
                "code": code,
                "name": pos.get("name", code),
                "buy_price": buy_price,
                "current_price": current_price,
                "trailing_stop": trailing_stop,
                "pnl_pct": pnl_pct,
                "status": status,
                "action": action
            })
            
        return results
