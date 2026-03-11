import yaml
import os
from datetime import datetime
from src.strategy.logic import RiskManager

HOLDINGS_FILE = "config/holdings.yml"

class HoldingsManager:
    def __init__(self):
        self.holdings = self._load_holdings()
        self.risk_manager = RiskManager()

    def _load_holdings(self):
        if not os.path.exists(HOLDINGS_FILE):
            return []
        try:
            with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
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
            
            # 计算加权平均成本和总股数
            transactions = pos.get('transactions', [])
            
            # 兼容旧格式 (如果没有 transactions 但有 buy_price)
            if not transactions and 'buy_price' in pos:
                 transactions = [{
                     'price': pos['buy_price'],
                     'date': pos.get('buy_date', ''),
                     'shares': pos.get('shares', 0)
                 }]
            
            if not transactions:
                continue
                
            total_shares = 0
            total_cost = 0.0
            first_buy_date = None
            
            for txn in transactions:
                shares = txn.get('shares', 0)
                price = txn.get('price', 0.0)
                date_str = str(txn.get('date', ''))
                
                total_shares += shares
                total_cost += shares * price
                
                # 寻找首次买入日期
                if date_str:
                    try:
                        txn_date = datetime.strptime(date_str, "%Y%m%d")
                        if first_buy_date is None or txn_date < first_buy_date:
                            first_buy_date = txn_date
                    except ValueError:
                        pass
                        
            avg_cost = total_cost / total_shares if total_shares > 0 else 0.0
            buy_price = avg_cost # 使用平均成本作为买入价
            
            # 计算持仓天数 (基于首次买入)
            days_held = -1
            if first_buy_date:
                days_held = (datetime.now() - first_buy_date).days

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
                "action": action,
                "days_held": days_held
            })
            
        return results
