import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter, RiskManager

def main():
    print("🚀 Starting Detailed Backtest (Last 60 Days)...")
    
    # 1. 初始化
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    model = XGBoostModel()
    if not model.load_model():
        print("❌ Model not found. Please train first.")
        return
        
    strat_filter = StrategyFilter()
    risk_manager = RiskManager()
    
    # 2. 准备时间
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    start_date_str = start_date.strftime("%Y%m%d")
    print(f"📅 Period: {start_date_str} - {end_date.strftime('%Y%m%d')}")
    
    # 3. 获取大盘数据 (用于风控)
    print("📊 Loading Market Index...")
    index_df = data_manager.update_and_get_data('000300.SH', is_index=True)
    index_df = feature_eng.calculate_technical_indicators(index_df)
    
    # 4. 回测循环
    results = []
    trade_logs = {} # {code: [trades]}
    
    ticker_list = tickers.get_ticker_list()
    
    for code in ticker_list:
        name = tickers.TICKERS[code]
        # print(f"Testing {name}...", end="\r")
        
        # 获取数据
        df = data_manager.update_and_get_data(code)
        if df.empty: continue
        
        df = feature_eng.calculate_technical_indicators(df)
        df = df.dropna()
        
        # 截取回测段 (保留一点 buffer 用于计算)
        test_df = df[df['trade_date'].astype(str) >= start_date_str].copy()
        if len(test_df) == 0: continue
        
        # 批量预测
        probs = model.predict_batch(test_df)
        test_df['score'] = probs
        
        # --- 逐日模拟交易 ---
        position = None # {buy_date, buy_price, stop_loss, trailing_stop, highest_price}
        trades = []
        equity = 1.0 # 初始资金净值
        
        for i in range(len(test_df)):
            curr_bar = test_df.iloc[i]
            curr_date = str(curr_bar['trade_date'])
            curr_price = curr_bar['close']
            score = curr_bar['score']
            atr = curr_bar['atr']
            
            # 1. 持仓处理 (卖出逻辑)
            if position:
                # 更新最高价和移动止盈
                if curr_price > position['highest_price']:
                    position['highest_price'] = curr_price
                    # 移动止盈: 最高价 - 2ATR (这里简化用当天的ATR，实盘是用history)
                    # 严谨一点应该用持有期间的 Max High
                    new_stop = position['highest_price'] - 2.0 * atr
                    if new_stop > position['trailing_stop']:
                        position['trailing_stop'] = new_stop
                
                # 检查卖出条件
                # 条件A: 跌破初始止损
                # 条件B: 跌破移动止盈
                sell_reason = None
                if curr_price < position['stop_loss']:
                    sell_reason = "Stop Loss"
                elif curr_price < position['trailing_stop']:
                    sell_reason = "Trailing Stop"
                
                if sell_reason:
                    # 执行卖出
                    pnl = (curr_price - position['buy_price']) / position['buy_price']
                    equity *= (1 + pnl)
                    
                    trades.append({
                        'buy_date': position['buy_date'],
                        'buy_price': position['buy_price'],
                        'sell_date': curr_date,
                        'sell_price': curr_price,
                        'pnl': pnl,
                        'reason': sell_reason,
                        'hold_days': i - position['idx']
                    })
                    position = None # 空仓
                    continue # 卖出当天不买入
            
            # 2. 空仓处理 (买入逻辑)
            if position is None:
                # 查大盘状态
                idx_row = index_df[index_df['trade_date'].astype(str) == curr_date]
                is_bull = True
                if not idx_row.empty:
                    idx_close = idx_row.iloc[0]['close']
                    idx_ma60 = idx_row.iloc[0]['ma60']
                    is_bull = idx_close > idx_ma60
                
                # 策略判断
                # 逻辑复刻 StrategyFilter
                threshold = 0.45 if is_bull else 0.75
                
                if score >= threshold:
                    # 买入!
                    # 计算止损位
                    stop_loss = curr_price - 2.0 * atr
                    trailing_stop = stop_loss # 初始移动止盈 = 初始止损
                    
                    position = {
                        'buy_date': curr_date,
                        'buy_price': curr_price,
                        'stop_loss': stop_loss,
                        'trailing_stop': trailing_stop,
                        'highest_price': curr_price,
                        'idx': i
                    }
        
        # 结束时如果还持仓，按最后一天的价格强平计算收益(仅用于统计)
        if position:
            pnl = (test_df.iloc[-1]['close'] - position['buy_price']) / position['buy_price']
            equity *= (1 + pnl)
            trades.append({
                'buy_date': position['buy_date'],
                'buy_price': position['buy_price'],
                'sell_date': 'Holding',
                'sell_price': test_df.iloc[-1]['close'],
                'pnl': pnl,
                'reason': 'End of Test',
                'hold_days': len(test_df) - position['idx']
            })
            
        trade_logs[code] = trades
        results.append({
            'code': code,
            'name': name,
            'return': equity - 1,
            'trades': len(trades)
        })

    # 5. 生成详细报告
    generate_report(results, trade_logs, start_date_str)

def generate_report(results, trade_logs, start_date):
    results.sort(key=lambda x: x['return'], reverse=True)
    
    content = []
    content.append(f"# 🔙 详细回测报告 (近2个月)\n")
    content.append(f"**开始日期**: {start_date}\n")
    content.append(f"**策略**: AI Scoring + Dynamic Threshold + ATR Trailing Stop\n")
    content.append("\n---\n")
    
    content.append("## 🏆 收益总览\n")
    content.append("| 代码 | 名称 | 区间收益 | 交易次数 |\n")
    content.append("|---|---|---|---|\n")
    
    for res in results:
        ret_str = f"{res['return']*100:+.2f}%"
        row = f"| {res['code']} | {res['name']} | **{ret_str}** | {res['trades']} |"
        content.append(row + "\n")
        
    content.append("\n---\n")
    content.append("## 📝 逐笔交易明细\n")
    
    for res in results:
        code = res['code']
        name = res['name']
        trades = trade_logs.get(code, [])
        
        if not trades:
            continue
            
        content.append(f"### {name} ({code})\n")
        content.append(f"**总收益**: {res['return']*100:+.2f}%\n\n")
        content.append("| 买入日期 | 买入价 | 卖出日期 | 卖出价 | **单笔盈亏** | 持仓天数 | 卖出原因 |\n")
        content.append("|---|---|---|---|---|---|---|\n")
        
        for t in trades:
            pnl_color = "🔴" if t['pnl'] < 0 else "🟢"
            pnl_str = f"{pnl_color} {t['pnl']*100:+.2f}%"
            row = f"| {t['buy_date']} | {t['buy_price']:.3f} | {t['sell_date']} | {t['sell_price']:.3f} | {pnl_str} | {t['hold_days']} | {t['reason']} |"
            content.append(row + "\n")
        content.append("\n")
        
    # 保存
    report_path = "reports/backtest_detail_2m.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(content))
        
    print(f"\n✅ Report generated: {report_path}")
    print(f"Check the report for daily details!")

if __name__ == "__main__":
    main()
