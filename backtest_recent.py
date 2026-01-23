import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter
from src.backtest.backtester import Backtester

def main():
    print("📉 Running Backtest for Last 3 Months (Hybrid Strategy)...")
    
    # 1. 设置回测时间段 (最近3个月)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    start_date_str = start_date.strftime("%Y%m%d")
    print(f"Period: {start_date_str} - {end_date.strftime('%Y%m%d')}")

    # 2. 初始化模块
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    backtester = Backtester()
    strat_filter = StrategyFilter()
    
    # 加载模型
    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("❌ XGBoost model not found. Please train first.")
        return
    else:
        print("✅ XGBoost model loaded.")

    # 3. 获取并处理大盘数据 (用于风控)
    print("📊 Preparing Market Index Data...")
    index_code = '000300.SH'
    index_df = data_manager.update_and_get_data(index_code, is_index=True)
    index_df = feature_eng.calculate_technical_indicators(index_df)
    
    # 将大盘状态映射到日期: {date: is_bull}
    # 逻辑: close > ma60
    market_status_map = {}
    for _, row in index_df.iterrows():
        d = str(row['trade_date'])
        is_bull = row['close'] > row['ma60'] if pd.notnull(row['ma60']) else True
        market_status_map[d] = is_bull

    # 4. 遍历标的进行回测
    results = []
    ticker_list = tickers.get_ticker_list()
    
    for code in ticker_list:
        # 获取数据
        df = data_manager.update_and_get_data(code)
        if df.empty:
            continue
            
        # 截取回测段 (需要多取一点数据用于计算指标)
        # 先计算指标，再截取
        df = feature_eng.calculate_technical_indicators(df)
        df = df.dropna()
        
        # 截取最近3个月
        test_df = df[df['trade_date'].astype(str) >= start_date_str].copy()
        
        if len(test_df) < 10:
            continue
            
        # 预测
        probs = model.predict_batch(test_df)
        
        # 应用混合策略风控: 如果当天是大盘熊市，强制将买入概率置为 0
        adjusted_probs = []
        bear_days = 0
        for i, prob in enumerate(probs):
            trade_date = str(test_df.iloc[i]['trade_date'])
            is_bull = market_status_map.get(trade_date, True) # 默认 True 避免漏数据
            
            if not is_bull:
                # 熊市：仅允许超高分抄底
                if prob >= 0.75:
                    adjusted_probs.append(prob)
                else:
                    adjusted_probs.append(0.0)
                    bear_days += 1
            else:
                # 牛市：对激进标的放宽阈值
                aggressive_tickers = [
                    "588000.SH", "159915.SZ",
                    "512480.SH", "515030.SH"
                ]
                if code in aggressive_tickers:
                     # 激进标的：阈值 0.45
                     # 下面 backtester.run 的阈值统一为 0.45
                     # 所以这里直接传入原始概率即可
                     adjusted_probs.append(prob)
                else:
                     # 普通标的：阈值 0.60
                     # 如果 prob < 0.60，强制置0
                     if prob < 0.60:
                         adjusted_probs.append(0.0)
                     else:
                         adjusted_probs.append(prob)
                
        adjusted_probs = np.array(adjusted_probs)
        
        # 执行回测 (阈值放宽至 0.45)
        res = backtester.run(test_df, adjusted_probs, threshold=0.45)
        
        res['code'] = code
        res['name'] = tickers.TICKERS[code]
        res['bear_days'] = bear_days
        results.append(res)

    # 5. 生成报告
    print("\n" + "="*80)
    print(f"📅 Backtest Report: Last 3 Months ({start_date_str} - Now)")
    print("Strategy: XGBoost Signal + Bear Market Filter (Hybrid)")
    print("="*80)
    
    print(f"{'Name':<12} {'Return':<10} {'WinRate':<10} {'Trades':<8} {'MaxDD':<10} {'BearDays'}")
    print("-" * 80)
    
    total_profit = 0
    total_trades = 0
    winning_trades = 0
    
    for res in results:
        win_rate_str = f"{res['win_rate']*100:.1f}%"
        # 简单估算最大回撤 (这里 backtester 没算，暂空)
        print(f"{res['name']:<12} {res['total_return']*100:6.2f}%    {win_rate_str:<10} {res['num_trades']:<8} {'-':<10} {res['bear_days']}")
        
        total_profit += res['total_return']
        total_trades += res['num_trades']
        # 反推胜场
        # win_rate = wins / trades => wins = rate * trades
        winning_trades += int(round(res['win_rate'] * res['num_trades']))

    print("="*80)
    avg_return = np.mean([r['total_return'] for r in results]) if results else 0
    overall_win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    print(f"Overall Average Return: {avg_return*100:.2f}%")
    print(f"Overall Win Rate:       {overall_win_rate*100:.2f}%")
    print(f"Total Trades:           {total_trades}")
    print("="*80)

if __name__ == "__main__":
    main()
