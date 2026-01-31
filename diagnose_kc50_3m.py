
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.settings import settings
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.backtest.backtester import Backtester

def main():
    print("🔍 Diagnosing KC50 ETF (3 Months) with Threshold > 0.65...")
    
    # 1. 设置时间
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    start_date_str = start_date.strftime("%Y%m%d")
    
    # 2. 初始化
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    backtester = Backtester()
    
    # 3. 加载模型
    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("❌ Model not found.")
        return
        
    # 4. 获取数据 (科创50ETF)
    code = '588000.SH'
    df = data_manager.update_and_get_data(code)
    if df.empty:
        print("❌ No data found.")
        return
        
    # 5. 特征工程
    df = feature_eng.calculate_technical_indicators(df)
    df = df.dropna()
    
    # 6. 截取最近3个月
    test_df = df[df['trade_date'].astype(str) >= start_date_str].copy()
    
    if len(test_df) < 10:
        print("❌ Not enough data.")
        return
        
    # 7. 预测
    probs = model.predict_batch(test_df)
    
    # 8. 回测 (多组阈值)
    thresholds = [0.65, 0.60, 0.55]
    
    for th in thresholds:
        print("\n" + "="*50)
        print(f"📊 Backtest Result for {code}")
        print(f"Threshold: > {th}")
        print("="*50)
        
        result = backtester.run(test_df, probs, threshold=th, code=code)
        
        trades = result['trades']
        if not trades:
            print("No trades triggered.")
        else:
            print(f"{'Date':<12} {'Action':<15} {'Price':<10} {'PnL':<10}")
            print("-" * 50)
            total_pnl = 0
            for t in trades:
                pnl_str = f"{t['pnl']:.2f}" if 'pnl' in t else "-"
                print(f"{str(t['date']):<12} {t['action']:<15} {t['price']:<10.3f} {pnl_str:<10}")
                if 'pnl' in t:
                    total_pnl += t['pnl']
            
            # 检查是否持仓
            last_trade = trades[-1]
            if last_trade['action'] == 'BUY':
                current_price = test_df.iloc[-1]['close']
                current_date = test_df.iloc[-1]['trade_date']
                buy_price = last_trade['price']
                hold_return = (current_price - buy_price) / buy_price
                print("-" * 50)
                print(f"🔄 HOLDING till {current_date}")
                print(f"   Current Price: {current_price:.3f}")
                print(f"   Floating PnL:  {hold_return*100:.2f}%")

            print("-" * 50)
            print(f"Total Return: {result['total_return']*100:.2f}%")
            print(f"Win Rate:     {result['win_rate']*100:.1f}%")

if __name__ == "__main__":
    main()
