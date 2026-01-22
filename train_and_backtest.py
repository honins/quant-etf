import pandas as pd
import numpy as np
from datetime import datetime
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.ml_model import MLModel
from src.models.xgb_model import XGBoostModel
from src.backtest.backtester import Backtester

def main():
    print("🧠 Starting Model Comparison: Random Forest vs XGBoost...")
    
    # 1. 初始化
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    backtester = Backtester()
    
    # 2. 准备数据
    print("📦 Preparing Data...")
    all_data = []
    ticker_list = tickers.get_ticker_list()
    dataset = {} 
    
    for code in ticker_list:
        print(f"Fetching {code}...", end="\r")
        df = data_manager.update_and_get_data(code)
        if df.empty or len(df) < 200:
            continue
        
        df = feature_eng.calculate_technical_indicators(df)
        df = feature_eng.add_labels(df, horizon=5, threshold=0.02)
        df = df.dropna()
        dataset[code] = df
        all_data.append(df)
        
    print(f"\nLoaded {len(all_data)} tickers.")
    
    if not all_data:
        return

    full_df = pd.concat(all_data)
    split_date = '20250101'
    
    train_df = full_df[full_df['trade_date'] < split_date]
    # test_df = full_df[full_df['trade_date'] >= split_date] # 仅用于统计
    
    # ==========================================
    # 模型 1: Random Forest
    # ==========================================
    print("\n🌲 Training Random Forest...")
    rf_model = MLModel(model_path="data/rf_model.pkl")
    rf_model.train(train_df)
    
    print("📈 Backtesting Random Forest...")
    rf_results = run_backtest(dataset, rf_model, backtester, split_date)
    rf_avg_ret = np.mean([r['total_return'] for r in rf_results])
    print(f"RF Average Return: {rf_avg_ret*100:.2f}%")

    # ==========================================
    # 模型 2: XGBoost
    # ==========================================
    print("\n🚀 Training XGBoost...")
    xgb_model = XGBoostModel(model_path="data/xgb_model.pkl")
    xgb_model.train(train_df)
    
    print("📈 Backtesting XGBoost...")
    xgb_results = run_backtest(dataset, xgb_model, backtester, split_date)
    xgb_avg_ret = np.mean([r['total_return'] for r in xgb_results])
    print(f"XGB Average Return: {xgb_avg_ret*100:.2f}%")
    
    # ==========================================
    # 总结对比
    # ==========================================
    print("\n" + "="*60)
    print("🏆 Model Comparison Report (Test Period: 2025-Now)")
    print("="*60)
    print(f"{'Model':<15} {'Avg Return':<15} {'Best Ticker':<15} {'Return':<10}")
    print("-" * 60)
    
    # 找 RF 最佳
    rf_best = max(rf_results, key=lambda x: x['total_return'])
    print(f"{'RandomForest':<15} {rf_avg_ret*100:6.2f}%         {rf_best['name']:<15} {rf_best['total_return']*100:6.2f}%")
    
    # 找 XGB 最佳
    xgb_best = max(xgb_results, key=lambda x: x['total_return'])
    print(f"{'XGBoost':<15} {xgb_avg_ret*100:6.2f}%         {xgb_best['name']:<15} {xgb_best['total_return']*100:6.2f}%")
    
    print("="*60)
    
    # 自动保存最佳模型为 'data/best_model.pkl' (逻辑上只需保留文件即可，main.py需修改以加载对应类)
    if xgb_avg_ret > rf_avg_ret:
        print("✅ XGBoost wins! Saving as default model...")
        xgb_model.save_model()
        # 可以在这里做一个标记文件，或者 main.py 尝试加载两个
    else:
        print("✅ RandomForest wins! Saving as default model...")
        rf_model.save_model()

def run_backtest(dataset, model, backtester, split_date):
    results = []
    for code, df in dataset.items():
        test_part = df[df['trade_date'] >= split_date].copy()
        if len(test_part) < 20:
            continue
        probs = model.predict_batch(test_part)
        res = backtester.run(test_part, probs, threshold=0.6)
        res['code'] = code
        res['name'] = tickers.TICKERS[code]
        results.append(res)
    return results

if __name__ == "__main__":
    main()
