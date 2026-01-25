import pandas as pd
import numpy as np
from datetime import datetime
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.backtest.backtester import Backtester

from datetime import datetime, timedelta

def main():
    print("🧠 Starting Model Training (XGBoost)...")
    
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
    
    # 动态设定分割点：训练集使用直到 3 个月前的数据
    # 这样可以让模型学习到 2025-01-01 到 2025-10-25 期间的市场特征
    split_date_obj = datetime.now() - timedelta(days=90)
    split_date = split_date_obj.strftime("%Y%m%d")
    print(f"📅 Split Date: {split_date} (Train before, Test after)")
    
    train_df = full_df[full_df['trade_date'].astype(str) < split_date]
    # test_df = full_df[full_df['trade_date'] >= split_date] # 仅用于统计
    
    # ==========================================
    # 模型: XGBoost
    # ==========================================
    print("\n🚀 Training XGBoost...")
    xgb_model = XGBoostModel(model_path="data/xgb_model.json")
    xgb_model.train(train_df)
    
    print("📈 Backtesting XGBoost...")
    xgb_results = run_backtest(dataset, xgb_model, backtester, split_date)
    xgb_avg_ret = np.mean([r['total_return'] for r in xgb_results])
    print(f"XGB Average Return: {xgb_avg_ret*100:.2f}%")
    
    # 找 XGB 最佳
    xgb_best = max(xgb_results, key=lambda x: x['total_return'])
    print(f"Best Ticker: {xgb_best['name']} ({xgb_best['total_return']*100:.2f}%)")

    print("✅ Saving XGBoost model...")
    xgb_model.save_model()

def run_backtest(dataset, model, backtester, split_date):
    results = []
    for code, df in dataset.items():
        test_part = df[df['trade_date'].astype(str) >= split_date].copy()
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
