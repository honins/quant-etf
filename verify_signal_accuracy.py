
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel

def main():
    print("🔍 Analyzing Signal Accuracy (Future 5D High > 2%)...")
    
    # 1. 准备数据
    # 取最近 6 个月的数据进行统计
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
    
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    
    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("❌ Model not found.")
        return
        
    ticker_list = tickers.get_ticker_list()
    all_results = []
    
    print(f"Fetching data since {start_date}...")
    
    for code in ticker_list:
        df = data_manager.update_and_get_data(code)
        if df.empty:
            continue
            
        # 计算指标
        df = feature_eng.calculate_technical_indicators(df)
        df = df.dropna()
        
        # 截取最近 6 个月
        df = df[df['trade_date'].astype(str) >= start_date].copy()
        
        if len(df) < 10:
            continue
            
        # 预测分数
        probs = model.predict_batch(df)
        df['score'] = probs
        
        # 计算真实标签: 未来 5 天最高价涨幅
        # 方法: 取 t+1 到 t+5 的最高价
        future_highs = []
        for i in range(1, 6):
            future_highs.append(df['high'].shift(-i))
            
        # 每一行的未来5天最高价
        future_max_high = pd.concat(future_highs, axis=1).max(axis=1)
        
        # 计算相对于当前收盘价的涨幅
        # 假设我们在收盘时做决策，并在收盘价买入(近似)或者第二天开盘买入
        # 这里为了计算"信号质量"，通常用 Close(t) 作为基准比较合理，
        # 意思是"如果在该信号出现时买入(收盘价)，未来5天最高能涨多少"
        df['max_return_5d'] = future_max_high / df['close'] - 1
        
        # 过滤掉最后 5 天（因为没有未来数据，全是 NaN）
        df = df.dropna(subset=['max_return_5d'])
        
        all_results.append(df[['trade_date', 'score', 'max_return_5d']])
        
    if not all_results:
        print("No data.")
        return
        
    full_df = pd.concat(all_results)
    
    # 统计分桶
    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    labels = ['<0.50', '0.50-0.55', '0.55-0.60', '0.60-0.65', '0.65-0.70', '0.70-0.75', '0.75-0.80', '>0.80']
    
    full_df['bin'] = pd.cut(full_df['score'], bins=bins, labels=labels, right=False)
    
    print("\n📊 Signal Accuracy Report (Target: Future 5-Day Max Return > 2%)")
    print(f"Data Period: Last 6 Months")
    print("-" * 80)
    print(f"{'Score Range':<15} {'Count':<10} {'Hits (>2%)':<15} {'Win Rate':<10} {'Avg MaxRet':<10}")
    print("-" * 80)
    
    # Group by bin
    grouped = full_df.groupby('bin', observed=False)
    
    for name, group in grouped:
        count = len(group)
        if count == 0:
            continue
            
        hits = len(group[group['max_return_5d'] > 0.02])
        win_rate = hits / count
        avg_ret = group['max_return_5d'].mean()
        
        print(f"{name:<15} {count:<10} {hits:<15} {win_rate*100:6.1f}%     {avg_ret*100:6.2f}%")
        
    print("-" * 80)
    print(f"Total Samples: {len(full_df)}")
    
    # 额外统计：如果阈值是 0.65，整体表现如何
    high_conf = full_df[full_df['score'] >= 0.65]
    if len(high_conf) > 0:
        hits = len(high_conf[high_conf['max_return_5d'] > 0.02])
        print(f"\n💡 Summary for Score >= 0.65:")
        print(f"   Sample Size: {len(high_conf)}")
        print(f"   Win Rate:    {hits/len(high_conf)*100:.1f}%")
        print(f"   Avg Return:  {high_conf['max_return_5d'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()
