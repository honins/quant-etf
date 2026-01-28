
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel

def main():
    print("🔍 Verifying Alpha: Model Performance vs Market Environment...")
    
    # 1. 准备数据 (最近 6 个月)
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
    
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("❌ Model not found.")
        return

    # 2. 获取大盘数据 (沪深300) 作为基准
    print("📊 Fetching Market Index (000300.SH)...")
    index_df = data_manager.update_and_get_data('000300.SH', is_index=True)
    if index_df.empty:
        print("❌ Index data not found.")
        return
    
    # 计算大盘未来 5 天涨跌幅
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    # 大盘我们看整体趋势，用收盘价比较合理
    index_df['market_ret_5d'] = index_df['close'].shift(-5) / index_df['close'] - 1
    
    # 将大盘数据转为字典方便查询: {date: market_ret}
    market_ret_map = index_df.set_index('trade_date')['market_ret_5d'].to_dict()
    
    # 3. 获取所有个股信号
    ticker_list = tickers.get_ticker_list()
    all_signals = []
    
    print(f"Fetching tickers since {start_date}...")
    for code in ticker_list:
        df = data_manager.update_and_get_data(code)
        if df.empty:
            continue
            
        df = feature_eng.calculate_technical_indicators(df)
        df = df.dropna()
        df = df[df['trade_date'].astype(str) >= start_date].copy()
        
        if len(df) < 10:
            continue
            
        # 预测
        probs = model.predict_batch(df)
        df['score'] = probs
        
        # 个股未来 5 天最高涨幅
        future_highs = []
        for i in range(1, 6):
            future_highs.append(df['high'].shift(-i))
        future_max_high = pd.concat(future_highs, axis=1).max(axis=1)
        df['stock_max_ret_5d'] = future_max_high / df['close'] - 1
        
        df = df.dropna(subset=['stock_max_ret_5d'])
        
        # 关联大盘表现
        df['market_ret_5d'] = df['trade_date'].map(market_ret_map)
        
        # 只保留分数较高 和 较低 的样本进行对比
        # High Score: > 0.60
        # Low Score: < 0.50
        all_signals.append(df[['trade_date', 'score', 'stock_max_ret_5d', 'market_ret_5d']])

    if not all_signals:
        print("No signals found.")
        return

    full_df = pd.concat(all_signals)
    full_df = full_df.dropna(subset=['market_ret_5d']) # 确保有大盘数据
    
    # 4. 划分市场环境
    # Bear Market: 大盘未来5天跌幅 < -1%
    # Neutral: -1% <= 跌幅 <= 1%
    # Bull Market: 大盘未来5天涨幅 > 1%
    
    def get_market_env(ret):
        if ret < -0.01: return '📉 Bear (Index < -1%)'
        if ret > 0.01: return '📈 Bull (Index > 1%)'
        return '⚖️ Shock (-1% ~ 1%)'
        
    full_df['market_env'] = full_df['market_ret_5d'].apply(get_market_env)
    
    # 5. 统计不同环境下，高分信号的表现
    # 我们关注：在熊市里，高分信号是否还能赚钱？
    
    high_score_df = full_df[full_df['score'] >= 0.65]
    
    print("\n🧐 Truth Test: Does the model work in Bear Markets?")
    print("=" * 80)
    print(f"Analyzing High Confidence Signals (Score >= 0.65) by Market Environment")
    print("-" * 80)
    print(f"{'Market Env':<25} {'Signals':<10} {'Win Rate (>2%)':<18} {'Avg Stock Ret':<15} {'Avg Index Ret':<15}")
    print("-" * 80)
    
    grouped = high_score_df.groupby('market_env', observed=False)
    
    for name, group in grouped:
        count = len(group)
        hits = len(group[group['stock_max_ret_5d'] > 0.02])
        win_rate = hits / count if count > 0 else 0
        avg_stock_ret = group['stock_max_ret_5d'].mean()
        avg_index_ret = group['market_ret_5d'].mean()
        
        print(f"{name:<25} {count:<10} {win_rate*100:6.1f}%            {avg_stock_ret*100:6.2f}%          {avg_index_ret*100:6.2f}%")
        
    print("-" * 80)
    print("\n💡 Interpretation:")
    print("1. If Win Rate is high (>70%) even in 'Bear' markets, the model has REAL Alpha.")
    print("2. If Win Rate drops significantly in 'Bear' markets, the model relies on Beta.")

if __name__ == "__main__":
    main()
