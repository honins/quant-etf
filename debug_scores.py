import pandas as pd
from datetime import datetime, timedelta
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel

def main():
    print("🔍 Debugging Scores for broad-based ETFs...")
    
    # 加载模型
    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("Model not found")
        return
    
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    start_date_str = start_date.strftime("%Y%m%d")
    
    # 收集所有历史评分
    all_scores = []
    
    ticker_list = tickers.get_ticker_list()
    
    for code in ticker_list:
        df = data_manager.update_and_get_data(code)
        if df.empty: continue
        
        df = feature_eng.calculate_technical_indicators(df)
        df = df.dropna()
        test_df = df[df['trade_date'].astype(str) >= start_date_str].copy()
        
        if test_df.empty: continue
        
        probs = model.predict_batch(test_df)
        
        name = tickers.TICKERS.get(code, code)
        for date, score, price in zip(test_df['trade_date'], probs, test_df['close']):
            all_scores.append({
                'date': date,
                'code': code,
                'name': name,
                'score': float(score),
                'price': price
            })
            
    # 排序
    all_scores.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*80)
    print(f"🏆 Top 10 Highest Scores (Last 3 Months)")
    print("="*80)
    print(f"{'Date':<12} {'Name':<12} {'Code':<10} {'Score':<8} {'Price'}")
    print("-" * 80)
    for item in all_scores[:10]:
        print(f"{item['date']:<12} {item['name']:<12} {item['code']:<10} {item['score']:.4f}   {item['price']:.3f}")
        
    print("\n" + "="*80)
    print(f"🧊 Top 10 Lowest Scores (Last 3 Months)")
    print("="*80)
    print(f"{'Date':<12} {'Name':<12} {'Code':<10} {'Score':<8} {'Price'}")
    print("-" * 80)
    for item in all_scores[-10:]:
        print(f"{item['date']:<12} {item['name']:<12} {item['code']:<10} {item['score']:.4f}   {item['price']:.3f}")

if __name__ == "__main__":
    main()
