import pandas as pd
from datetime import datetime, timedelta
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel

def main():
    print("🔍 Debugging Scores for broad-based ETFs...")
    
    # 关注的标的
    target_codes = ["510050.SH", "510300.SH", "510500.SH", "588000.SH"]
    
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    model = XGBoostModel()
    model.load_model()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    start_date_str = start_date.strftime("%Y%m%d")
    
    print(f"{'Code':<10} {'Name':<10} {'Max Score':<10} {'Avg Score':<10} {'Days > 0.5'}")
    print("-" * 60)
    
    for code in target_codes:
        df = data_manager.update_and_get_data(code)
        if df.empty: continue
        
        df = feature_eng.calculate_technical_indicators(df)
        df = df.dropna()
        test_df = df[df['trade_date'].astype(str) >= start_date_str].copy()
        
        if test_df.empty: continue
        
        probs = model.predict_batch(test_df)
        
        max_score = probs.max()
        avg_score = probs.mean()
        high_score_days = (probs > 0.5).sum()
        
        name = tickers.TICKERS.get(code, code)
        print(f"{code:<10} {name:<10} {max_score:.4f}     {avg_score:.4f}     {high_score_days}")

if __name__ == "__main__":
    main()
