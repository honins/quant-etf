import pandas as pd
from datetime import datetime
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter

def main():
    print("🔍 Diagnosing KC50 (588000.SH) Missed Opportunity...")
    
    # 目标：分析 2025-12-01 至今的数据 (假设用户指的是最近的 12/17，即 2025年)
    # 注意：当前环境日期是 2026-01-23，所以用户说的 12/17 应该是 2025-12-17
    start_analyze_date = "20251201"
    
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    
    # 加载模型
    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("Model not found")
        return

    # 1. 获取大盘数据 (沪深300) 用于风控判断
    print("📊 Loading Market Data...")
    index_df = data_manager.update_and_get_data('000300.SH', is_index=True)
    index_df = feature_eng.calculate_technical_indicators(index_df)
    
    # 2. 获取科创50数据
    print("📈 Loading KC50 Data...")
    kc50_code = "588000.SH"
    df = data_manager.update_and_get_data(kc50_code)
    df = feature_eng.calculate_technical_indicators(df)
    df = df.dropna()
    
    # 截取分析段
    target_df = df[df['trade_date'].astype(str) >= start_analyze_date].copy()
    
    # 3. 逐日分析
    print("\n📅 Daily Analysis (Threshold=0.6)")
    print(f"{'Date':<10} {'Close':<8} {'PctChg':<8} {'AI Score':<10} {'Market':<10} {'Result'}")
    print("-" * 70)
    
    for _, row in target_df.iterrows():
        date_str = str(row['trade_date'])
        
        # 准备单日数据进行预测 (模拟当时的情况)
        # 注意：这里直接用 row 可能不太准，因为 rolling 计算需要历史。
        # 正确做法是取到这一天为止的切片。但为了简单，我们直接用已经算好的 feature
        # 只要 feature 没用到未来数据就行 (ma, rsi 都是历史)
        
        # 构造单行 DataFrame
        single_day_df = pd.DataFrame([row]) 
        score = model.predict(single_day_df)
        
        # 查大盘状态
        idx_row = index_df[index_df['trade_date'].astype(str) == date_str]
        if not idx_row.empty:
            idx_close = idx_row.iloc[0]['close']
            idx_ma60 = idx_row.iloc[0]['ma60']
            is_bull = idx_close > idx_ma60
            market_status = "Bull" if is_bull else "Bear(❌)"
        else:
            market_status = "Unknown"
            is_bull = True # 默认
            
        # 判定结果
        if not is_bull:
            result = "Blocked by Market"
        elif score < 0.6:
            result = f"Score Low (<0.6)"
        else:
            result = "✅ BUY SIGNAL"
            
        pct_chg = f"{row['pct_chg']:.2f}%"
        print(f"{date_str:<10} {row['close']:<8} {pct_chg:<8} {score:.4f}     {market_status:<10} {result}")

if __name__ == "__main__":
    main()
