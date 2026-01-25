import pandas as pd
from datetime import datetime, timedelta
import os
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel

def main():
    print("🚀 Exporting Daily Signals (Last 60 Days)...")
    
    # 1. 初始化
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    model = XGBoostModel()
    if not model.load_model():
        print("❌ Model not found.")
        return
        
    # 2. 设定时间
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    start_date_str = start_date.strftime("%Y%m%d")
    print(f"📅 Range: {start_date_str} - {end_date.strftime('%Y%m%d')}")
    
    # 3. 准备大盘状态字典 {date: is_bull}
    print("📊 Loading Market Status...")
    index_df = data_manager.update_and_get_data('000300.SH', is_index=True)
    index_df = feature_eng.calculate_technical_indicators(index_df)
    
    market_status_map = {}
    for _, row in index_df.iterrows():
        d = str(row['trade_date'])
        # 简单判定：收盘价 > MA60 为牛市
        is_bull = row['close'] > row['ma60'] if pd.notnull(row['ma60']) else True
        market_status_map[d] = "Bull" if is_bull else "Bear"

    # 4. 遍历所有标的
    all_signals = []
    ticker_list = tickers.get_ticker_list()
    
    for code in ticker_list:
        name = tickers.TICKERS[code]
        # print(f"Processing {name}...", end="\r")
        
        df = data_manager.update_and_get_data(code)
        if df.empty: continue
        
        df = feature_eng.calculate_technical_indicators(df)
        df = df.dropna()
        
        # 截取最近60天
        target_df = df[df['trade_date'].astype(str) >= start_date_str].copy()
        if target_df.empty: continue
        
        # 批量预测分数
        probs = model.predict_batch(target_df)
        target_df['ai_score'] = probs
        
        # 逐日整理
        for i in range(len(target_df)):
            row = target_df.iloc[i]
            date_str = str(row['trade_date'])
            
            market_status = market_status_map.get(date_str, "Unknown")
            score = row['ai_score']
            
            # 策略逻辑复现
            threshold = 0.45 if market_status == "Bull" else 0.75
            signal = "BUY" if score >= threshold else "WAIT"
            
            # 信号强度描述
            if signal == "BUY":
                if score > 0.7: strength = "Strong Buy 🔥"
                elif score > 0.6: strength = "Buy ✅"
                else: strength = "Weak Buy ⚠️"
            else:
                if market_status == "Bear": strength = "Bear Filter 🛡️"
                else: strength = "Low Score ⚪"

            all_signals.append({
                "Date": date_str,
                "Code": code,
                "Name": name,
                "Close": row['close'],
                "PctChg": f"{row['pct_chg']:.2f}%",
                "AI_Score": f"{score:.3f}",
                "Market": market_status,
                "Threshold": threshold,
                "Signal": signal,
                "Desc": strength
            })

    # 5. 保存 CSV
    if not all_signals:
        print("No signals generated.")
        return
        
    res_df = pd.DataFrame(all_signals)
    # 按日期降序，然后按分数降序
    res_df = res_df.sort_values(['Date', 'AI_Score'], ascending=[False, False])
    
    output_path = "reports/daily_signals_last_60_days.csv"
    res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Exported {len(res_df)} rows to: {output_path}")
    print("You can open this CSV file in Excel to view daily details.")
    
    # 打印最近5天的 Top 3
    print("\n👀 Preview (Last 3 Days Top Signals):")
    recent_dates = res_df['Date'].unique()[:3]
    for d in recent_dates:
        print(f"\n📅 {d}:")
        day_data = res_df[res_df['Date'] == d].head(3)
        print(day_data[['Name', 'Close', 'PctChg', 'AI_Score', 'Signal', 'Desc']].to_string(index=False))

if __name__ == "__main__":
    main()
