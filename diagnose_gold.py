import pandas as pd
from datetime import datetime, timedelta
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer

def main():
    print("🔍 Diagnosing Gold ETF (518880.SH)...")
    
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    
    # 获取数据
    code = "518880.SH"
    df = data_manager.update_and_get_data(code)
    
    # 计算指标
    df = feature_eng.calculate_technical_indicators(df)
    
    # 重新计算 Label (看看系统认为它是 0 还是 1)
    # 逻辑: 未来5天最高价涨幅 > 2%
    df = feature_eng.add_labels(df, horizon=5, threshold=0.02)
    
    # 截取最近3个月
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    start_date_str = start_date.strftime("%Y%m%d")
    
    df = df[df['trade_date'].astype(str) >= start_date_str].copy()
    
    # 打印详细数据
    print(f"\n{'Date':<10} {'Close':<8} {'Change%':<8} {'Next5DayMax%':<12} {'Target (Label)'}")
    print("-" * 60)
    
    for i in range(len(df)):
        if i + 5 >= len(df): break # 最后几天没法算未来
        
        row = df.iloc[i]
        date = row['trade_date']
        close = row['close']
        prev_close = df.iloc[i-1]['close'] if i > 0 else close
        change = (close / prev_close - 1) * 100
        
        # 手动算一下未来5天最高收益
        future_prices = df.iloc[i+1 : i+6]['high']
        max_future_price = future_prices.max()
        max_ret = (max_future_price / close - 1) * 100
        
        target = row['target']
        
        print(f"{date:<10} {close:<8.3f} {change:6.2f}%   {max_ret:6.2f}%       {target}")

    # 统计涨幅
    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    total_ret = (end_price / start_price - 1) * 100
    print(f"\nTotal Return (Last 3 Months): {total_ret:.2f}%")

if __name__ == "__main__":
    main()