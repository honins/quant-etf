import pandas as pd
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer

def main():
    print("🔍 Diagnosing Satellite ETF (159206.SZ)...")
    
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    
    # 获取数据
    df = data_manager.update_and_get_data("159206.SZ")
    if df.empty:
        print("No data found!")
        return
        
    # 计算指标
    df = feature_eng.calculate_technical_indicators(df)
    
    # 打印最后 3 天的数据
    print("\n📅 Last 3 Days Data:")
    cols = ['trade_date', 'close', 'pct_chg', 'vol', 'rsi_14', 'ma5', 'upper']
    print(df[cols].tail(3))
    
    # 分析最新一天
    latest = df.iloc[-1]
    print(f"\n📊 Technical Check ({latest['trade_date']}):")
    print(f"- Close: {latest['close']}")
    print(f"- Pct Chg: {latest['pct_chg']:.2f}%")
    print(f"- RSI_14: {latest['rsi_14']:.2f} (Overbought > 70?)")
    
    # 乖离率 (Bias): (Close - MA5) / MA5
    bias_5 = (latest['close'] - latest['ma5']) / latest['ma5'] * 100
    print(f"- Bias MA5: {bias_5:.2f}% (Too high?)")
    
    # 是否突破布林带上轨
    print(f"- Above Upper Band: {latest['close'] > latest['upper']}")

if __name__ == "__main__":
    main()
