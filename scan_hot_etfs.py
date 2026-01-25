import pandas as pd
from src.data_loader.tushare_loader import TushareLoader
from config import tickers

def main():
    print("🛰️ Scanning for Hot ETFs (Volume & Momentum)...")
    
    loader = TushareLoader()
    
    # 1. 获取全市场 ETF 列表
    # 注意: tushare 的 fund_basic 接口可以获取
    try:
        # 这里为了演示，我们假设直接从 tushare 拉取每日行情，按成交额排序
        # 实盘中通常用 pro.fund_daily(trade_date='20240126')
        today_date = pd.Timestamp.now().strftime("%Y%m%d")
        
        # 由于这里是演示环境，我们无法直接调取全市场实时数据
        # 但逻辑如下：
        # df = pro.fund_daily(trade_date=latest_date)
        # df = df.sort_values('amount', ascending=False).head(50)
        
        print("⚠️ Note: Full market scan requires pro.fund_daily API with sufficient permissions.")
        print("   Checking current watchlist for volume surge instead...")
        
        # 2. 替代方案：扫描当前监控池中的“异动”
        # 并提示用户去哪里找新标的
        
        print("\n💡 How to find NEXT big theme:")
        print("1. The system currently monitors fixed tickers in config/tickers.py")
        print("2. To auto-discover new themes, look for ETFs with:")
        print("   - High Volume (Amount > 100M)")
        print("   - Rising Trend (Price > MA20)")
        print("   - Fund Inflow (Shares increasing)")
        
        print("\n🔍 Analyzing current tickers for breakout potential:")
        
        # 简单的异动扫描
        from src.data_loader.data_manager import DataManager
        from src.features.technical import FeatureEngineer
        
        dm = DataManager(loader)
        fe = FeatureEngineer()
        
        candidates = []
        for code in tickers.get_ticker_list():
            df = dm.update_and_get_data(code)
            if df.empty: continue
            
            # 计算 5日均量
            vol_ma5 = df['vol'].rolling(5).mean().iloc[-1]
            curr_vol = df.iloc[-1]['vol']
            
            if curr_vol > vol_ma5 * 1.5:
                candidates.append({
                    "code": code,
                    "name": tickers.TICKERS[code],
                    "reason": "Volume Surge (>1.5x)"
                })
                
        if candidates:
            print("\n🔥 Hot Tickers in Watchlist:")
            for c in candidates:
                print(f"- {c['name']} ({c['code']}): {c['reason']}")
        else:
            print("No abnormal volume detected in watchlist.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
