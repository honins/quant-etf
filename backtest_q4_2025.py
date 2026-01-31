
import pandas as pd
import numpy as np
from datetime import datetime
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.backtest.backtester import Backtester

def main():
    # 1. 设置回测时间段
    start_date = "20251001"
    end_date = "20251231"
    
    print(f"🚀 Starting Backtest for Q4 2025 ({start_date} - {end_date})...")
    
    # 2. 初始化组件
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    backtester = Backtester(initial_capital=100000.0)
    
    # 3. 加载模型
    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("❌ Model not found. Please train the model first.")
        return

    # 4. 获取 ETF 列表
    ticker_list = tickers.get_ticker_list()
    
    results = []
    
    print(f"📊 Analyzing {len(ticker_list)} ETFs...")
    
    for code in ticker_list:
        # 获取数据
        df = data_manager.update_and_get_data(code)
        if df.empty:
            continue
            
        # 特征工程
        df = feature_eng.calculate_technical_indicators(df)
        df = df.dropna()
        
        # 筛选回测时间段的数据
        # 注意：我们需要保留一些历史数据用于计算指标，但在 run() 内部会处理
        # 这里为了简单，先保留所有数据，但在预测和回测时只关注目标区间
        # 不过 backtester.run 通常接受整个 df，我们需要确保 df 包含目标区间
        
        # 预测
        probs = model.predict_batch(df)
        
        # 截取目标时间段的数据进行回测
        mask = (df['trade_date'].astype(str) >= start_date) & (df['trade_date'].astype(str) <= end_date)
        test_df = df[mask].copy()
        test_probs = probs[mask]
        
        if test_df.empty:
            continue
            
        # 执行回测
        result = backtester.run(test_df, test_probs, threshold=0.6, code=code)
        
        if result['num_trades'] > 0:
            results.append({
                'code': code,
                'total_return': result['total_return'],
                'win_rate': result['win_rate'],
                'trades': result['num_trades'],
                'final_equity': result['final_equity']
            })
            
            # 打印个股详情
            print(f"[{code}] Return: {result['total_return']*100:6.2f}% | Win Rate: {result['win_rate']*100:6.1f}% | Trades: {result['num_trades']}")

    # 5. 汇总报告
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        print("\n" + "="*50)
        print(f"📈 Q4 2025 Backtest Summary ({start_date}-{end_date})")
        print("="*50)
        print(f"Avg Return: {results_df['total_return'].mean()*100:.2f}%")
        print(f"Total Trades: {results_df['trades'].sum()}")
        print("-" * 50)
        print(results_df[['code', 'total_return', 'win_rate', 'trades']].to_string(index=False, float_format=lambda x: "{:.2f}".format(x) if isinstance(x, float) and x < 10 else "{:.4f}".format(x)))
        
        # 保存详细报告
        report_path = "reports/backtest_q4_2025.csv"
        results_df.to_csv(report_path, index=False)
        print(f"\n📄 Report saved to {report_path}")
    else:
        print("\n⚠️ No trades triggered during this period.")

if __name__ == "__main__":
    main()
