import sys
import pandas as pd
from datetime import datetime
from config import tickers
from config.settings import settings
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.scoring_model import RuleBasedModel
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter, RiskManager
from src.utils.reporter import Reporter
from src.utils.holdings_manager import HoldingsManager
from src.utils.explainer import TechnicalExplainer
from src.utils.feishu_bot import FeishuBot

def main():
    print("🚀 Starting Quant-ETF System...")
    
    # 1. 初始化模块
    try:
        loader = TushareLoader()
    except ValueError as e:
        print(f"Error: {e}")
        return

    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    holdings_manager = HoldingsManager()
    
    # 切换为 ML 模型
    # 优先尝试加载 XGBoost，其次 Random Forest，最后回退到规则模型
    model = None
    
    # 1. Try XGBoost
    try:
        xgb = XGBoostModel()
        if xgb.load_model():
            print("🤖 Loaded AI Model (XGBoost).")
            model = xgb
    except Exception as e:
        print(f"XGB load failed: {e}")

    # 2. Fallback to Rules
    if model is None:
        print("⚠️ No trained AI models found. Falling back to RuleBasedModel.")
        print("Tip: Run 'python train_and_backtest.py' to train the AI model first.")
        model = RuleBasedModel()

    strat_filter = StrategyFilter()
    risk_manager = RiskManager()
    reporter = Reporter()
    
    # 2. 获取大盘指数数据 (以沪深300为例: 000300.SH, 或者是上证指数 000001.SH)
    print("📊 Analyzing Market Trend...")
    # 注意: Tushare 指数代码通常是 000001.SH (上证) 或 399006.SZ (创业板)
    # 这里用沪深300代表大盘
    index_code = '000300.SH' 
    # 使用 DataManager 获取并缓存指数数据
    index_df = data_manager.update_and_get_data(index_code, is_index=True)
    
    if not index_df.empty:
        # 计算指数均线用于判断牛熊
        index_df = feature_eng.calculate_technical_indicators(index_df)
    
    # 3. 遍历标的池
    results = []
    ticker_list = tickers.get_ticker_list()
    
    for code in ticker_list:
        name = tickers.TICKERS[code]
        print(f"Processing {name} ({code})...")
        
        # a. 获取数据 (自动增量更新)
        df = data_manager.update_and_get_data(code, is_index=False)
        
        if df.empty:
            print(f"⚠️ No data for {code}")
            continue
            
        # b. 特征工程
        df = feature_eng.calculate_technical_indicators(df)
        df = model.prepare_data(df) # 补充模型需要的额外特征
        
        if len(df) < 60:
            print(f"⚠️ Not enough data for {code} (need > 60 days)")
            continue

        # c. 模型打分
        score = model.predict(df)
        
        # d. 策略过滤
        is_buy, market_status = strat_filter.filter_signal(score, index_df, code=code)
        
        # e. 风控计算
        risk_data = risk_manager.calculate_stops(df)
        
        # f. 技术面解释 (新增)
        explanations = TechnicalExplainer.explain(df)
        
        results.append({
            'code': code,
            'name': name,
            'score': score,
            'is_buy': is_buy,
            'current_price': df.iloc[-1]['close'],
            'risk': risk_data,
            'reasons': explanations # 传递解释列表
        })
        
    # 3.5 检查现有持仓 (新增功能)
    holdings_status = holdings_manager.check_holdings(data_manager, feature_eng)
    
    # 4. 生成报告
    print("📝 Generating Report...")
    # 获取最后计算的 market_status，如果没跑循环则默认 Unknown
    m_status = "Unknown"
    if 'market_status' in locals():
        m_status = market_status
        
    report_path = reporter.generate_markdown(results, m_status, holdings_status)
    print("✅ Report Generated!")
    
    # 5. 发送飞书通知 (替代邮件)
    try:
        # 读取报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        bot = FeishuBot()
        # 提取标题
        title = f"Quant-ETF Daily Report ({datetime.now().strftime('%Y-%m-%d')})"
        bot.send_markdown(title, content)
    except Exception as e:
        print(f"⚠️ Notification failed: {e}")
    
    print("🎉 All tasks completed.")

if __name__ == "__main__":
    main()
