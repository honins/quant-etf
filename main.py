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


def _calc_position_size(risk_data: dict, total_capital: float = 100_000, risk_pct: float = 0.02) -> dict:
    """
    【优化4】 基于波动率倒数的动态仓位建议 (Volatility Targeting)

    核心思路：每笔交易承担的最大风险 = 总资金 × risk_pct（默认2%）
    建议股数 = (总资金 × 最大单笔风险) / 每股风险 (entry - stop_loss)
    这确保了高波动ETF自动买少，低波动ETF自动买多，整体账户波动率趋于稳定。

    Args:
        risk_data: RiskManager.calculate_stops() 的返回值
        total_capital: 账户总资金（默认10万元，仅用于计算比例）
        risk_pct: 单笔愿意承担的最大亏损占总资金比例（默认2%）

    Returns:
        dict 包含建议股数和建议金额（以参考资金为基准的比例）
    """
    if not risk_data:
        return {}

    risk_per_share = risk_data.get('risk_per_share', 0)
    current_price = risk_data.get('current_price', 0)

    if risk_per_share <= 0 or current_price <= 0:
        return {}

    max_loss = total_capital * risk_pct
    suggested_shares = int(max_loss / risk_per_share / 100) * 100  # 向下取整到100股
    suggested_value = round(suggested_shares * current_price, 2)
    suggested_weight = round(suggested_value / total_capital, 4)  # 占总资金比例

    return {
        "suggested_shares": suggested_shares,
        "suggested_value": suggested_value,
        "suggested_weight_pct": round(suggested_weight * 100, 2),
    }


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
    ticker_list = tickers.get_ticker_list(include_observe=True)

    def use_dynamic_for_code(code: str):
        category = tickers.get_ticker_category(code)
        if category == "core":
            return False
        if category == "satellite":
            return True
        return settings.USE_DYNAMIC_THRESHOLD
    
    market_status = "Unknown"
    for code in ticker_list:
        name = tickers.TICKERS[code]
        category = tickers.get_ticker_category(code)
        category_label = tickers.get_ticker_category_label(code)
        print(f"Processing {name} ({code})...")
        
        # a. 获取数据 (自动增量更新)
        df = data_manager.update_and_get_data(code, is_index=False)
        
        if df.empty:
            print(f"⚠️ No data for {code}")
            continue
            
        # b. 特征工程
        df = feature_eng.calculate_technical_indicators(df)
        df = model.prepare_data(df)  # 补充模型需要的额外特征
        # 【优化2】注入相对大盘强弱特征（跨截面特征，只有指数数据有效时才计算）
        if not index_df.empty:
            df = feature_eng.add_relative_strength(df, index_df, period=20)
        
        if len(df) < 60:
            print(f"⚠️ Not enough data for {code} (need > 60 days)")
            continue

        # c. 模型打分
        score = model.predict(df)

        dynamic_threshold = None
        use_dynamic = use_dynamic_for_code(code)
        if use_dynamic and callable(getattr(model, "predict_batch", None)):
            lookback = min(settings.DYNAMIC_THRESHOLD_LOOKBACK, len(df))
            recent_scores = model.predict_batch(df.tail(lookback))
            dynamic_threshold = StrategyFilter.dynamic_threshold(recent_scores)
        
        # d. 策略过滤
        is_buy, market_status = strat_filter.filter_signal(score, index_df, code=code, dynamic_threshold=dynamic_threshold)
        decision_note = ""
        if category == "observe":
            is_buy = False
            decision_note = "观察池标的，仅跟踪不进入默认实盘交易。"
        
        # e. 风控计算
        risk_data = risk_manager.calculate_stops(df, code=code)
        
        # f. 【优化4】计算动态建议仓位
        position_size = _calc_position_size(risk_data)
        
        # g. 技术面解释
        explanations = TechnicalExplainer.explain(df)
        
        results.append({
            'code': code,
            'name': name,
            'category': category,
            'category_label': category_label,
            'score': score,
            'is_buy': is_buy,
            'current_price': df.iloc[-1]['close'],
            'risk': risk_data,
            'reasons': explanations,
            'decision_note': decision_note,
            'position_size': position_size,  # 【优化4】仓位建议
        })
        
    # 3.5 检查现有持仓 (新增功能)
    holdings_status = holdings_manager.check_holdings(data_manager, feature_eng)
    
    # 4. 生成报告
    print("📝 Generating Report...")
    report_path = reporter.generate_markdown(results, market_status, holdings_status)
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
