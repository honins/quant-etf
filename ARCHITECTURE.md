# 架构说明

## 目录

```text
quant-etf/
├─ config/
│  ├─ settings.py
│  ├─ tickers.py
│  └─ holdings.yml
├─ data/
│  ├─ market_data.db
│  └─ xgb_model.json
├─ reports/
├─ src/
│  ├─ backtest/
│  │  ├─ backtester.py
│  │  ├─ hybrid_runner.py
│  │  └─ strategy_config.py
│  ├─ core/
│  │  └─ interfaces.py
│  ├─ data_loader/
│  │  ├─ data_manager.py
│  │  └─ tushare_loader.py
│  ├─ features/
│  │  └─ technical.py
│  ├─ models/
│  │  ├─ scoring_model.py
│  │  └─ xgb_model.py
│  ├─ strategy/
│  │  └─ logic.py
│  └─ utils/
│     ├─ explainer.py
│     ├─ feishu_bot.py
│     ├─ holdings_manager.py
│     └─ reporter.py
├─ tests/
│  └─ test_strategy_filter.py
├─ main.py
├─ backtest_recent.py
├─ backtest_3m.py
├─ backtest_q4_2025.py
├─ train_and_backtest.py
└─ optimize_strategy.py
```

## 主流程

1. `DataManager` 从本地数据库读数据，不足部分通过 `TushareLoader` 增量更新。
2. `FeatureEngineer` 计算技术指标、相对强弱和训练标签。
3. `XGBoostModel` 负责训练、保存、加载和打分。
4. `StrategyFilter` 按市场状态和标的类型过滤入场信号。
5. `RiskManager` 计算 ATR 止损和移动止盈。
6. `Reporter` 输出 Markdown 报告，`FeishuBot` 负责发送通知。

## 回测收口

- 所有单标的回测都统一走 `src/backtest/hybrid_runner.py`
- `backtest_recent.py` 是主入口
- `backtest_3m.py` 只是设置 `LOOKBACK_DAYS=90`
- `backtest_q4_2025.py` 是固定时间窗包装器

## 设计原则

- 尽量只保留一条真实生效的策略链路
- 入口脚本只做参数解析和展示
- 逻辑、配置、回测核心下沉到 `src/`
- 生成产物不进入版本控制
