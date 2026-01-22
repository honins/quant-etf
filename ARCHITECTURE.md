# 系统架构设计文档 (Architecture Design)

## 1. 设计原则 (Design Principles)
本项目严格遵循 SOLID 原则，特别是：
- **单一职责 (SRP)**: 每个模块只负责一个核心功能（如数据获取、信号计算、风控）。
- **依赖倒置 (DIP)**: 高层业务逻辑依赖于抽象接口，而不是具体实现（例如 `DataProvider` 接口）。

## 2. 目录结构 (Directory Structure)

```
quant-etf/
├── config/                 # 配置文件
│   ├── settings.py         # 全局配置
│   └── tickers.py          # 标的池配置 (<=20)
├── data/                   # 本地数据存储 (SQLite/CSV)
├── reports/                # 生成的日报
├── src/                    # 源代码
│   ├── core/               # 核心抽象层 (Interfaces)
│   │   ├── interfaces.py   # 定义 DataProvider, Strategy, RiskManager 等接口
│   ├── data_loader/        # 数据层
│   │   ├── tushare_loader.py
│   │   └── data_manager.py # 负责数据的增量更新与缓存
│   ├── features/           # 特征工程层
│   │   ├── technical.py    # 技术指标计算 (RSI, MA, ATR, OBV)
│   ├── models/             # 信号挖掘层 (AI/Scoring)
│   │   ├── base_model.py
│   │   └── scoring_model.py # 机器学习或规则打分模型
│   ├── strategy/           # 策略与风控层
│   │   ├── filter.py       # 策略过滤 (牛熊判断)
│   │   └── risk.py         # 风控执行 (ATR止损计算)
│   └── utils/              # 工具类 (Logger, DateUtils)
├── main.py                 # 程序入口
├── requirements.txt        # 依赖包
└── README.md               # 项目说明
```

## 3. 核心模块流程 (Core Flow)

1.  **Data Phase**: `DataManager` 读取配置 -> 调用 `TushareLoader` -> 更新本地 `data/stock.db` -> 返回最新 `DataFrame`。
2.  **Feature Phase**: `FeatureEngineer` 接收原始数据 -> 计算 TA-Lib 指标 (MA, ATR, RSI, OBV) -> 输出带有特征的 `DataFrame`。
3.  **Signal Phase**: `ScoringModel` 接收特征数据 -> 运行预测/打分逻辑 -> 输出 `SignalResult` (包含分数 0-1)。
4.  **Strategy Phase**: `StrategyFilter` 结合大盘趋势与个股信号 -> 决定是否由 "观察" 转为 "买入建议"。
5.  **Risk Phase**: `RiskManager` 计算当前价格的 ATR 止损位、移动止盈位。
6.  **Reporting Phase**: 将上述结果汇总生成 Markdown 报告。

## 4. 扩展性设计 (Scalability)

- **数据源切换**: 若要从 Tushare 切换到 AKshare，只需实现 `DataProvider` 接口的新类，无需修改策略代码。
- **模型升级**: 可以在 `src/models` 中新增更复杂的深度学习模型，只要保持输入输出接口一致。

