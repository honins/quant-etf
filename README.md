# Quant-ETF

基于 A 股 ETF 的单标的量化分析项目。当前主线已经收口到一套统一实现：

- `main.py`: 生成当日信号报告并推送飞书
- `backtest_recent.py`: 统一回测入口，支持近窗、对比、网格测试
- `backtest_3m.py`: 近 90 天包装器
- `backtest_q4_2025.py`: 固定区间回测
- `train_and_backtest.py`: 重训模型并做 walk-forward 验证
- `optimize_strategy.py`: 搜索策略参数

## 当前架构

- 数据层: `src/data_loader/`
- 特征工程: `src/features/technical.py`
- 模型: `src/models/xgb_model.py`
- 策略与风控: `src/strategy/logic.py`
- 回测核心: `src/backtest/`
- 报告与通知: `src/utils/`

## 常用命令

```bash
python main.py
python backtest_recent.py
python backtest_3m.py
python backtest_q4_2025.py
python train_and_backtest.py
python optimize_strategy.py
python -m unittest tests.test_strategy_filter -v
```

## 环境准备

1. 创建虚拟环境并安装依赖

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. 配置 `.env`

- `TUSHARE_TOKEN`
- `FEISHU_WEBHOOK`（可选）

3. 如需持仓监控，维护 `config/holdings.yml`

## 说明

- `reports/` 下的 Markdown 和 CSV 属于运行产物，不再纳入版本控制。
- `data/market_data.db` 和 `data/xgb_model.json` 仍然是项目运行所需资产。
- `RuleBasedModel` 仅作为无训练模型时的兜底，不代表主策略。

## 文档

- `STRATEGIES.md`: 模型/策略思路
- `ALGORITHM_DETAILS.md`: 当前算法与标签定义
- `ARCHITECTURE.md`: 当前目录与模块职责
- `DEPLOY.md`: 部署方式
