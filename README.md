# Quant-ETF

基于 A 股 ETF 的单标的量化分析项目。当前主线已经收口到一套统一实现：

- `main.py`: 生成当日信号报告，检查持仓，并推送飞书
- `backtest_recent.py`: 统一回测入口，支持最近窗口、阈值对比、动态/固定模式对比
- `backtest_3m.py`: 最近 90 天回测包装器
- `backtest_q4_2025.py`: 固定区间回测
- `train_and_backtest.py`: 使用当前交易池重训 XGBoost，并做 walk-forward 验证
- `optimize_strategy.py`: 搜索策略参数

## Current Design

- 数据层: `src/data_loader/`
- 特征工程: `src/features/technical.py`
- 模型: `src/models/xgb_model.py`
- 策略与风控: `src/strategy/logic.py`
- 回测核心: `src/backtest/`
- 报告与通知: `src/utils/`

## Ticker Universe

当前标的池分成三类：

- `Core tradable`: 宽基、红利、低波类，默认走固定阈值
- `Satellite tradable`: 行业、主题类，默认走动态阈值
- `Observe only`: 保留更新、打分和日报展示，但默认不进入实盘交易池

默认可交易池当前为 `24` 只，观察池为 `4` 只。

仓库还保留了重复指数代码的别名映射，但不会再进入默认交易池：

- `510330.SH -> 510300.SH`
- `159919.SZ -> 510300.SH`
- `512000.SH -> 512880.SH`

这样可以避免同一指数暴露被重复计入回测和统计。

## Environment Setup

1. 创建虚拟环境并安装依赖

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. 配置 `.env`

- `TUSHARE_TOKEN`
- `FEISHU_WEBHOOK`（可选）

3. 如果需要持仓监控，维护 `config/holdings.yml`

## Common Commands

```bash
python main.py
python backtest_recent.py
python backtest_3m.py
python backtest_q4_2025.py
python train_and_backtest.py
python optimize_strategy.py
python -m unittest tests.test_strategy_filter tests.test_ticker_groups -v
```

常用回测参数：

```bash
python backtest_recent.py
$env:LOOKBACK_DAYS="180"; python backtest_recent.py
$env:DIFF_MODE="1"; python backtest_recent.py
$env:SELECT_MODE="1"; python backtest_recent.py
```

默认情况下，`backtest_recent.py` 只回测可交易池，不包含观察池。需要纳入观察池时：

```bash
$env:INCLUDE_OBSERVE="1"; python backtest_recent.py
```

## Runtime Notes

- `main.py` 默认会拉取最新数据、生成 `reports/` 下的日报，并尝试发送飞书通知。
- `train_and_backtest.py` 会重写 `data/xgb_model.json`。
- `data/market_data.db` 是项目的本地行情数据库快照。
- `RuleBasedModel` 仅作为没有训练模型时的兜底，不代表主策略。

在部分 Windows GBK 终端中，训练脚本的 emoji 输出可能触发编码错误。可以先执行：

```bash
$env:PYTHONIOENCODING="utf-8"
```

再运行训练脚本：

```bash
python train_and_backtest.py
```

## Data Integrity

当前数据层已经做了两层去重：

- 写入数据库前，按 `ts_code + trade_date` 去重
- 从数据库读取后，再按 `trade_date` 去重

这样可以避免重复行情把回测结果抬高。

## Outputs

- `reports/*.md`: 日报和回测输出
- `data/xgb_model.json`: 当前训练好的 XGBoost 模型
- `data/market_data.db`: 本地行情数据库

## Documents

- `STRATEGIES.md`: 策略思路
- `ALGORITHM_DETAILS.md`: 算法与标签说明
- `ARCHITECTURE.md`: 模块与目录结构
- `DEPLOY.md`: 部署说明
