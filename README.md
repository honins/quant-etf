# Quant-ETF

基于 A 股 ETF 的量化分析与工作台项目，当前包含三类能力：

- 日常信号与日报生成
- 本地 Dashboard 工作台与常用操作入口
- 回测、训练与策略参数探索

当前主工作流已经收敛为一套更统一的实现：

- `main.py`: 生成当日信号报告，检查持仓，并尝试推送飞书
- `dashboard.py`: 提供本地 Dashboard API，并服务 React 前端工作台
- `backtest_recent.py`: 最近窗口回测入口，支持固定阈值、动态阈值等模式对比
- `backtest_3m.py`: 最近 90 天回测包装器
- `backtest_q4_2025.py`: 固定区间回测
- `train_and_backtest.py`: 使用当前交易池训练 XGBoost，并做 walk-forward 验证
- `optimize_strategy.py`: 搜索策略参数

## 项目结构

- `src/data_loader/`: 数据加载
- `src/features/technical.py`: 技术指标与特征工程
- `src/models/xgb_model.py`: 模型定义
- `src/strategy/logic.py`: 策略与风控逻辑
- `src/backtest/`: 回测核心
- `src/dashboard/`: Dashboard 数据组织与前端代码
- `src/utils/`: 报告与通知

## 标的分组

当前标的池分为三类：

- `Core tradable`: 核心可交易，默认固定阈值
- `Satellite tradable`: 卫星可交易，默认动态阈值
- `Observe only`: 只观察，不默认进入交易池

默认可交易池当前为 `24` 只，观察池为 `4` 只。

仓库中保留了重复指数代码的别名映射，但不会再进入默认交易池：

- `510330.SH -> 510300.SH`
- `159919.SZ -> 510300.SH`
- `512000.SH -> 512880.SH`

这样可以避免同一指数暴露被重复计入回测和统计。

## 环境准备

1. 创建虚拟环境并安装依赖

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. 配置 `.env`

- `TUSHARE_TOKEN`
- `FEISHU_WEBHOOK`，可选

3. 如需持仓监控，维护 `config/holdings.yml`

## Dashboard 工作台

当前 Dashboard 已升级为 `React + Vite + TypeScript` 的单页工作台，Python 侧负责：

- 提供 Dashboard 数据 API
- 处理刷新、生成日报等动作
- 服务前端构建产物

前端代码位于：

- `src/dashboard/frontend/`

首次使用或前端有变更时，需要先安装并构建前端：

```bash
cd src/dashboard/frontend
npm install
npm run build
```

然后回到项目根目录启动本地工作台：

```bash
python dashboard.py --serve --port 8765
```

浏览器访问：

- [http://127.0.0.1:8765/dashboard.html](http://127.0.0.1:8765/dashboard.html)

当前工作台支持的高频操作包括：

- 查看实时信号
- 点击信号卡片切换标的详情
- 查看持仓监控与跟踪止损
- 查看 90/180 天回测摘要
- 生成今日日报
- 生成并推送飞书
- 刷新工作台数据

## 常用命令

```bash
python main.py
python dashboard.py
python dashboard.py --serve
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

默认情况下，`backtest_recent.py` 只回测可交易池，不包含观察池。若需要纳入观察池：

```bash
$env:INCLUDE_OBSERVE="1"; python backtest_recent.py
```

## 运行说明

- `main.py` 默认会拉取最新数据，生成 `reports/` 下的日报，并尝试发送飞书通知
- `dashboard.py` 启动时会生成 `reports/dashboard-data.json` 快照，并在 `--serve` 时提供本地 Web 工作台
- `train_and_backtest.py` 会重写 `data/xgb_model.json`
- `data/market_data.db` 是项目的本地行情数据库快照
- `RuleBasedModel` 仅作为没有训练模型时的兜底，不代表主策略

在部分 Windows GBK 终端中，带中文或特殊字符的输出可能触发编码问题。可先执行：

```bash
$env:PYTHONIOENCODING="utf-8"
```

再运行相关脚本。

## 数据一致性

当前数据层做了两层去重：

- 写入数据库前，按 `ts_code + trade_date` 去重
- 从数据库读取后，再按 `trade_date` 去重

这样可以避免重复行情把回测结果抬高。

## 输出文件

- `reports/*.md`: 日报与回测输出
- `reports/dashboard-data.json`: Dashboard 快照
- `data/xgb_model.json`: 当前训练好的 XGBoost 模型
- `data/market_data.db`: 本地行情数据库

## 文档

- `STRATEGIES.md`: 策略思路
- `ALGORITHM_DETAILS.md`: 算法与标签说明
- `ARCHITECTURE.md`: 模块与目录结构
- `DEPLOY.md`: 部署说明
