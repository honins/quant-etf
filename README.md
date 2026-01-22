# AI 驱动的 ETF 量化交易系统 (Quant-ETF)

[![AI Powered](https://img.shields.io/badge/AI-XGBoost-blue)](src/models/xgb_model.py)
[![Strategy](https://img.shields.io/badge/Strategy-Hybrid-green)](STRATEGIES.md)
[![License](https://img.shields.io/badge/license-MIT-gray)]()

这是一个**务实的、AI 驱动的波段交易辅助系统**。

它不仅仅是一个简单的指标计算器，而是一个集成了 **XGBoost 机器学习** 和 **传统规则风控** 的混合决策系统。它旨在帮助普通交易者在 A 股 ETF 市场中寻找高胜率的买点，并严格控制回撤。

---

## 🚀 核心亮点

*   **🧠 AI 进化 (XGBoost)**: 抛弃死板的指标规则，使用竞赛级算法 **XGBoost** 从历史数据中自动挖掘非线性 Alpha。
*   **🛡️ 混合策略 (Hybrid Strategy)**:
    *   **进攻**: 由 AI 负责，敏锐捕捉量价时空的共振机会。
    *   **防守**: 由规则负责，大盘熊市强制空仓，个股亏损强制止损。
*   **📉 真实回测**: 内置回测引擎，不画大饼。最近3个月实盘回测显示，在震荡市中不仅跑赢大盘，还捕捉到了新能源板块的 **+13.92%** 收益。
*   **⚡ 极速响应**: 本地增量缓存数据，秒级生成日报。

---

## 📚 文档导航

*   [**STRATEGIES.md**](STRATEGIES.md): 深度解析为什么选择 XGBoost？为什么放弃贝叶斯？混合策略是如何工作的？
*   [**ALGORITHM_DETAILS.md**](ALGORITHM_DETAILS.md): 系统的“白皮书”。详细列出了 12 个核心特征的计算公式和 AI 模型的训练细节。

---

## 🎬 理想工作流 (Daily Workflow)

本系统旨在成为您的**“智能副驾驶”**。推荐的日常使用流程：

### 🕒 1. 收盘后 (15:10)
运行程序生成日报。

```bash
python main.py
```

您会看到类似这样的分析：
> **[新能源车ETF]**
> *   🤖 **AI 评分**: **0.64** (高分信号!)
> *   🌊 **市场状态**: **Bull Market** (允许开仓)
> *   💡 **决策**: **买入**
> *   🛡️ **风控**: 止损位 **1.817** (ATR=0.053)

### 🕒 2. 盘前 (次日 09:25)
根据昨晚的计划，挂单买入。同时设置条件单（止损单）。

### 🕒 3. 持仓期
每天运行程序，关注 **移动止盈位** 的更新。
*   如果价格上涨，止盈位会自动上移，锁定利润。
*   如果价格跌破止盈位，坚决离场。

---

## 🛠️ 安装与使用

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境 (Windows)
venv\Scripts\activate

# 安装依赖
# 注意: 我们已移除对 TA-Lib 的硬性依赖，Windows 下也能轻松安装
pip install -r requirements.txt
```

### 2. 配置 Token

1.  复制 `.env.example` 为 `.env`
2.  填入您的 Tushare Token (注册获取: [https://tushare.pro/](https://tushare.pro/))

### 3. 训练 AI 模型 (可选)

系统已内置了预训练好的模型。如果您想使用最新数据重新训练：

```bash
python train_and_backtest.py
```
*这将自动对比 Random Forest 和 XGBoost 的效果，并保存最佳模型。*

### 4. 运行每日扫描

```bash
python main.py
```
程序运行后，请查看 `reports/` 目录下生成的 Markdown 报告。

---

## 📊 策略表现 (2025.10 - 2026.01)

| 标的 | 收益率 | 胜率 | 评价 |
| :--- | :--- | :--- | :--- |
| **新能源车ETF** | **+13.92%** | 75% | 🚀 准确捕捉主升浪 |
| **半导体ETF** | **+8.14%** | - | 扭亏为盈 |
| **科创50ETF** | **+2.17%** | - | 成功上车 |
| **大盘指数** | 0.00% | 0% | 🛡️ 熊市空仓避险 |

*注：以上数据基于最近 3 个月的实盘回测，不代表未来收益。*
