# 算法说明

## 目标

项目不是预测明天的精确价格，而是为每个 ETF 生成一个 `0~1` 的交易评分，再结合市场状态和风控规则决定是否入场。

## 数据与特征

- 行情来源: Tushare
- 数据缓存: `data/market_data.db`
- 主要特征在 `src/features/technical.py`

当前特征大致分为几类：

- 趋势: 均线、均线斜率、突破、价格位置
- 动量: RSI、MACD、涨跌幅
- 波动: ATR、布林带、回撤
- 量能: 成交量相对均量、量价配合
- 相对强弱: 相对沪深 300 的强弱特征

## 标签

训练标签不是旧版的“未来 5 天最高涨幅是否超过 2%”单一条件，而是更贴近交易收益的多维标签。核心参数在 `config/settings.py`：

- `TRAIN_LABEL_HORIZON = 7`
- `TRAIN_LABEL_THRESHOLD = 0.025`
- `TRAIN_LABEL_END_WEIGHT = 0.30`
- `TRAIN_LABEL_DRAWDOWN_PENALTY = 1.20`

标签构造会同时考虑：

- 未来窗口内最大上涨空间
- 未来窗口结束时收益
- 未来窗口内最大回撤惩罚

## 模型

- 主模型: `XGBoostModel`
- 兜底模型: `RuleBasedModel`

`RuleBasedModel` 只在没有训练好模型文件时兜底，不参与当前主策略优化。

## 信号过滤

`src/strategy/logic.py` 负责市场状态识别和阈值过滤：

- 牛市: 用牛市阈值
- 震荡市: 用更保守阈值
- 熊市: 只允许极高分信号
- 部分激进标的和部分指定标的可使用单独阈值
- 动态阈值只在允许的标的和回测模式下启用

## 风控

- 初始止损: 基于 ATR
- 移动止盈: 基于最近高点和 ATR
- 组合参数统一由 `StrategyConfig` 管理

## 回测

当前单标的回测统一走 `src/backtest/hybrid_runner.py`，不再维护多套并行回测逻辑。

统一输出的指标包括：

- 收益率
- 胜率
- 交易次数
- 最大回撤
- 波动率
- 夏普

## 优化方式

- `optimize_strategy.py` 负责搜索策略参数
- `train_and_backtest.py` 负责 walk-forward 训练验证
- 当前优化目标不是单看收益，而是收益优先，同时惩罚回撤和波动
