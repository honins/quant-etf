# Quant-ETF 研究升级精简路线图

这份路线图只保留当前最值得先做的部分，目标不是一次性把系统做复杂，而是先把研究质量、验证质量和实盘可用性拉起来。

适用前提：

- 当前仓库已经有可运行主链路
- 当前主模型仍以 `src/models/xgb_model.py` 为主
- 近期目标是提高结果可信度，而不是堆更多模型

## 1. 先做什么，不先做什么

### 现在先做

1. 统一实验与回测输出
2. 强化验证，减少“看起来有效、实际上过拟合”
3. 升级标签与特征，让 XGBoost baseline 先吃到更好的输入

### 现在不做

- 文本因子
- 分钟级特征
- 多模型集成
- 组合优化器
- 强化学习

原因很简单：如果验证和基线还不够扎实，后面这些都会放大噪音，不会稳定提高实盘质量。

## 2. 当前仓库最重要的 3 个瓶颈

1. 验证还不够严格

`train_and_backtest.py` 已经有 walk-forward，但还缺更标准的实验输出、窗口分布统计和更强的稳定性检查。

2. 标签过于单一

当前标签设计偏单一二分类，不利于区分“涨得快”“涨得慢”“先止损还是先止盈”这些交易上很重要的差异。

3. 回测输出不够标准化

`src/backtest/backtester.py` 能跑，但还没有形成稳定的实验产物格式，后续做比较、复现和 dashboard 接入都会比较累。

## 3. 第一阶段目标

先用 4 周完成一个可持续迭代的研究底座。

完成后应该达到：

- 任意实验都能复现
- 任意窗口表现都能单独查看
- baseline 的收益、回撤、稳定性可统一比较
- 新标签和新特征能被快速接入现有 XGBoost 流程

## 4. 4 周执行计划

### 第 1 周：统一实验输出

目标：先让实验结果可追踪、可复现。

要做的事：

1. 新建 `src/research/metrics.py`
2. 新建 `src/research/experiment_runner.py`
3. 统一 `src/backtest/backtester.py` 输出结构
4. 约定实验结果目录

建议输出格式：

```text
reports/experiments/
  YYYYMMDD_HHMMSS/
    config.json
    metrics.json
    trades.csv
    equity_curve.csv
```

验收标准：

- 同一次实验的配置、指标、交易记录能完整落盘
- 不同实验的结果格式一致

### 第 2 周：升级标签

目标：先提升样本表达能力，不急着换模型。

要做的事：

1. 新建 `src/research/labeler.py`
2. 保留旧标签作为 baseline 对照
3. 新增少量最有价值的标签

建议第一批标签只保留这 5 个：

- `future_ret_5d`
- `future_ret_10d`
- `future_max_ret_10d`
- `future_min_ret_10d`
- `hit_stop_loss_first`

验收标准：

- 标签生成无未来信息泄露
- 训练脚本可通过参数切换新旧标签
- 新标签能直接接进当前 XGBoost 流程

### 第 3 周：拆分特征层

目标：让特征工程开始模块化，但不要一次拆太多。

要做的事：

1. 保留 `src/features/technical.py` 兼容入口
2. 拆出 `price_features.py`
3. 拆出 `regime_features.py`
4. 只补一小批高价值特征

第一批建议新增：

- 近 20 日收益排名
- 近 20 日波动率排名
- ETF 相对全池强弱分位
- 大盘趋势状态
- 市场波动率状态

验收标准：

- 特征组可单独开关
- 新特征能被 `main.py` 和训练脚本复用

### 第 4 周：强化验证

目标：判断模型是否真的稳定，而不是只看平均收益。

要做的事：

1. 新建 `src/research/validation.py`
2. 在 `train_and_backtest.py` 中接入统一验证模式
3. 输出窗口级结果汇总

第一阶段只做这 3 个验证方式：

- anchored walk-forward
- rolling walk-forward
- bull / bear / volatile 分 regime 统计

验收标准：

- 能自动输出各窗口收益、回撤、Sharpe 分布
- 能看到最差窗口表现
- 新方案要先打赢当前 baseline，才继续推进

## 5. 第一阶段完成后的判断标准

只有满足下面条件，才值得进入下一阶段：

1. 新标签或新特征在多个 OOS 窗口有效
2. 不是只提升收益，同时没有明显恶化回撤
3. 结果不依赖单一 ETF
4. 实验结果可复现

如果这四条做不到，就不要继续上更复杂的模型。

## 6. 暂缓项

下面这些方向不是不要做，而是明确后移：

- LightGBM / CatBoost / Ranking model
- 分钟级数据
- 文本结构化因子
- 集成模型
- 组合优化
- Meta execution model
- 强化学习

建议顺序是：

先把 baseline 研究底座做好，再决定是否值得扩展模型族。

## 7. 明天就能开工的清单

如果现在就开始做，建议按这个顺序推进：

1. 新建 `src/research/metrics.py`
2. 新建 `src/research/experiment_runner.py`
3. 标准化 `src/backtest/backtester.py` 输出
4. 新建 `src/research/labeler.py`
5. 改造 `train_and_backtest.py`

这 5 项做完，再开始拆特征和补验证。
