# Quant-ETF 策略强化实施路线图

本文档面向“已有自有大模型供应商、算力充足、不优先考虑成本”的前提，目标不是做一个更花哨的 AI 版本，而是系统性提高策略的真实可交易效果。

核心原则：

- 先提高研究与验证质量，再提高模型复杂度
- 先做 point-in-time 数据和严格 OOS，再做大模型和强化学习
- 先优化组合层，再继续打磨单标的分数
- 所有升级都要能落到当前仓库结构里，避免空泛方案

## 1. 当前系统的真实瓶颈

结合当前仓库，主要瓶颈不是算力，而是以下 5 点：

1. 标签过于单一
当前主要在 `src/features/technical.py` 里构建固定 horizon 二分类标签，容易把“不同上涨路径”压成一个目标。

2. 模型族太单一
当前实质主模型是 `src/models/xgb_model.py`，虽然强，但对时序结构、横截面排序、多市场状态切换的表达仍有限。

3. 回测仍偏单标的信号回放
`src/backtest/backtester.py` 目前更接近“单标的进出场模拟器”，还不是完整的组合级模拟器。

4. 验证框架还不够强
`train_and_backtest.py` 已有 walk-forward，但还缺 purged split、regime split、模型稳定性分析、过拟合防护。

5. 文本与多周期信息没有进入主模型
你现在的优势是大模型和算力，但仓库还没有把新闻、公告、政策、分钟级结构真正转成 point-in-time 因子。

## 2. 总体目标

3 个月内把系统从：

- 日线技术因子
- 单模型二分类
- 阈值过滤入场
- 单标的回测

升级为：

- 多周期数值 + 文本结构化特征
- 多任务预测 + 横截面排序
- 多模型集成 + Regime 专家模型
- 组合级优化与回测
- 严格 OOS 与稳定性评估

## 3. 交付目标

3 个月后，至少交付以下成果：

1. 新的数据层
- 日线、分钟线、指数上下文、文本事件、资金流统一入库

2. 新的特征与标签层
- 多任务标签、横截面特征、文本因子、regime 特征

3. 新的模型层
- XGBoost baseline
- GBDT 集成
- 时序模型
- 多模态融合模型

4. 新的验证层
- Purged walk-forward
- Regime OOS
- 稳定性与过拟合检测

5. 新的组合层
- top-k 选标
- 风险预算
- 相关性约束
- 换手惩罚

## 4. 目标仓库结构

建议在现有结构基础上新增以下目录：

```text
src/
  research/
    dataset_builder.py
    labeler.py
    validation.py
    metrics.py
    experiment_runner.py
  portfolio/
    optimizer.py
    allocator.py
    constraints.py
  features/
    technical.py
    price_features.py
    cross_section_features.py
    regime_features.py
    text_features.py
  models/
    xgb_model.py
    lgbm_model.py
    catboost_model.py
    ranking_model.py
    timeseries_model.py
    ensemble.py
    model_registry.py
```

当前文件的建议角色调整：

- `src/features/technical.py`
  保留兼容入口，但内部逐步拆分

- `src/models/xgb_model.py`
  降级为 baseline，而不是唯一主模型

- `train_and_backtest.py`
  改造成实验编排入口

- `src/backtest/backtester.py`
  只负责执行模拟，组合逻辑抽到 `portfolio/`

- `src/strategy/logic.py`
  收缩成硬风控和 regime filter，不再承载主 alpha 逻辑

## 5. 三个月按周执行计划

### 第 1 月：研究底座重构

目标：

- 让“好策略”和“过拟合策略”能被分开
- 建立新数据与标签框架
- 不急着追求更高收益

#### 第 1 周：研究框架搭底

本周交付：

- 新增 `src/research/`
- 新增统一实验配置
- 统一回测输入输出格式

具体任务：

1. 新建 `src/research/experiment_runner.py`
- 输入：数据窗口、模型名、特征集、标签配置、组合配置
- 输出：单次实验结果对象

2. 新建 `src/research/metrics.py`
- 指标至少包含：
  - annual_return
  - max_drawdown
  - sharpe
  - calmar
  - turnover
  - avg_holding_days
  - win_rate
  - regime_breakdown

3. 把 `src/backtest/backtester.py` 的输出结构标准化
- 要能稳定输出 trade list、equity curve、daily pnl、turnover、position history

4. 新增实验结果目录约定

```text
reports/experiments/
  YYYYMMDD_HHMMSS/
    config.json
    metrics.json
    trades.csv
    equity_curve.csv
    feature_importance.csv
```

验收标准：

- 任意一次实验都能被复现实验配置
- 不同模型输出结构一致
- dashboard 后续可直接读取实验结果

#### 第 2 周：标签重构

本周交付：

- 从单一二分类标签升级为多任务标签

具体任务：

1. 新建 `src/research/labeler.py`

标签至少包括：

- `future_ret_1d`
- `future_ret_3d`
- `future_ret_5d`
- `future_ret_10d`
- `future_ret_20d`
- `future_max_ret_10d`
- `future_min_ret_10d`
- `hit_take_profit_first`
- `hit_stop_loss_first`
- `best_holding_days`

2. 保留旧标签，作为 baseline 对照

3. 衍生 meta-label
- 基础信号命中后，是否值得执行
- 用于后续“是否做这笔交易”的二阶段模型

验收标准：

- 同一个样本可同时拥有分类、回归、排序标签
- 标签生成无未来信息泄露
- 标签配置可以通过参数切换

#### 第 3 周：特征层拆分

本周交付：

- 特征层不再只有一份 `technical.py`

具体任务：

1. 从 `src/features/technical.py` 中拆出：
- `price_features.py`
- `cross_section_features.py`
- `regime_features.py`

2. 新增横截面特征
- ETF 相对全池分位数
- ETF 相对同类分位数
- 近 20 日相对强弱排名
- 近 20 日波动率排名
- 成交额活跃度排名

3. 新增 regime 特征
- 大盘趋势状态
- 市场宽度
- 风格偏好
- 波动率环境
- 量能环境

验收标准：

- 每组特征可单独启用/关闭
- 可以做 ablation study

#### 第 4 周：验证框架升级

本周交付：

- 严格 OOS 验证

具体任务：

1. 新建 `src/research/validation.py`

至少实现：

- anchored walk-forward
- rolling walk-forward
- purged split
- bull / bear / volatile regime split

2. 给 `train_and_backtest.py` 增加实验模式

3. 新增稳定性报告
- 各窗口表现分布
- 各 ETF 表现分布
- 各 regime 表现分布

验收标准：

- 同一模型能在多个窗口自动跑完整验证
- 输出不仅有平均值，还要有分布和最差窗口表现

### 第 2 月：Alpha 引擎升级

目标：

- 扩展模型族
- 引入多周期与文本因子
- 做出比单一 XGBoost 明显更强的候选体系

#### 第 5 周：Baseline 模型族搭建

本周交付：

- 多模型统一接口

具体任务：

1. 新建 `src/models/model_registry.py`
- 统一注册模型名、特征需求、训练入口、预测入口

2. 新增模型：
- `lgbm_model.py`
- `catboost_model.py`
- `ranking_model.py`

3. 让 `src/models/xgb_model.py` 只做一件事：baseline tabular learner

4. 所有模型统一输出：
- score
- expected_return
- downside_risk
- confidence

验收标准：

- 任意模型都能用统一实验脚本跑
- baseline 对比结果能自动汇总

#### 第 6 周：分钟级与多周期特征

本周交付：

- 多周期数据进入主数据流

具体任务：

1. 扩展 `src/data_loader/data_manager.py`
- 支持 `1m`、`5m`、`30m` 级别数据表
- 所有数据按 point-in-time 规则存取

2. 新增分钟级特征：
- 开盘后 30 分钟方向
- VWAP 偏离
- 量价冲击
- intraday trend persistence
- morning reversal / breakout

3. 把分钟级特征聚合回日级样本

验收标准：

- 不直接把未来分钟信息泄露给当日决策
- 日线样本可稳定挂载分钟级摘要特征

#### 第 7 周：文本结构化因子

本周交付：

- 让你的大模型真正进入策略主链路

具体任务：

1. 新建 `src/features/text_features.py`

2. 建文本处理链：
- 新闻抓取
- 公告抓取
- 研报抓取
- 政策文本抓取

3. 用你的大模型做结构化抽取，不直接输出买卖建议，只输出：
- 主题标签
- 情绪方向
- 利多/利空强度
- 涉及行业
- 时效性
- 是否是政策级事件
- 是否影响 ETF 主题主线

4. 生成可回测的日级文本因子

验收标准：

- 文本因子有明确时间戳
- 文本因子与行情特征能 join 到同一训练样本
- 至少完成 3 类文本源接入

#### 第 8 周：集成与专家模型

本周交付：

- 多模型集成与 regime 分专家

具体任务：

1. 新建 `src/models/ensemble.py`

集成形式建议：

- level-0:
  - XGBoost
  - LightGBM
  - CatBoost
  - Ranking model
  - Timeseries model

- level-1:
  - meta learner

2. 按 regime 训练专家模型
- bull 专家
- bear 专家
- volatile 专家

3. 做 champion / challenger 框架

验收标准：

- 新模型必须能在严格 OOS 上打赢 XGBoost baseline，才进候选池

### 第 3 月：组合与执行升级

目标：

- 从“买不买”升级到“买哪些、各买多少、何时换仓”

#### 第 9 周：组合优化器

本周交付：

- 新建 `src/portfolio/optimizer.py`

具体任务：

1. 优化目标至少支持：
- 最大化预期收益
- 惩罚回撤
- 惩罚波动
- 惩罚换手
- 惩罚高度相关

2. 输入：
- expected_return
- downside_risk
- confidence
- pairwise correlation
- liquidity

3. 输出：
- 组合权重
- top-k 持仓列表
- 预期换手

验收标准：

- 每日不再只生成信号，而是生成组合方案

#### 第 10 周：执行与持有期模型

本周交付：

- 二阶段执行模型

具体任务：

1. 新增 meta-label 执行模型
- 第一阶段模型负责“找候选”
- 第二阶段模型负责“这笔要不要做、做多大、持多久”

2. 新增持有期建议
- 1d / 3d / 5d / 10d / 20d 最优持有期

3. 新增减仓逻辑
- 分批止盈
- 回撤减仓
- 信心衰减减仓

验收标准：

- 输出不再是单个阈值，而是完整执行建议

#### 第 11 周：组合级回测与归因

本周交付：

- 组合级回测成为主评估路径

具体任务：

1. 扩展 `src/backtest/backtester.py` 或拆出组合回测器

2. 加入：
- 成本模型
- 滑点模型
- 容量约束
- 再平衡频率
- 最大单标的权重
- 最大行业/主题暴露

3. 新增归因报告
- 收益来自哪些 ETF
- 收益来自哪些 regime
- 收益来自哪些特征组

验收标准：

- 结果可解释
- 组合收益、回撤、换手、容量一并可看

#### 第 12 周：Dashboard 与研究工作流整合

本周交付：

- 研究结果进入工作台

具体任务：

1. 在 Dashboard 中新增研究面板

建议新增能力：

- champion 模型
- challenger 模型
- 最近实验排行榜
- 各 regime 表现对比
- 特征组 ablation 结果
- 组合建议而不只是单标的信号

2. 让日报包含：
- 当前 champion 模型
- 近 20 次实验最佳配置
- 风险提示

验收标准：

- 研究结果和实盘工作流不再割裂

## 6. 数据层详细实施建议

### 第一优先级数据

- ETF 日线 OHLCV
- ETF 1m / 5m / 30m
- 沪深 300、中证 1000、创业板指等指数
- ETF 份额变化
- 成交额与换手
- 宏观基准：利率、汇率、商品、海外股指

### 第二优先级数据

- 北向资金
- 板块强弱
- 两融数据
- 资金流向
- 情绪指标

### 第三优先级数据

- 新闻
- 公告
- 政策
- 研报
- 社媒情绪

### 数据表建议

```text
daily_data
index_daily_data
etf_intraday_1m
etf_intraday_5m
etf_intraday_30m
etf_flow_data
macro_daily_data
text_events
experiment_results
```

要求：

- 所有表都有 `as_of_time` 或等价字段
- 所有文本因子都能追溯到源文本和抓取时间

## 7. 模型路线建议

### 必做模型

1. XGBoost
- 继续保留，做最强 baseline

2. LightGBM
- 更适合大规模搜索

3. CatBoost
- 对类别与噪声稳健

4. Ranking model
- 每日横截面排序比绝对阈值更适合 ETF 池选择

5. Timeseries model
- 用于多周期序列表征

### 第二阶段模型

1. 多模态融合模型
- 数值 + 文本因子融合

2. Regime Mixture-of-Experts
- 按市场状态切换专家

3. Meta execution model
- 决定做不做、做多大、持多久

### 暂缓模型

- 强化学习
- 端到端大语言模型直接预测价格

原因：

- 在 point-in-time 数据、组合仿真、执行日志都没完全成熟前，投入产出比不高

## 8. 验证规则

任何新模型进入候选池，必须满足：

1. 比当前 XGBoost baseline 更优
- 不是只赢平均收益
- 至少还要赢稳定性或回撤

2. 在多个 OOS 窗口成立

3. 在多个 market regime 下不崩

4. 加入成本后仍然成立

5. 不依赖单一 ETF 或单一行情阶段

## 9. 推荐的实验优先顺序

建议不要同时铺太多线，优先级如下：

1. 多任务标签
2. 横截面排序模型
3. 组合优化器
4. 分钟级特征
5. 文本结构化因子
6. 集成模型
7. Regime 专家模型
8. Meta execution model
9. 多模态融合
10. 强化学习

## 10. 第一阶段可直接开工的开发清单

如果明天开始做，建议按下面顺序直接落：

### Sprint A

- 新建 `src/research/metrics.py`
- 新建 `src/research/labeler.py`
- 新建 `src/research/validation.py`
- 重构 `train_and_backtest.py`

完成标准：

- 能跑多标签实验
- 能跑多窗口验证
- 能输出统一 metrics.json

### Sprint B

- 拆分 `src/features/technical.py`
- 新建 `cross_section_features.py`
- 新建 `regime_features.py`
- 给 `src/data_loader/data_manager.py` 增加多周期数据支持

完成标准：

- 特征分层清楚
- 日线样本可挂载多周期特征

### Sprint C

- 新建 `model_registry.py`
- 新建 `lgbm_model.py`
- 新建 `ranking_model.py`
- 跑 baseline 对比

完成标准：

- 至少 3 个模型统一比较

### Sprint D

- 新建 `portfolio/optimizer.py`
- 组合级回测
- 加成本、换手、约束

完成标准：

- Dashboard 不再只展示分数，而是展示组合建议

## 11. 你现在最不该做的事

1. 继续围绕固定阈值小修小补

2. 直接让大模型输出“买哪个 ETF”

3. 在弱验证框架下做海量搜索

4. 先做 RL 再补数据和回测

## 12. 最终判断标准

3 个月后，不是看“模型更复杂了”，而是看是否达成下面 4 点：

1. 新系统在严格 OOS 下稳定优于当前基线

2. 回撤、换手、容量纳入主目标

3. 文本和分钟级数据真正进入主模型

4. 工作台输出从“单标的打分”升级为“组合级执行建议”

## 13. 一句话建议

你的优势不是“可以训练一个更大的模型”，而是“可以把数值、文本、多周期、组合优化和严格验证一起做对”。真正能把策略效果拉开的，通常是这套系统工程，而不是单独某个模型名字。
