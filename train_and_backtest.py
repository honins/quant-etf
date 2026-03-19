from datetime import datetime, timedelta

import pandas as pd

from config import tickers
from src.backtest.backtester import Backtester
from src.backtest.hybrid_runner import prepare_index_data
from src.data_loader.data_manager import DataManager
from src.data_loader.tushare_loader import TushareLoader
from src.features.technical import FeatureEngineer
from src.models.model_registry import ModelRegistry
from src.models.xgb_model import XGBoostModel
from src.research.experiment_runner import ExperimentRunner
from src.research.labeler import MultiTaskLabeler
from src.research.validation import (
    ValidationWindow,
    generate_purged_walk_forward_windows,
    generate_walk_forward_windows,
    split_windows_by_regime,
)
from src.strategy.logic import StrategyFilter


PURGE_DAYS = 5


# ============================================================
# 【优化5】 滚动时间窗口交叉验证 (Walk-Forward Validation)
#
# 核心思想：金融市场存在周期漂移 (Regime Shift)，使用单次固定切分
# 容易过拟合到某一特定历史行情。滚动训练更贴近实盘场景：
#   - 每次只用"过去训练窗口"内的数据训练模型
#   - 在紧接着的"测试窗口"内评估模型
#   - 窗口整体向后滚动，汇总所有测试结果评估泛化能力
# ============================================================

# --- 配置参数 ---
TRAIN_WINDOW_DAYS  = 365 * 2   # 训练窗口：2年
TEST_WINDOW_DAYS   = 90        # 测试窗口：3个月
WALK_STEP_DAYS     = 90        # 每次向后滚动：3个月（与测试窗口一致，无重叠）
MIN_TRAIN_SAMPLES  = 200       # 训练数据最少样本数


def load_all_data(data_manager: DataManager, feature_eng: FeatureEngineer, index_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """加载并处理所有标的的历史数据（含相对大盘强弱特征）"""
    dataset = {}
    ticker_list = tickers.get_tradable_ticker_list()
    labeler = MultiTaskLabeler()

    for code in ticker_list:
        print(f"Loading {code}...", end="\r")
        df = data_manager.update_and_get_data(code)
        if df.empty or len(df) < MIN_TRAIN_SAMPLES:
            continue

        df = feature_eng.calculate_technical_indicators(df)
        # 【优化2】注入横截面相对强弱特征（与 main.py 保持一致）
        if not index_df.empty:
            df = feature_eng.add_relative_strength(df, index_df, period=20)
            df = feature_eng.add_regime_features(df, index_df)
        df = labeler.add_labels(df)
        df = df.dropna()
        dataset[code] = df

    print(f"\nLoaded {len(dataset)} tickers with sufficient history.")
    return dataset


def generate_walk_windows(earliest_date: str, latest_date: str) -> list[ValidationWindow]:
    """
    生成滚动训练/测试窗口列表。

    Returns:
        ValidationWindow 列表
    """
    return generate_walk_forward_windows(
        earliest_date=earliest_date,
        latest_date=latest_date,
        train_window_days=TRAIN_WINDOW_DAYS,
        test_window_days=TEST_WINDOW_DAYS,
        step_days=WALK_STEP_DAYS,
        mode="rolling",
    )


def run_walk_forward(
    dataset: dict[str, pd.DataFrame],
    backtester: Backtester,
    market_status_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    执行完整的滚动时间窗口验证，返回每个窗口-每个标的的回测结果列表。
    """
    # 找出所有标的的可用日期范围
    all_dates = []
    for df in dataset.values():
        all_dates.extend(df['trade_date'].astype(str).tolist())
    if not all_dates:
        print("No data available for walk-forward validation.")
        return []

    earliest = min(all_dates)
    latest   = max(all_dates)
    rolling_windows = generate_walk_windows(earliest, latest)

    if not rolling_windows:
        print("Not enough history for walk-forward. Falling back to single split.")
        return _run_single_split(dataset, backtester, latest)

    print(f"\nWalk-Forward Windows: {len(rolling_windows)} folds")
    runner = ExperimentRunner()
    anchored_windows = generate_walk_forward_windows(
        earliest_date=earliest,
        latest_date=latest,
        train_window_days=TRAIN_WINDOW_DAYS,
        test_window_days=TEST_WINDOW_DAYS,
        step_days=WALK_STEP_DAYS,
        mode="anchored",
    )
    purged_windows = generate_purged_walk_forward_windows(
        earliest_date=earliest,
        latest_date=latest,
        train_window_days=TRAIN_WINDOW_DAYS,
        test_window_days=TEST_WINDOW_DAYS,
        step_days=WALK_STEP_DAYS,
        purge_days=PURGE_DAYS,
        mode="rolling",
    )
    regime_windows = split_windows_by_regime(rolling_windows, market_status_map or {})
    print(
        f"Anchored folds: {len(anchored_windows)} | Purged folds: {len(purged_windows)} | Regime buckets: "
        f"bull={len(regime_windows.get('Bull Market', []))} bear={len(regime_windows.get('Bear Market', []))} volatile={len(regime_windows.get('Volatile Market', []))}"
    )
    model_registry = ModelRegistry()
    validation_specs = [
        ("rolling", rolling_windows),
        ("anchored", anchored_windows),
        ("purged", purged_windows),
    ]
    benchmark = runner.run_benchmark_suite(
        dataset=dataset,
        model_specs=[
            ("xgboost", lambda: model_registry.create("xgboost")),
            ("logistic", lambda: model_registry.create("logistic")),
        ],
        validation_specs=validation_specs,
        backtester=backtester,
        threshold=0.6,
        ticker_names=tickers.TICKERS,
        suite_name="model_benchmark",
        market_status_map=market_status_map,
    )
    print(f"Benchmark suite written to: {benchmark['suite_dir']}")

    selection = benchmark.get("selection", {}) if isinstance(benchmark, dict) else {}
    champion = selection.get("champion", {}) if isinstance(selection, dict) else {}
    champion_name = str(champion.get("model_name", "")).strip().lower() if isinstance(champion, dict) else ""
    if champion_name in {"xgboost", "logistic"}:
        persist_result = runner.persist_selected_model(
            dataset=dataset,
            model_name=champion_name,
            model_factory=lambda name=champion_name: model_registry.create(name),
        )
        print(f"Champion persist result: {persist_result}")

    all_results: list[dict] = []
    for run in benchmark["runs"]:
        results = run.get("results", [])
        if isinstance(results, list):
            all_results.extend(results)
    return all_results


def _run_single_split(dataset: dict[str, pd.DataFrame], backtester: Backtester, latest_date: str) -> list[dict]:
    """历史数据不足时退化为单次切分（兼容保底方案）"""
    split_date = (datetime.strptime(latest_date, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")
    print(f"Single-split fallback. Split date: {split_date}")

    train_frames = [frame[frame['trade_date'].astype(str) < split_date] for frame in dataset.values()]
    full_train_df = pd.concat([f for f in train_frames if len(f) >= 60], ignore_index=True)

    model = XGBoostModel()
    model.train(full_train_df)

    results = []
    for code, df in dataset.items():
        test_part = df[df['trade_date'].astype(str) >= split_date].copy()
        if len(test_part) < 10:
            continue
        probs = model.predict_batch(test_part)
        res = backtester.run(test_part, probs, threshold=0.6)
        res['code'] = code
        res['name'] = tickers.TICKERS.get(code, code)
        results.append(res)
    return results


def print_walk_forward_summary(all_results: list[dict]):
    """汇总并打印滚动验证结果"""
    if not all_results:
        print("No results to summarize.")
        return

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("Walk-Forward Validation Summary")
    print("=" * 60)

    # 按标的汇总
    if 'total_return' in df.columns and 'name' in df.columns:
        per_ticker = df.groupby('name', as_index=True)['total_return'].agg(['mean', 'std', 'count'])
        per_ticker = per_ticker.sort_values(by='mean', ascending=False)
        display = per_ticker.rename(columns={'mean': '平均收益率', 'std': '收益率标准差', 'count': '测试折数'}).copy()
        display['平均收益率'] = display['平均收益率'].map(lambda value: f"{value * 100:.2f}%")
        display['收益率标准差'] = display['收益率标准差'].fillna(0.0).map(lambda value: f"{value * 100:.2f}%")
        print("\n[各标的跨折平均表现]")
        print(display.to_string())

    # 总体统计
    if 'total_return' in df.columns:
        avg_ret = df['total_return'].mean()
        print(f"\n整体平均收益率 (all folds, all tickers): {avg_ret*100:.2f}%")
        print(f"总测试样本数 (ticker × fold): {len(df)}")


def main():
    print("Starting Model Training with Walk-Forward Validation (XGBoost)...")
    
    # 1. 初始化
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    backtester = Backtester()
    strat_filter = StrategyFilter()
    
    # 2. 先加载大盘指数数据（用于计算相对强弱特征，保持与 main.py 一致）
    print("Loading index data (000300.SH)...")
    index_df = data_manager.update_and_get_data('000300.SH', is_index=True)
    if not index_df.empty:
        index_df = feature_eng.calculate_technical_indicators(index_df)
        print(f"  Index data: {len(index_df)} rows loaded.")
    else:
        print("  Index data unavailable, rs_20d / rel_vol features will be NaN.")

    market_status_map: dict[str, str] = {}
    if not index_df.empty:
        _, market_status_map = prepare_index_data(data_manager, feature_eng, strat_filter, index_code='000300.SH')

    # 3. 加载所有标的数据（含相对大盘强弱特征）
    print("Loading and preparing data...")
    dataset = load_all_data(data_manager, feature_eng, index_df)

    if not dataset:
        print("No usable dataset. Exiting.")
        return

    # 4. Walk-Forward 验证
    all_results = run_walk_forward(dataset, backtester, market_status_map=market_status_map)

    # 5. 汇总报告
    print_walk_forward_summary(all_results)

    # 6. 用全量数据重新训练最终模型并保存 (供 main.py 实盘使用)
    print("\nTraining final model on ALL available data...")
    all_frames = list(dataset.values())
    full_df = pd.concat(all_frames, ignore_index=True)

    final_model = XGBoostModel(model_path="data/xgb_model.json")
    final_model.train(full_df)
    if getattr(final_model, "is_trained", False):
        final_model.save_model()
        print("Final model saved to data/xgb_model.json")
    else:
        print("Final XGBoost model not saved because training is unavailable in this environment.")


if __name__ == "__main__":
    main()
