import pandas as pd
from datetime import datetime, timedelta
from config import tickers
from src.data_loader.tushare_loader import TushareLoader
from src.data_loader.data_manager import DataManager
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.backtest.backtester import Backtester


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
    ticker_list = tickers.get_ticker_list()

    for code in ticker_list:
        print(f"Loading {code}...", end="\r")
        df = data_manager.update_and_get_data(code)
        if df.empty or len(df) < MIN_TRAIN_SAMPLES:
            continue

        df = feature_eng.calculate_technical_indicators(df)
        # 【优化2】注入横截面相对强弱特征（与 main.py 保持一致）
        if not index_df.empty:
            df = feature_eng.add_relative_strength(df, index_df, period=20)
        df = feature_eng.add_labels(df)
        df = df.dropna()
        dataset[code] = df

    print(f"\nLoaded {len(dataset)} tickers with sufficient history.")
    return dataset


def generate_walk_windows(earliest_date: str, latest_date: str) -> list[dict]:
    """
    生成滚动训练/测试窗口列表。

    Returns:
        list of dicts, each with keys: train_start, train_end, test_start, test_end (均为 "%Y%m%d" 格式)
    """
    fmt = "%Y%m%d"
    start = datetime.strptime(earliest_date, fmt)
    end   = datetime.strptime(latest_date, fmt)
    train_window = timedelta(days=TRAIN_WINDOW_DAYS)
    test_window  = timedelta(days=TEST_WINDOW_DAYS)
    step         = timedelta(days=WALK_STEP_DAYS)

    windows = []
    train_start = start
    while True:
        train_end  = train_start + train_window
        test_start = train_end + timedelta(days=1)
        test_end   = test_start + test_window

        if test_end > end:
            break

        windows.append({
            "train_start": train_start.strftime(fmt),
            "train_end":   train_end.strftime(fmt),
            "test_start":  test_start.strftime(fmt),
            "test_end":    test_end.strftime(fmt),
        })
        train_start += step

    return windows


def run_walk_forward(dataset: dict[str, pd.DataFrame], backtester: Backtester) -> list[dict]:
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
    windows  = generate_walk_windows(earliest, latest)

    if not windows:
        print("⚠️ Not enough history for walk-forward. Falling back to single split.")
        return _run_single_split(dataset, backtester, latest)

    print(f"\n📅 Walk-Forward Windows: {len(windows)} folds")
    all_results = []

    for i, w in enumerate(windows):
        print(f"\n--- Fold {i+1}/{len(windows)} | Train: {w['train_start']}~{w['train_end']} | Test: {w['test_start']}~{w['test_end']} ---")

        # 构建跨标的训练集
        train_frames = []
        for df in dataset.values():
            dates = df['trade_date'].astype(str)
            chunk = df[(dates >= w['train_start']) & (dates <= w['train_end'])]
            if len(chunk) >= 60:
                train_frames.append(chunk)

        if not train_frames:
            print("  Skipping: insufficient training data in this window.")
            continue

        full_train_df = pd.concat(train_frames, ignore_index=True)

        # 训练 XGBoost
        model = XGBoostModel()
        model.train(full_train_df)

        if not model.is_trained:
            print("  Training failed. Skipping.")
            continue

        # 回测测试集
        for code, df in dataset.items():
            dates = df['trade_date'].astype(str)
            test_part = df[(dates >= w['test_start']) & (dates <= w['test_end'])].copy()
            if len(test_part) < 10:
                continue

            probs = model.predict_batch(test_part)
            res = backtester.run(test_part, probs, threshold=0.6)
            res['code'] = code
            res['name'] = tickers.TICKERS.get(code, code)
            res['fold'] = i + 1
            res['test_start'] = w['test_start']
            res['test_end'] = w['test_end']
            all_results.append(res)

    return all_results


def _run_single_split(dataset: dict, backtester: Backtester, latest_date: str) -> list[dict]:
    """历史数据不足时退化为单次切分（兼容保底方案）"""
    split_date = (datetime.strptime(latest_date, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")
    print(f"Single-split fallback. Split date: {split_date}")

    train_frames = [df[df['trade_date'].astype(str) < split_date] for df in dataset.values()]
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
    print("📊 Walk-Forward Validation Summary")
    print("=" * 60)

    # 按标的汇总
    if 'total_return' in df.columns:
        per_ticker = df.groupby('name')['total_return'].agg(['mean', 'std', 'count'])
        per_ticker.columns = ['平均收益率', '收益率标准差', '测试折数']
        per_ticker['平均收益率'] = per_ticker['平均收益率'].map(lambda x: f"{x*100:.2f}%")
        per_ticker['收益率标准差'] = per_ticker['收益率标准差'].map(lambda x: f"{x*100:.2f}%")
        print("\n[各标的跨折平均表现]")
        print(per_ticker.sort_values('平均收益率', ascending=False).to_string())

    # 总体统计
    if 'total_return' in df.columns:
        avg_ret = df['total_return'].mean()
        print(f"\n整体平均收益率 (all folds, all tickers): {avg_ret*100:.2f}%")
        print(f"总测试样本数 (ticker × fold): {len(df)}")


def main():
    print("🧠 Starting Model Training with Walk-Forward Validation (XGBoost)...")
    
    # 1. 初始化
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    backtester = Backtester()
    
    # 2. 先加载大盘指数数据（用于计算相对强弱特征，保持与 main.py 一致）
    print("📊 Loading index data (000300.SH)...")
    index_df = data_manager.update_and_get_data('000300.SH', is_index=True)
    if not index_df.empty:
        index_df = feature_eng.calculate_technical_indicators(index_df)
        print(f"  Index data: {len(index_df)} rows loaded.")
    else:
        print("  ⚠️ Index data unavailable, rs_20d / rel_vol features will be NaN.")

    # 3. 加载所有标的数据（含相对大盘强弱特征）
    print("📦 Loading and preparing data...")
    dataset = load_all_data(data_manager, feature_eng, index_df)

    if not dataset:
        print("❌ No usable dataset. Exiting.")
        return

    # 4. Walk-Forward 验证
    all_results = run_walk_forward(dataset, backtester)

    # 5. 汇总报告
    print_walk_forward_summary(all_results)

    # 6. 用全量数据重新训练最终模型并保存 (供 main.py 实盘使用)
    print("\n🔧 Training final model on ALL available data...")
    all_frames = list(dataset.values())
    full_df = pd.concat(all_frames, ignore_index=True)

    final_model = XGBoostModel(model_path="data/xgb_model.json")
    final_model.train(full_df)
    final_model.save_model()
    print("✅ Final model saved to data/xgb_model.json")


if __name__ == "__main__":
    main()
