import os
from datetime import datetime, timedelta

import numpy as np

from config import tickers
from config.settings import settings
from src.backtest.backtester import Backtester
from src.backtest.hybrid_runner import (
    build_adjusted_probs,
    build_data_cache,
    prepare_index_data,
    run_backtest_for_cache,
    summarize_results,
    use_dynamic_for_code,
)
from src.data_loader.data_manager import DataManager
from src.data_loader.tushare_loader import TushareLoader
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter


def main():
    print("📉 Running Backtest for Recent Window (Hybrid Strategy)...")
    grid_thresholds_env = os.getenv("GRID_THRESHOLDS", "").strip()
    grid_tickers_env = os.getenv("GRID_TICKERS", "588000.SH,515070.SH")
    select_mode_env = os.getenv("SELECT_MODE", "").strip().lower()
    select_mode = select_mode_env in ("1", "true", "yes", "y")
    grid_thresholds = []
    if grid_thresholds_env:
        grid_thresholds = [float(x.strip()) for x in grid_thresholds_env.split(",") if x.strip()]

    use_dynamic_env = os.getenv("USE_DYNAMIC_THRESHOLD", "").strip().lower()
    if use_dynamic_env:
        settings.USE_DYNAMIC_THRESHOLD = use_dynamic_env in ("1", "true", "yes", "y")

    overrides_env = os.getenv("OVERRIDE_THRESHOLDS", "").strip()
    override_thresholds = None
    if overrides_env:
        override_thresholds = {}
        for item in overrides_env.split(","):
            if "=" not in item:
                continue
            code, val = item.split("=", 1)
            code = code.strip()
            val = val.strip()
            if code and val:
                override_thresholds[code] = float(val)

    lookback_days_env = os.getenv("LOOKBACK_DAYS", "").strip()
    lookback_days = int(lookback_days_env) if lookback_days_env else 90
    train_days_env = os.getenv("TRAIN_DAYS", "").strip()
    test_days_env = os.getenv("TEST_DAYS", "").strip()
    train_days = int(train_days_env) if train_days_env else 180
    test_days = int(test_days_env) if test_days_env else 90
    if select_mode and lookback_days < (train_days + test_days):
        train_days = max(30, int(lookback_days * 2 / 3))
        test_days = max(10, lookback_days - train_days)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    start_date_str = start_date.strftime("%Y%m%d")
    print(f"Period: {start_date_str} - {end_date.strftime('%Y%m%d')}")

    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    backtester = Backtester()
    strat_filter = StrategyFilter()

    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("❌ XGBoost model not found. Please train first.")
        return
    print("✅ XGBoost model loaded.")

    print("📊 Preparing Market Index Data...")
    index_df, market_status_map = prepare_index_data(data_manager, feature_eng, strat_filter, index_code="000300.SH")

    ticker_list = tickers.get_ticker_list()
    if grid_thresholds:
        ticker_list = [t.strip() for t in grid_tickers_env.split(",") if t.strip()]

    data_cache = build_data_cache(
        ticker_list,
        data_manager,
        feature_eng,
        index_df,
        model,
        start_date_str,
    )

    def run_with_overrides(threshold_overrides: dict[str, float] | None):
        return run_backtest_for_cache(
            data_cache,
            backtester,
            market_status_map,
            threshold_overrides=threshold_overrides,
        )

    if grid_thresholds:
        print("\n" + "=" * 80)
        print(f"📅 Backtest Grid: Last 3 Months ({start_date_str} - Now)")
        print("Strategy: XGBoost Signal + Bear Market Filter (Hybrid)")
        print("=" * 80)
        print(f"{'Threshold':<10} {'Name':<12} {'Return':<10} {'WinRate':<10} {'Trades':<8}")
        print("-" * 80)
        for threshold in grid_thresholds:
            overrides = {code: threshold for code in ticker_list}
            results = run_with_overrides(overrides)
            for res in results:
                win_rate_str = f"{res['win_rate'] * 100:.1f}%"
                print(f"{threshold:<10.2f} {res['name']:<12} {res['total_return'] * 100:6.2f}%    {win_rate_str:<10} {res['num_trades']:<8}")
        print("=" * 80)
        return

    results = []
    diff_mode_env = os.getenv("DIFF_MODE", "").strip().lower()
    diff_mode = diff_mode_env in ("1", "true", "yes", "y")

    if select_mode:
        test_start_date = end_date - timedelta(days=test_days)
        test_start_str = test_start_date.strftime("%Y%m%d")
        for code, payload in data_cache.items():
            test_df = payload["test_df"]
            probs = payload["probs"]
            train_df = test_df[test_df["trade_date"].astype(str) < test_start_str].copy()
            train_probs = probs[: len(train_df)]
            eval_df = test_df[test_df["trade_date"].astype(str) >= test_start_str].copy()
            eval_probs = probs[len(train_df) :]
            if len(train_df) < 20 or len(eval_df) < 10:
                use_dynamic = use_dynamic_for_code(code, override_thresholds)
                adjusted_probs, bear_days = build_adjusted_probs(
                    test_df,
                    probs,
                    market_status_map,
                    code,
                    use_dynamic,
                    override_thresholds,
                )
                res = backtester.run(test_df, adjusted_probs, threshold=0.0, code=code)
                res["code"] = code
                res["name"] = tickers.TICKERS[code]
                res["bear_days"] = bear_days
                res["mode"] = "dynamic" if use_dynamic else "fixed"
                results.append(res)
                continue

            train_dynamic_probs, _ = build_adjusted_probs(
                train_df,
                train_probs,
                market_status_map,
                code,
                True,
                None,
            )
            train_fixed_probs, _ = build_adjusted_probs(
                train_df,
                train_probs,
                market_status_map,
                code,
                False,
                override_thresholds,
            )
            train_dynamic = backtester.run(train_df, train_dynamic_probs, threshold=0.0, code=code)
            train_fixed = backtester.run(train_df, train_fixed_probs, threshold=0.0, code=code)
            choose_dynamic = False
            if train_dynamic["sharpe"] > train_fixed["sharpe"]:
                choose_dynamic = True
            elif train_dynamic["sharpe"] == train_fixed["sharpe"] and train_dynamic["total_return"] > train_fixed["total_return"]:
                choose_dynamic = True

            eval_probs_adj, bear_days = build_adjusted_probs(
                eval_df,
                eval_probs,
                market_status_map,
                code,
                choose_dynamic,
                override_thresholds if not choose_dynamic else None,
            )
            res = backtester.run(eval_df, eval_probs_adj, threshold=0.0, code=code)
            res["code"] = code
            res["name"] = tickers.TICKERS[code]
            res["bear_days"] = bear_days
            res["mode"] = "dynamic" if choose_dynamic else "fixed"
            results.append(res)
    elif diff_mode:
        dynamic_results = []
        fixed_results = []
        for code, payload in data_cache.items():
            test_df = payload["test_df"]
            probs = payload["probs"]
            dyn_probs, dyn_bear = build_adjusted_probs(
                test_df,
                probs,
                market_status_map,
                code,
                True,
                None,
            )
            fix_probs, fix_bear = build_adjusted_probs(
                test_df,
                probs,
                market_status_map,
                code,
                False,
                override_thresholds,
            )
            dyn_res = backtester.run(test_df, dyn_probs, threshold=0.0, code=code)
            dyn_res["code"] = code
            dyn_res["name"] = tickers.TICKERS[code]
            dyn_res["bear_days"] = dyn_bear
            fix_res = backtester.run(test_df, fix_probs, threshold=0.0, code=code)
            fix_res["code"] = code
            fix_res["name"] = tickers.TICKERS[code]
            fix_res["bear_days"] = fix_bear
            dynamic_results.append(dyn_res)
            fixed_results.append(fix_res)

        fixed_map = {r["code"]: r for r in fixed_results}
        for dyn in dynamic_results:
            fix = fixed_map.get(dyn["code"])
            if not fix:
                continue
            results.append(
                {
                    "code": dyn["code"],
                    "name": dyn["name"],
                    "dynamic_return": dyn["total_return"],
                    "fixed_return": fix["total_return"],
                    "diff_return": dyn["total_return"] - fix["total_return"],
                }
            )
    else:
        results = run_with_overrides(override_thresholds)

    print("\n" + "=" * 80)
    if diff_mode:
        print(f"📅 动态/固定收益对比 ({start_date_str} - Now)")
        print("策略: XGBoost 信号 + 熊市过滤 (Hybrid)")
    else:
        print(f"📅 近期窗口回测报告 ({start_date_str} - Now)")
        print("策略: XGBoost 信号 + 熊市过滤 (Hybrid)")
    print("=" * 80)

    if diff_mode:
        print(f"{'排名':<6} {'代码':<10} {'名称':<12} {'动态':<10} {'固定':<10} {'差异':<10}")
        print("-" * 80)
        results_sorted = sorted(results, key=lambda x: x["diff_return"], reverse=True)
        for idx, item in enumerate(results_sorted, start=1):
            dyn_str = f"{item['dynamic_return'] * 100:.2f}%"
            fix_str = f"{item['fixed_return'] * 100:.2f}%"
            diff_str = f"{item['diff_return'] * 100:+.2f}%"
            print(f"{idx:<6} {item['code']:<10} {item['name']:<12} {dyn_str:<10} {fix_str:<10} {diff_str:<10}")
        print("=" * 80)
        return

    results_sorted = sorted(results, key=lambda x: x["total_return"], reverse=True)
    if select_mode:
        print(f"{'代码':<10} {'名称':<12} {'模式':<8} {'收益':<10} {'胜率':<10} {'交易':<8} {'最大回撤':<8} {'波动':<8} {'夏普':<8} {'熊市天数'}")
    else:
        print(f"{'代码':<10} {'名称':<12} {'收益':<10} {'胜率':<10} {'交易':<8} {'最大回撤':<8} {'波动':<8} {'夏普':<8} {'熊市天数'}")
    print("-" * 80)

    for res in results_sorted:
        win_rate_str = f"{res['win_rate'] * 100:.1f}%"
        max_dd_str = f"{res['max_drawdown'] * 100:.2f}%"
        vol_str = f"{res['volatility'] * 100:.2f}%"
        sharpe_str = f"{res['sharpe']:.2f}"
        if select_mode:
            print(f"{res['code']:<10} {res['name']:<12} {res.get('mode', ''):<8} {res['total_return'] * 100:6.2f}%    {win_rate_str:<10} {res['num_trades']:<8} {max_dd_str:<8} {vol_str:<8} {sharpe_str:<8} {res['bear_days']}")
        else:
            print(f"{res['code']:<10} {res['name']:<12} {res['total_return'] * 100:6.2f}%    {win_rate_str:<10} {res['num_trades']:<8} {max_dd_str:<8} {vol_str:<8} {sharpe_str:<8} {res['bear_days']}")

    print("=" * 80)
    summary = summarize_results(results)
    print(f"整体平均收益: {summary['avg_return'] * 100:.2f}%")
    print(f"整体胜率:     {summary['overall_win_rate'] * 100:.2f}%")
    print(f"总交易次数:   {summary['total_trades']}")
    print("=" * 80)

    def summarize_group(title: str, codes: list[str]):
        group = [r for r in results if r["code"] in codes]
        if not group:
            return
        group_trades = sum(r["num_trades"] for r in group)
        group_wins = sum(int(round(r["win_rate"] * r["num_trades"])) for r in group)
        avg_return = np.mean([r["total_return"] for r in group])
        avg_max_dd = np.mean([r["max_drawdown"] for r in group])
        avg_vol = np.mean([r["volatility"] for r in group])
        avg_sharpe = np.mean([r["sharpe"] for r in group])
        win_rate = group_wins / group_trades if group_trades > 0 else 0
        print(f"{title}: 数量={len(group)} 平均收益={avg_return * 100:.2f}% 平均胜率={win_rate * 100:.2f}% 平均回撤={avg_max_dd * 100:.2f}% 平均波动={avg_vol * 100:.2f}% 平均夏普={avg_sharpe:.2f} 交易={group_trades}")

    summarize_group("宽基", tickers.WIDE_INDEX_TICKERS)
    summarize_group("行业", tickers.SECTOR_TICKERS)
    if select_mode:
        mode_counts = {"dynamic": 0, "fixed": 0}
        for res in results:
            mode = res.get("mode")
            if mode in mode_counts:
                mode_counts[mode] += 1
        print(f"已选模式: 动态={mode_counts['dynamic']} 固定={mode_counts['fixed']}")


if __name__ == "__main__":
    main()
