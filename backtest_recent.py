import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

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
from src.backtest.strategy_config import StrategyConfig
from src.data_loader.data_manager import DataManager
from src.data_loader.tushare_loader import TushareLoader
from src.features.technical import FeatureEngineer
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import StrategyFilter


def _print_diff_results(results: list[dict], start_date_str: str):
    print("\n" + "=" * 80)
    print(f"Dynamic vs Fixed Threshold ({start_date_str} - Now)")
    print("Strategy: XGBoost signal + regime filter")
    print("=" * 80)
    print(f"{'Rank':<6} {'Code':<10} {'Name':<12} {'Dynamic':<10} {'Fixed':<10} {'Diff':<10}")
    print("-" * 80)
    results_sorted = sorted(results, key=lambda x: x["diff_return"], reverse=True)
    for idx, item in enumerate(results_sorted, start=1):
        dyn_str = f"{item['dynamic_return'] * 100:.2f}%"
        fix_str = f"{item['fixed_return'] * 100:.2f}%"
        diff_str = f"{item['diff_return'] * 100:+.2f}%"
        print(f"{idx:<6} {item['code']:<10} {item['name']:<12} {dyn_str:<10} {fix_str:<10} {diff_str:<10}")
    print("=" * 80)


def _print_results(results: list[dict], start_date_str: str, select_mode: bool):
    print("\n" + "=" * 80)
    print(f"Recent Window Backtest ({start_date_str} - Now)")
    print("Strategy: XGBoost signal + regime filter")
    print("=" * 80)
    if select_mode:
        print(f"{'Code':<10} {'Name':<12} {'Mode':<8} {'Return':<10} {'WinRate':<10} {'Trades':<8} {'MaxDD':<8} {'Vol':<8} {'Sharpe':<8} {'BearDays'}")
    else:
        print(f"{'Code':<10} {'Name':<12} {'Return':<10} {'WinRate':<10} {'Trades':<8} {'MaxDD':<8} {'Vol':<8} {'Sharpe':<8} {'BearDays'}")
    print("-" * 80)

    results_sorted = sorted(results, key=lambda x: x["total_return"], reverse=True)
    for res in results_sorted:
        win_rate_str = f"{res['win_rate'] * 100:.1f}%"
        max_dd_str = f"{res['max_drawdown'] * 100:.2f}%"
        vol_str = f"{res['volatility'] * 100:.2f}%"
        sharpe_str = f"{res['sharpe']:.2f}"
        if select_mode:
            print(
                f"{res['code']:<10} {res['name']:<12} {res.get('mode', ''):<8} "
                f"{res['total_return'] * 100:6.2f}%    {win_rate_str:<10} {res['num_trades']:<8} "
                f"{max_dd_str:<8} {vol_str:<8} {sharpe_str:<8} {res['bear_days']}"
            )
        else:
            print(
                f"{res['code']:<10} {res['name']:<12} "
                f"{res['total_return'] * 100:6.2f}%    {win_rate_str:<10} {res['num_trades']:<8} "
                f"{max_dd_str:<8} {vol_str:<8} {sharpe_str:<8} {res['bear_days']}"
            )

    print("=" * 80)
    summary = summarize_results(results)
    print(f"Average Return: {summary['avg_return'] * 100:.2f}%")
    print(f"Overall Win Rate: {summary['overall_win_rate'] * 100:.2f}%")
    print(f"Average Drawdown: {summary['avg_max_drawdown'] * 100:.2f}%")
    print(f"Trades: {summary['total_trades']}")
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
        win_rate = group_wins / group_trades if group_trades > 0 else 0.0
        print(
            f"{title}: count={len(group)} avg_return={avg_return * 100:.2f}% "
            f"win_rate={win_rate * 100:.2f}% avg_dd={avg_max_dd * 100:.2f}% "
            f"avg_vol={avg_vol * 100:.2f}% avg_sharpe={avg_sharpe:.2f} trades={group_trades}"
        )

    summarize_group("Core", tickers.CORE_TRADE_TICKERS)
    summarize_group("Satellite", tickers.SATELLITE_TRADE_TICKERS)
    summarize_group("Observe", tickers.OBSERVE_TICKERS)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y", "on")


def _parse_codes_env(name: str) -> set[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _render_ascii_trade_chart(
    code: str,
    name: str,
    test_df: pd.DataFrame,
    trades: list[dict],
    width: int = 100,
    height: int = 16,
):
    if test_df.empty:
        return

    closes = test_df["close"].astype(float).to_numpy()
    dates = test_df["trade_date"].astype(str).to_numpy()
    n = len(closes)
    if n < 2:
        return

    width = max(30, min(width, n))
    sample_idx = np.linspace(0, n - 1, width).round().astype(int)
    sampled_close = closes[sample_idx]

    p_min = float(np.min(sampled_close))
    p_max = float(np.max(sampled_close))
    if np.isclose(p_max, p_min):
        p_max = p_min + 1e-6

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def y_of(price: float) -> int:
        ratio = (price - p_min) / (p_max - p_min)
        return height - 1 - int(round(ratio * (height - 1)))

    for x in range(width):
        y = y_of(float(sampled_close[x]))
        grid[y][x] = "."

    date_to_col = {}
    for col, src_i in enumerate(sample_idx):
        date_to_col[str(dates[src_i])] = col

    for t in trades:
        action = str(t.get("action", ""))
        marker = "B" if action == "BUY" else "S" if action.startswith("SELL") else None
        if marker is None:
            continue
        d = str(t.get("date", ""))
        if d not in date_to_col:
            continue
        col = date_to_col[d]
        y = y_of(float(sampled_close[col]))
        grid[y][col] = marker

    print("\n" + "-" * 80)
    print(f"{code} {name} Price/Trades Chart")
    print(f"Range: {dates[sample_idx[0]]} -> {dates[sample_idx[-1]]}  Price: {p_min:.3f} ~ {p_max:.3f}")
    for row in grid:
        print("".join(row))
    print("Legend: . close  B buy  S sell")
    print("-" * 80)


def _print_trade_charts(
    results: list[dict],
    data_cache: dict[str, dict],
    chart_codes: set[str] | None = None,
):
    chart_codes = chart_codes or set()
    for res in results:
        code = res.get("code", "")
        if chart_codes and code not in chart_codes:
            continue
        payload = data_cache.get(code)
        if payload is None:
            continue
        _render_ascii_trade_chart(
            code=code,
            name=res.get("name", code),
            test_df=payload["test_df"],
            trades=res.get("trades", []),
        )


def main():
    print("Running backtest for recent window...")
    grid_thresholds_env = os.getenv("GRID_THRESHOLDS", "").strip()
    grid_tickers_env = os.getenv("GRID_TICKERS", "588000.SH,515070.SH")
    select_mode = os.getenv("SELECT_MODE", "").strip().lower() in ("1", "true", "yes", "y")
    diff_mode = os.getenv("DIFF_MODE", "").strip().lower() in ("1", "true", "yes", "y")
    show_chart = _env_flag("SHOW_CHART", default=False)
    chart_codes = _parse_codes_env("CHART_CODES")

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

    lookback_days = int(os.getenv("LOOKBACK_DAYS", "90").strip() or "90")
    train_days = int(os.getenv("TRAIN_DAYS", "180").strip() or "180")
    test_days = int(os.getenv("TEST_DAYS", "90").strip() or "90")
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
    config = StrategyConfig.from_settings()

    model = XGBoostModel(model_path="data/xgb_model.json")
    if not model.load_model():
        print("XGBoost model not found. Please train first.")
        return
    print("XGBoost model loaded.")

    index_df, market_status_map = prepare_index_data(data_manager, feature_eng, strat_filter, index_code="000300.SH")

    include_observe = os.getenv("INCLUDE_OBSERVE", "").strip().lower() in ("1", "true", "yes", "y")
    ticker_list = tickers.get_ticker_list(include_observe=include_observe)
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
            config=config,
        )

    if grid_thresholds:
        print("\n" + "=" * 80)
        print(f"Backtest grid ({start_date_str} - Now)")
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

    if select_mode:
        results = []
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
                use_dynamic = use_dynamic_for_code(code, override_thresholds, config)
                entry_probs, exit_probs, bear_days = build_adjusted_probs(
                    test_df,
                    probs,
                    market_status_map,
                    code,
                    use_dynamic,
                    override_thresholds,
                    config,
                )
                res = backtester.run(test_df, entry_probs, threshold=0.0, code=code, exit_probs=exit_probs, config=config)
                res["code"] = code
                res["name"] = tickers.TICKERS[code]
                res["bear_days"] = bear_days
                res["mode"] = "dynamic" if use_dynamic else "fixed"
                results.append(res)
                continue

            train_dyn_entry, train_dyn_exit, _ = build_adjusted_probs(
                train_df, train_probs, market_status_map, code, True, None, config
            )
            train_fix_entry, train_fix_exit, _ = build_adjusted_probs(
                train_df, train_probs, market_status_map, code, False, override_thresholds, config
            )
            train_dynamic = backtester.run(train_df, train_dyn_entry, threshold=0.0, code=code, exit_probs=train_dyn_exit, config=config)
            train_fixed = backtester.run(train_df, train_fix_entry, threshold=0.0, code=code, exit_probs=train_fix_exit, config=config)
            choose_dynamic = (
                train_dynamic["sharpe"] > train_fixed["sharpe"]
                or (
                    train_dynamic["sharpe"] == train_fixed["sharpe"]
                    and train_dynamic["total_return"] > train_fixed["total_return"]
                )
            )

            eval_entry, eval_exit, bear_days = build_adjusted_probs(
                eval_df,
                eval_probs,
                market_status_map,
                code,
                choose_dynamic,
                override_thresholds if not choose_dynamic else None,
                config,
            )
            res = backtester.run(eval_df, eval_entry, threshold=0.0, code=code, exit_probs=eval_exit, config=config)
            res["code"] = code
            res["name"] = tickers.TICKERS[code]
            res["bear_days"] = bear_days
            res["mode"] = "dynamic" if choose_dynamic else "fixed"
            results.append(res)

        _print_results(results, start_date_str, select_mode=True)
        if show_chart:
            _print_trade_charts(results, data_cache, chart_codes=chart_codes)
        return

    if diff_mode:
        dynamic_results = []
        fixed_results = []
        for code, payload in data_cache.items():
            test_df = payload["test_df"]
            probs = payload["probs"]
            dyn_entry, dyn_exit, dyn_bear = build_adjusted_probs(
                test_df, probs, market_status_map, code, True, None, config
            )
            fix_entry, fix_exit, fix_bear = build_adjusted_probs(
                test_df, probs, market_status_map, code, False, override_thresholds, config
            )
            dyn_res = backtester.run(test_df, dyn_entry, threshold=0.0, code=code, exit_probs=dyn_exit, config=config)
            dyn_res["code"] = code
            dyn_res["name"] = tickers.TICKERS[code]
            dyn_res["bear_days"] = dyn_bear
            fix_res = backtester.run(test_df, fix_entry, threshold=0.0, code=code, exit_probs=fix_exit, config=config)
            fix_res["code"] = code
            fix_res["name"] = tickers.TICKERS[code]
            fix_res["bear_days"] = fix_bear
            dynamic_results.append(dyn_res)
            fixed_results.append(fix_res)

        fixed_map = {r["code"]: r for r in fixed_results}
        results = []
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
        _print_diff_results(results, start_date_str)
        return

    results = run_with_overrides(override_thresholds)
    _print_results(results, start_date_str, select_mode=False)
    if show_chart:
        _print_trade_charts(results, data_cache, chart_codes=chart_codes)


if __name__ == "__main__":
    main()
