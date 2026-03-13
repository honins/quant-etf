from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from config import tickers
from config.settings import settings
from src.backtest.backtester import Backtester
from src.backtest.hybrid_runner import (
    prepare_index_data,
    run_backtest_for_cache,
    summarize_results,
)
from src.backtest.strategy_config import StrategyConfig
from src.data_loader.data_manager import DataManager
from src.data_loader.tushare_loader import TushareLoader
from src.features.technical import FeatureEngineer
from src.models.scoring_model import RuleBasedModel
from src.models.xgb_model import XGBoostModel
from src.strategy.logic import RiskManager, StrategyFilter
from src.utils.explainer import TechnicalExplainer
from src.utils.feishu_bot import FeishuBot
from src.utils.holdings_manager import HoldingsManager


def _float_or_none(value: object, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _pct(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value * 100.0, digits)


def _report_path_for_today() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return str(settings.REPORTS_DIR / f"daily_report_{today}.md")


def _report_url_for_today() -> str:
    return Path(_report_path_for_today()).name


def _recent_reports(limit: int = 6) -> list[dict]:
    reports = sorted(
        settings.REPORTS_DIR.glob("daily_report_*.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    items: list[dict] = []
    for report_path in reports[:limit]:
        stat = report_path.stat()
        items.append(
            {
                "name": report_path.name,
                "url": report_path.name,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(stat.st_size / 1024.0, 1),
            }
        )
    return items


def _calc_position_size(
    risk_data: dict,
    total_capital: float = 100_000,
    risk_pct: float = 0.02,
) -> dict:
    if not risk_data:
        return {}

    risk_per_share = risk_data.get("risk_per_share", 0)
    current_price = risk_data.get("current_price", 0)
    if risk_per_share <= 0 or current_price <= 0:
        return {}

    max_loss = total_capital * risk_pct
    suggested_shares = int(max_loss / risk_per_share / 100) * 100
    suggested_value = round(suggested_shares * current_price, 2)
    suggested_weight = round(suggested_value / total_capital, 4)
    return {
        "suggested_shares": suggested_shares,
        "suggested_value": suggested_value,
        "suggested_weight_pct": round(suggested_weight * 100, 2),
    }


MARKET_STATUS_LABELS = {
    "Bull Market": "牛市",
    "Bear Market": "熊市",
    "Volatile Market": "震荡市",
    "Unknown Market": "未知市场",
}

MODEL_NAME_LABELS = {
    "XGBoost": "XGBoost 模型",
    "Rules": "规则模型",
}

MODE_LABELS = {
    "dynamic": "动态阈值",
    "fixed": "固定阈值",
}

SIGNAL_BUCKET_LABELS = {
    "buy": "买入",
    "watch": "观察",
    "observe": "观察池",
    "idle": "待机",
}


def _market_status_label(status: str) -> str:
    return MARKET_STATUS_LABELS.get(status, status)


def _model_name_label(model_name: str) -> str:
    return MODEL_NAME_LABELS.get(model_name, model_name)


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def _signal_bucket_label(bucket: str) -> str:
    return SIGNAL_BUCKET_LABELS.get(bucket, bucket)


def _holding_status_label(status: object) -> str:
    raw_status = str(status or "")
    if "SELL" in raw_status:
        return "卖出（止损触发）"
    if "HOLD" in raw_status:
        return "持有"
    return raw_status or "-"


def _holding_action_label(status: object, action: object) -> str:
    raw_status = str(status or "")
    if "SELL" in raw_status:
        return "卖出止盈/止损"
    if "HOLD" in raw_status:
        return "继续持有"
    raw_action = str(action or "")
    return raw_action or "-"


def _format_compact_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _load_model():
    model = XGBoostModel(model_path="data/xgb_model.json")
    if model.load_model():
        return model, "XGBoost"
    return RuleBasedModel(), "Rules"


def _use_dynamic_for_live_signal(code: str) -> bool:
    category = tickers.get_ticker_category(code)
    if category == "core":
        return False
    if category == "satellite":
        return True
    return settings.USE_DYNAMIC_THRESHOLD


def _resolve_live_threshold(
    code: str,
    market_status: str,
    dynamic_threshold: float | None,
) -> float:
    if market_status == "Bull Market":
        threshold = dynamic_threshold
        if threshold is None:
            threshold = settings.TICKER_BULL_THRESHOLDS.get(code)
        if threshold is None:
            threshold = (
                settings.BULL_AGGRESSIVE_THRESHOLD
                if code in settings.AGGRESSIVE_TICKERS
                else settings.BULL_BASE_THRESHOLD
            )
    elif market_status == "Bear Market":
        threshold = settings.BEAR_THRESHOLD
    else:
        threshold = dynamic_threshold if dynamic_threshold is not None else settings.VOLATILE_THRESHOLD
    return round(float(threshold), 4)


def _serialize_history(scored_df, history_days: int) -> list[dict]:
    history = []
    for _, row in scored_df.tail(history_days).iterrows():
        history.append(
            {
                "date": str(row["trade_date"]),
                "close": _float_or_none(row["close"], 4),
                "ma20": _float_or_none(row.get("ma20"), 4),
                "ma60": _float_or_none(row.get("ma60"), 4),
                "score": _float_or_none(row.get("_score"), 4),
            }
        )
    return history


def _signal_bucket(result: dict) -> str:
    if result["is_buy"]:
        return "buy"
    if result["category"] == "observe":
        return "observe" if result["score"] >= 0.6 else "idle"
    if result["score"] >= 0.6:
        return "watch"
    return "idle"


def build_live_snapshot(
    data_manager: DataManager,
    feature_eng: FeatureEngineer,
    index_df,
    model,
    model_name: str,
    history_days: int = 120,
) -> dict:
    strat_filter = StrategyFilter()
    risk_manager = RiskManager()
    holdings_manager = HoldingsManager()
    market_status = strat_filter._detect_market_regime(index_df) if not index_df.empty else "Unknown Market"

    results: list[dict] = []
    histories: dict[str, list[dict]] = {}
    datasets: dict[str, object] = {}

    for code in tickers.get_ticker_list(include_observe=True):
        raw_df = data_manager.update_and_get_data(code, is_index=False)
        if raw_df.empty:
            continue

        feature_df = feature_eng.calculate_technical_indicators(raw_df.copy())
        feature_df = model.prepare_data(feature_df)
        if not index_df.empty:
            feature_df = feature_eng.add_relative_strength(feature_df, index_df, period=20)
        scored_df = feature_df.dropna().copy()

        if len(scored_df) < 60:
            continue

        score = model.predict(scored_df)
        use_dynamic = _use_dynamic_for_live_signal(code)
        dynamic_threshold = None
        if use_dynamic and callable(getattr(model, "predict_batch", None)):
            lookback = min(settings.DYNAMIC_THRESHOLD_LOOKBACK, len(scored_df))
            recent_scores = model.predict_batch(scored_df.tail(lookback))
            dynamic_threshold = StrategyFilter.dynamic_threshold(recent_scores)

        is_buy, filtered_market_status = strat_filter.filter_signal(
            score,
            index_df,
            code=code,
            dynamic_threshold=dynamic_threshold,
        )
        market_status = filtered_market_status.split(" (", 1)[0]
        risk_data = risk_manager.calculate_stops(feature_df, code=code)
        explanations = TechnicalExplainer.explain(feature_df)
        position_size = _calc_position_size(risk_data)
        category = tickers.get_ticker_category(code)
        decision_note = ""
        if category == "observe":
            is_buy = False
            decision_note = "仅观察"

        if callable(getattr(model, "predict_batch", None)):
            scored_df = scored_df.copy()
            scored_df["_score"] = model.predict_batch(scored_df)
            histories[code] = _serialize_history(scored_df, history_days)
        datasets[code] = scored_df

        entry_threshold = _resolve_live_threshold(code, market_status, dynamic_threshold)
        result = {
            "code": code,
            "name": tickers.get_ticker_name(code),
            "category": category,
            "category_label": tickers.get_ticker_category_label(code),
            "score": round(float(score), 4),
            "is_buy": bool(is_buy),
            "market_status": market_status,
            "market_status_label": _market_status_label(market_status),
            "mode": "dynamic" if use_dynamic else "fixed",
            "mode_label": _mode_label("dynamic" if use_dynamic else "fixed"),
            "entry_threshold": entry_threshold,
            "threshold_gap": round(score - entry_threshold, 4),
            "current_price": _float_or_none(feature_df.iloc[-1]["close"], 4),
            "risk": {
                "current_price": _float_or_none(risk_data.get("current_price"), 4) if risk_data else None,
                "atr": _float_or_none(risk_data.get("atr"), 4) if risk_data else None,
                "initial_stop_loss": _float_or_none(risk_data.get("initial_stop_loss"), 4) if risk_data else None,
                "trailing_stop_loss": _float_or_none(risk_data.get("trailing_stop_loss"), 4) if risk_data else None,
                "risk_per_share": _float_or_none(risk_data.get("risk_per_share"), 4) if risk_data else None,
            },
            "reasons": explanations or [],
            "decision_note": decision_note,
            "position_size": position_size,
            "signal_bucket": "",
            "return_90d": None,
            "win_rate_90d": None,
            "trades_90d": None,
            "return_180d": None,
            "win_rate_180d": None,
            "trades_180d": None,
            "signal_bucket_label": "",
        }
        result["signal_bucket"] = _signal_bucket(result)
        result["signal_bucket_label"] = _signal_bucket_label(result["signal_bucket"])
        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)

    holdings = holdings_manager.check_holdings(data_manager, feature_eng)
    serialized_holdings = []
    for item in holdings:
        serialized_holdings.append(
            {
                "code": item["code"],
                "name": item["name"],
                "buy_price": _float_or_none(item["buy_price"], 4),
                "current_price": _float_or_none(item["current_price"], 4),
                "trailing_stop": _float_or_none(item["trailing_stop"], 4),
                "pnl_pct": _float_or_none(item["pnl_pct"], 2),
                "status": item["status"],
                "action": item["action"],
                "status_label": _holding_status_label(item["status"]),
                "action_label": _holding_action_label(item["status"], item["action"]),
                "days_held": int(item["days_held"]),
            }
        )

    buy = [item for item in results if item["signal_bucket"] == "buy"]
    watch = [item for item in results if item["signal_bucket"] == "watch"]
    observe = [item for item in results if item["signal_bucket"] == "observe"]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_path": _report_path_for_today(),
        "report_url": _report_url_for_today(),
        "market_status": market_status,
        "market_status_label": _market_status_label(market_status),
        "model_name": model_name,
        "model_name_label": _model_name_label(model_name),
        "results": results,
        "buy": buy,
        "watch": watch,
        "observe": observe,
        "holdings": serialized_holdings,
        "histories": histories,
        "datasets": datasets,
    }


def _serialize_backtest_results(results: list[dict]) -> list[dict]:
    serialized = []
    for item in sorted(results, key=lambda row: row["total_return"], reverse=True):
        serialized.append(
            {
                "code": item["code"],
                "name": item["name"],
                "category": tickers.get_ticker_category(item["code"]),
                "category_label": tickers.get_ticker_category_label(item["code"]),
                "mode": item.get("mode", ""),
                "mode_label": _mode_label(item.get("mode", "")),
                "total_return_pct": _pct(item["total_return"]),
                "win_rate_pct": _pct(item["win_rate"]),
                "num_trades": int(item["num_trades"]),
                "max_drawdown_pct": _pct(item["max_drawdown"]),
                "volatility_pct": _pct(item["volatility"]),
                "sharpe": _float_or_none(item["sharpe"], 2),
                "bear_days": int(item.get("bear_days", 0)),
            }
        )
    return serialized


def _serialize_summary(summary: dict[str, float], ticker_count: int) -> dict[str, float]:
    return {
        "ticker_count": ticker_count,
        "avg_return_pct": _pct(summary["avg_return"]),
        "avg_max_drawdown_pct": _pct(summary["avg_max_drawdown"]),
        "avg_volatility_pct": _pct(summary["avg_volatility"]),
        "positive_ratio_pct": _pct(summary["positive_ratio"]),
        "overall_win_rate_pct": _pct(summary["overall_win_rate"]),
        "total_trades": int(summary["total_trades"]),
    }


def _serialize_backtest_charts(
    results: list[dict],
    data_cache: dict[str, dict],
    lookback_days: int,
) -> dict[str, dict]:
    charts: dict[str, dict] = {}
    for item in results:
        code = item["code"]
        payload = data_cache.get(code)
        if payload is None:
            continue
        test_df = payload.get("test_df")
        if test_df is None or test_df.empty:
            continue

        buy_map: dict[str, float] = {}
        sell_map: dict[str, float] = {}
        trade_points: list[dict] = []

        for trade in item.get("trades", []):
            action = str(trade.get("action", ""))
            date = str(trade.get("date", ""))
            if not date:
                continue
            price = _float_or_none(trade.get("price"), 4)
            trade_type = ""
            if action == "BUY":
                trade_type = "buy"
                if price is not None:
                    buy_map[date] = price
            elif action.startswith("SELL"):
                trade_type = "sell"
                if price is not None:
                    sell_map[date] = price
            if not trade_type:
                continue
            trade_points.append(
                {
                    "date": date,
                    "price": price,
                    "type": trade_type,
                    "action": action,
                    "pnl": _float_or_none(trade.get("pnl"), 2),
                }
            )

        series: list[dict] = []
        for _, row in test_df.iterrows():
            date = str(row["trade_date"])
            series.append(
                {
                    "date": date,
                    "close": _float_or_none(row.get("close"), 4),
                    "buy_price": _float_or_none(buy_map.get(date), 4),
                    "sell_price": _float_or_none(sell_map.get(date), 4),
                }
            )

        charts[code] = {
            "window_days": lookback_days,
            "start_date": str(test_df.iloc[0]["trade_date"]),
            "end_date": str(test_df.iloc[-1]["trade_date"]),
            "series": series,
            "trades": trade_points,
        }

    return charts


def build_backtest_snapshot(
    datasets: dict[str, object],
    market_status_map: dict[str, str],
    lookback_days: int,
) -> dict:
    backtester = Backtester()
    config = StrategyConfig.from_settings()
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=lookback_days)).strftime("%Y%m%d")
    data_cache = {}
    for code in tickers.get_tradable_ticker_list():
        scored_df = datasets.get(code)
        if scored_df is None or scored_df.empty:
            continue
        test_df = scored_df[scored_df["trade_date"].astype(str) >= start_date].copy()
        if len(test_df) < 10:
            continue
        probs = (
            test_df["_score"].to_numpy()
            if "_score" in test_df.columns
            else np.zeros(len(test_df), dtype=float)
        )
        data_cache[code] = {"test_df": test_df, "probs": probs}
    results = run_backtest_for_cache(
        data_cache,
        backtester,
        market_status_map,
        config=config,
    )
    serialized_results = _serialize_backtest_results(results)
    serialized_charts = _serialize_backtest_charts(results, data_cache, lookback_days)
    summary = _serialize_summary(summarize_results(results), len(serialized_results))
    return {
        "window_days": lookback_days,
        "start_date": start_date,
        "end_date": end_date.strftime("%Y%m%d"),
        "start_date_label": _format_compact_date(start_date),
        "end_date_label": _format_compact_date(end_date.strftime("%Y%m%d")),
        "summary": summary,
        "results": serialized_results,
        "charts": serialized_charts,
    }


def build_dashboard_payload(history_days: int = 120) -> dict:
    loader = TushareLoader()
    data_manager = DataManager(loader)
    feature_eng = FeatureEngineer()
    strat_filter = StrategyFilter()
    model, model_name = _load_model()

    index_df, market_status_map = prepare_index_data(
        data_manager,
        feature_eng,
        strat_filter,
        index_code="000300.SH",
    )
    live_snapshot = build_live_snapshot(
        data_manager,
        feature_eng,
        index_df,
        model,
        model_name=model_name,
        history_days=history_days,
    )
    datasets = live_snapshot.pop("datasets")
    backtest_90 = build_backtest_snapshot(datasets, market_status_map, lookback_days=90)
    backtest_180 = build_backtest_snapshot(datasets, market_status_map, lookback_days=180)

    bt90_map = {item["code"]: item for item in backtest_90["results"]}
    bt180_map = {item["code"]: item for item in backtest_180["results"]}
    for item in live_snapshot["results"]:
        if item["code"] in bt90_map:
            item["return_90d"] = bt90_map[item["code"]]["total_return_pct"]
            item["win_rate_90d"] = bt90_map[item["code"]]["win_rate_pct"]
            item["trades_90d"] = bt90_map[item["code"]]["num_trades"]
        if item["code"] in bt180_map:
            item["return_180d"] = bt180_map[item["code"]]["total_return_pct"]
            item["win_rate_180d"] = bt180_map[item["code"]]["win_rate_pct"]
            item["trades_180d"] = bt180_map[item["code"]]["num_trades"]

    top_live = live_snapshot["results"][0] if live_snapshot["results"] else None
    top_90 = backtest_90["results"][0] if backtest_90["results"] else None
    top_180 = backtest_180["results"][0] if backtest_180["results"] else None

    return {
        "generated_at": live_snapshot["generated_at"],
        "market_status": live_snapshot["market_status"],
        "market_status_label": live_snapshot["market_status_label"],
        "model_name": live_snapshot["model_name"],
        "model_name_label": live_snapshot["model_name_label"],
        "report_path": live_snapshot["report_path"],
        "report_url": live_snapshot["report_url"],
        "recent_reports": _recent_reports(),
        "controls": {
            "history_days": history_days,
            "feishu_configured": bool(FeishuBot().webhook),
        },
        "stats": {
            "active_tickers": len(live_snapshot["results"]),
            "tradable_tickers": len(tickers.get_tradable_ticker_list()),
            "observe_tickers": len(tickers.get_observe_ticker_list()),
            "buy_count": len(live_snapshot["buy"]),
            "watch_count": len(live_snapshot["watch"]),
            "holdings_count": len(live_snapshot["holdings"]),
            "top_live_score": top_live["score"] if top_live else None,
            "top_live_name": top_live["name"] if top_live else None,
            "top_90_name": top_90["name"] if top_90 else None,
            "top_90_return_pct": top_90["total_return_pct"] if top_90 else None,
            "top_180_name": top_180["name"] if top_180 else None,
            "top_180_return_pct": top_180["total_return_pct"] if top_180 else None,
        },
        "signals": {
            "buy": live_snapshot["buy"],
            "watch": live_snapshot["watch"],
            "observe": live_snapshot["observe"],
            "all": live_snapshot["results"],
        },
        "holdings": live_snapshot["holdings"],
        "histories": live_snapshot["histories"],
        "backtests": {
            "90d": backtest_90,
            "180d": backtest_180,
        },
    }


def build_dashboard_json(history_days: int = 120) -> str:
    return json.dumps(build_dashboard_payload(history_days=history_days), ensure_ascii=False)
