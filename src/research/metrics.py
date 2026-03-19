from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def annualize_return(total_return: float, periods: int) -> float:
    if periods <= 0:
        return 0.0
    growth = 1.0 + _safe_float(total_return)
    if growth <= 0:
        return -1.0
    return float(growth ** (TRADING_DAYS_PER_YEAR / periods) - 1.0)


def summarize_backtest_result(result: Mapping[str, Any]) -> dict[str, Any]:
    equity_curve = result.get("equity_curve", []) or []
    trades = result.get("trades", []) or []
    periods = max(len(equity_curve) - 1, 0)

    total_return = _safe_float(result.get("total_return"))
    max_drawdown = _safe_float(result.get("max_drawdown"))
    annual_return = _safe_float(result.get("annual_return"), annualize_return(total_return, periods))
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    sell_trades = [trade for trade in trades if str(trade.get("action", "")).startswith("SELL")]
    holding_days = [
        _safe_float(trade.get("holding_days"), np.nan)
        for trade in sell_trades
        if trade.get("holding_days") is not None
    ]
    holding_days = [value for value in holding_days if not np.isnan(value)]

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "volatility": _safe_float(result.get("volatility")),
        "sharpe": _safe_float(result.get("sharpe")),
        "calmar": calmar,
        "turnover": _safe_float(result.get("turnover")),
        "avg_holding_days": float(np.mean(holding_days)) if holding_days else 0.0,
        "win_rate": _safe_float(result.get("win_rate")),
        "num_trades": int(_safe_float(result.get("num_trades"))),
        "final_equity": _safe_float(result.get("final_equity")),
        "daily_points": periods,
        "trade_count": len(sell_trades),
    }


def build_regime_breakdown(results: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(results)
    if frame.empty:
        return {}

    regime_col = None
    for candidate in ("regime", "market_status", "regime_label"):
        if candidate in frame.columns:
            regime_col = candidate
            break

    if regime_col is None:
        return {}

    breakdown: dict[str, dict[str, float]] = {}
    grouped = frame.groupby(regime_col)
    for regime, group in grouped:
        breakdown[str(regime)] = {
            "count": int(len(group)),
            "avg_total_return": float(group["total_return"].mean()) if "total_return" in group else 0.0,
            "avg_max_drawdown": float(group["max_drawdown"].mean()) if "max_drawdown" in group else 0.0,
            "avg_sharpe": float(group["sharpe"].mean()) if "sharpe" in group else 0.0,
        }
    return breakdown


def summarize_experiment_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "result_count": 0,
            "avg_total_return": 0.0,
            "avg_annual_return": 0.0,
            "avg_max_drawdown": 0.0,
            "avg_volatility": 0.0,
            "avg_sharpe": 0.0,
            "avg_calmar": 0.0,
            "avg_turnover": 0.0,
            "avg_holding_days": 0.0,
            "overall_win_rate": 0.0,
            "total_trades": 0,
            "positive_ratio": 0.0,
            "regime_breakdown": {},
        }

    normalized = []
    for result in results:
        metrics = summarize_backtest_result(result)
        merged = dict(result)
        merged.update(metrics)
        normalized.append(merged)

    frame = pd.DataFrame(normalized)
    return {
        "result_count": int(len(frame)),
        "avg_total_return": float(frame["total_return"].mean()),
        "avg_annual_return": float(frame["annual_return"].mean()),
        "avg_max_drawdown": float(frame["max_drawdown"].mean()),
        "avg_volatility": float(frame["volatility"].mean()),
        "avg_sharpe": float(frame["sharpe"].mean()),
        "avg_calmar": float(frame["calmar"].mean()),
        "avg_turnover": float(frame["turnover"].mean()),
        "avg_holding_days": float(frame["avg_holding_days"].mean()),
        "overall_win_rate": float(frame["win_rate"].mean()),
        "total_trades": int(frame["num_trades"].sum()),
        "positive_ratio": float((frame["total_return"] > 0).mean()),
        "regime_breakdown": build_regime_breakdown(normalized),
    }


def build_coverage_report(dataset: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    ticker_rows: list[dict[str, Any]] = []
    feature_coverage: dict[str, list[float]] = {}

    for code, frame in dataset.items():
        if frame.empty:
            ticker_rows.append(
                {
                    "code": code,
                    "rows": 0,
                    "start_date": None,
                    "end_date": None,
                    "missing_ratio": 1.0,
                }
            )
            continue

        missing_ratio = float(frame.isna().mean().mean()) if not frame.empty else 1.0
        ticker_rows.append(
            {
                "code": code,
                "rows": int(len(frame)),
                "start_date": str(frame["trade_date"].iloc[0]) if "trade_date" in frame.columns else None,
                "end_date": str(frame["trade_date"].iloc[-1]) if "trade_date" in frame.columns else None,
                "missing_ratio": missing_ratio,
            }
        )

        for column in frame.columns:
            if column == "trade_date":
                continue
            coverage = float(frame[column].notna().mean())
            feature_coverage.setdefault(column, []).append(coverage)

    feature_summary = {
        column: {
            "coverage_mean": float(np.mean(values)),
            "coverage_min": float(np.min(values)),
            "coverage_max": float(np.max(values)),
        }
        for column, values in feature_coverage.items()
    }
    return {
        "ticker_count": len(dataset),
        "tickers": ticker_rows,
        "feature_coverage": feature_summary,
    }


def build_leakage_report(dataset: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    suspicious_columns = [
        "future_ret_1d",
        "future_ret_3d",
        "future_ret_5d",
        "future_ret_10d",
        "future_ret_20d",
        "future_max_ret_20d",
        "future_min_ret_20d",
        "target",
        "meta_target",
    ]
    rows = []
    for code, frame in dataset.items():
        available = [column for column in suspicious_columns if column in frame.columns]
        if not available:
            rows.append({"code": code, "checked_columns": [], "trailing_nan_ok": True, "notes": "no future columns"})
            continue

        trailing_ok = True
        notes = []
        for column in available:
            tail = frame[column].tail(25)
            if tail.notna().all():
                trailing_ok = False
                notes.append(f"{column}: no tail NaN buffer")
        rows.append(
            {
                "code": code,
                "checked_columns": available,
                "trailing_nan_ok": trailing_ok,
                "notes": "; ".join(notes) if notes else "tail NaN buffer present",
            }
        )
    return {
        "checked_tickers": len(dataset),
        "issues_found": int(sum(0 if row["trailing_nan_ok"] else 1 for row in rows)),
        "rows": rows,
    }
