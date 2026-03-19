from __future__ import annotations

from typing import Any

import pandas as pd


def replay_portfolio_allocations(
    portfolio_plan: dict[str, Any],
    datasets: dict[str, pd.DataFrame],
    lookback_days: int,
) -> dict[str, Any]:
    weights = portfolio_plan.get("weights", {}) or {}
    if not weights:
        return {
            "series": [],
            "summary": {
                "total_return": 0.0,
                "volatility": 0.0,
                "num_days": 0,
            },
        }

    return_frame = pd.DataFrame()
    for code, weight in weights.items():
        frame = datasets.get(code)
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        recent = frame.tail(lookback_days).copy()
        if len(recent) < 2:
            continue
        returns = recent["close"].pct_change().fillna(0.0).reset_index(drop=True)
        return_frame[code] = returns * float(weight)

    if return_frame.empty:
        return {
            "series": [],
            "summary": {
                "total_return": 0.0,
                "volatility": 0.0,
                "num_days": 0,
            },
        }

    portfolio_returns = return_frame.sum(axis=1)
    equity = (1.0 + portfolio_returns).cumprod()
    series = [
        {
            "day": int(index),
            "equity": float(value),
            "daily_return": float(portfolio_returns.iloc[index]),
        }
        for index, value in enumerate(equity)
    ]
    summary = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "volatility": float(portfolio_returns.std(ddof=0) * (252 ** 0.5)),
        "num_days": int(len(portfolio_returns)),
    }
    return {"series": series, "summary": summary}


def build_portfolio_backtest_report(
    replay: dict[str, Any],
) -> dict[str, Any]:
    series = replay.get("series", []) or []
    if not series:
        return {
            "max_drawdown": 0.0,
            "ending_equity": 1.0,
            "positive_days": 0,
            "negative_days": 0,
        }

    equity = pd.Series([float(point.get("equity", 1.0)) for point in series], dtype=float)
    returns = pd.Series([float(point.get("daily_return", 0.0)) for point in series], dtype=float)
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return {
        "max_drawdown": float(drawdown.min()),
        "ending_equity": float(equity.iloc[-1]),
        "positive_days": int((returns > 0).sum()),
        "negative_days": int((returns < 0).sum()),
    }


def build_weighted_portfolio_plan(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in results
        if float(row.get("total_return", 0.0)) > -0.01 or int(row.get("num_trades", 0)) > 0
    ]
    if not eligible:
        return {"weights": {}}

    scored = []
    for row in eligible:
        total_return = float(row.get("total_return", 0.0))
        max_drawdown = float(row.get("max_drawdown", 0.0))
        sharpe = float(row.get("sharpe", 0.0))
        trade_count = float(row.get("num_trades", 0))
        score = max(total_return, 0.0) + 0.20 * max(sharpe, 0.0) - 0.35 * max(max_drawdown, 0.0) + 0.01 * trade_count
        if score > 0:
            scored.append((str(row.get("code", "")), score))

    if not scored:
        return {"weights": {}}

    score_sum = sum(score for _, score in scored)
    weights = {code: score / score_sum for code, score in scored}
    return {"weights": weights}


def build_quality_portfolio_plan(
    results: list[dict[str, Any]],
    min_win_rate: float,
    min_return: float,
    min_sharpe: float,
    max_weight: float,
    top_n: int | None = None,
) -> dict[str, Any]:
    eligible = []
    for row in results:
        total_return = float(row.get("total_return", 0.0))
        win_rate = float(row.get("win_rate", 0.0))
        sharpe = float(row.get("sharpe", 0.0))
        if total_return < min_return:
            continue
        if win_rate < min_win_rate:
            continue
        if sharpe < min_sharpe:
            continue
        eligible.append(dict(row))

    if not eligible:
        return {"weights": {}}

    scores = []
    for row in eligible:
        total_return = float(row.get("total_return", 0.0))
        win_rate = float(row.get("win_rate", 0.0))
        sharpe = float(row.get("sharpe", 0.0))
        max_drawdown = float(row.get("max_drawdown", 0.0))
        score = 1.15 * max(total_return, 0.0) + 0.45 * win_rate + 0.25 * max(sharpe, 0.0) - 0.25 * max(max_drawdown, 0.0)
        if score > 0:
            scores.append((str(row.get("code", "")), score))

    if not scores:
        return {"weights": {}}

    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    if top_n is not None and top_n > 0:
        scores = scores[:top_n]

    score_sum = sum(score for _, score in scores)
    raw_weights = {code: score / score_sum for code, score in scores}
    clipped = {code: min(weight, max_weight) for code, weight in raw_weights.items()}
    clipped_sum = sum(clipped.values())
    if clipped_sum <= 0:
        return {"weights": {}}
    normalized = {code: weight / clipped_sum for code, weight in clipped.items()}
    return {"weights": normalized}


def build_defensive_portfolio_plan(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    for row in results:
        total_return = float(row.get("total_return", 0.0))
        win_rate = float(row.get("win_rate", 0.0))
        sharpe = float(row.get("sharpe", 0.0))
        max_drawdown = float(row.get("max_drawdown", 0.0))
        volatility = float(row.get("volatility", 0.0))
        if total_return < 0.0:
            continue
        if win_rate < 0.45:
            continue
        if sharpe < 0.5:
            continue
        if max_drawdown > 0.10:
            continue
        if volatility > 0.30:
            continue
        eligible.append(dict(row))

    if not eligible:
        return {"weights": {}}

    scored = []
    for row in eligible:
        total_return = float(row.get("total_return", 0.0))
        win_rate = float(row.get("win_rate", 0.0))
        sharpe = float(row.get("sharpe", 0.0))
        max_drawdown = float(row.get("max_drawdown", 0.0))
        volatility = float(row.get("volatility", 0.0))
        score = 1.10 * total_return + 0.50 * win_rate + 0.20 * sharpe - 0.35 * max_drawdown - 0.20 * volatility
        if score > 0:
            scored.append((str(row.get("code", "")), score))

    if not scored:
        return {"weights": {}}

    scored = sorted(scored, key=lambda item: item[1], reverse=True)[:3]
    score_sum = sum(score for _, score in scored)
    raw_weights = {code: score / score_sum for code, score in scored}
    capped = {code: min(weight, 0.45) for code, weight in raw_weights.items()}
    capped_sum = sum(capped.values())
    return {"weights": {code: weight / capped_sum for code, weight in capped.items()}}


def build_portfolio_variants(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggressive = build_weighted_portfolio_plan(results)

    balanced_rows = []
    for row in results:
        total_return = float(row.get("total_return", 0.0))
        sharpe = float(row.get("sharpe", 0.0))
        max_drawdown = float(row.get("max_drawdown", 0.0))
        if total_return <= -0.01 and sharpe <= 0:
            continue
        adjusted = dict(row)
        adjusted["total_return"] = max(total_return, 0.0)
        adjusted["sharpe"] = max(sharpe, 0.0)
        adjusted["max_drawdown"] = max_drawdown * 1.2
        balanced_rows.append(adjusted)

    balanced = build_weighted_portfolio_plan(balanced_rows)
    quality = build_quality_portfolio_plan(
        results,
        min_win_rate=0.45,
        min_return=0.0,
        min_sharpe=0.0,
        max_weight=0.40,
        top_n=4,
    )
    elite = build_quality_portfolio_plan(
        results,
        min_win_rate=0.50,
        min_return=0.005,
        min_sharpe=0.8,
        max_weight=0.38,
        top_n=3,
    )
    return {
        "aggressive": aggressive,
        "balanced": balanced,
        "quality": quality,
        "elite": elite,
        "defensive": build_defensive_portfolio_plan(results),
    }
