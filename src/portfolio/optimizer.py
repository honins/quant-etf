from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PortfolioOptimizerConfig:
    max_positions: int = 5
    max_weight: float = 0.35
    min_weight: float = 0.05
    turnover_penalty: float = 0.15
    correlation_penalty: float = 0.20
    downside_penalty: float = 0.35
    confidence_weight: float = 0.25
    min_score: float = 0.0


class PortfolioOptimizer:
    def __init__(self, config: PortfolioOptimizerConfig | None = None):
        self.config = config or PortfolioOptimizerConfig()

    def optimize(
        self,
        candidates: pd.DataFrame,
        previous_weights: dict[str, float] | None = None,
        return_matrix: pd.DataFrame | None = None,
        trading_costs: dict[str, float] | None = None,
    ) -> dict[str, object]:
        if candidates.empty:
            return {
                "weights": {},
                "top_k": [],
                "expected_turnover": 0.0,
                "score_table": pd.DataFrame(),
                "diagnostics": {},
            }

        previous_weights = previous_weights or {}
        trading_costs = trading_costs or {}
        scored = candidates.copy()
        scored = scored.fillna(
            {
                "expected_return": 0.0,
                "downside_risk": 0.0,
                "confidence": 0.0,
                "avg_correlation": 0.0,
                "liquidity": 0.0,
            }
        )
        scored["prev_weight"] = scored["code"].map(previous_weights).fillna(0.0)
        scored["trading_cost"] = scored["code"].map(trading_costs).fillna(0.0)
        scored["avg_correlation"] = self._inject_correlation(scored, return_matrix)

        liquidity_rank = self._rank_series(scored.get("liquidity", pd.Series(0.0, index=scored.index)))
        confidence = scored.get("confidence", pd.Series(0.0, index=scored.index))
        correlation = scored.get("avg_correlation", pd.Series(0.0, index=scored.index))
        downside = scored.get("downside_risk", pd.Series(0.0, index=scored.index))

        scored["portfolio_score"] = (
            scored["expected_return"]
            + self.config.confidence_weight * confidence
            + 0.10 * liquidity_rank
            - self.config.downside_penalty * downside
            - self.config.correlation_penalty * correlation
            - 0.50 * scored["trading_cost"]
            - self.config.turnover_penalty * scored["prev_weight"].sub(scored["prev_weight"].mean()).abs()
        )
        scored = scored[scored["portfolio_score"] >= self.config.min_score].copy()
        scored = scored.sort_values("portfolio_score", ascending=False).head(self.config.max_positions).reset_index(drop=True)

        if scored.empty:
            return {
                "weights": {},
                "top_k": [],
                "expected_turnover": 0.0,
                "score_table": scored,
                "diagnostics": {
                    "selected_count": 0,
                    "rejected_count": int(len(candidates)),
                    "weight_sum": 0.0,
                    "avg_correlation": 0.0,
                    "avg_trading_cost": 0.0,
                },
            }

        raw_scores = scored["portfolio_score"].clip(lower=0.0)
        if float(raw_scores.sum()) <= 0:
            raw_scores = pd.Series(np.ones(len(scored)), index=scored.index, dtype=float)

        weights = raw_scores / raw_scores.sum()
        weights = weights.clip(lower=self.config.min_weight, upper=self.config.max_weight)
        weights = self._normalize_with_caps(weights)

        weight_map = {row["code"]: float(weights.iloc[idx]) for idx, row in scored.iterrows()}
        expected_turnover = self._estimate_turnover(weight_map, previous_weights)
        top_k = list(scored["code"].head(self.config.max_positions))

        scored["suggested_weight"] = scored["code"].map(weight_map)
        return {
            "weights": weight_map,
            "top_k": top_k,
            "expected_turnover": expected_turnover,
            "score_table": scored,
            "diagnostics": {
                "selected_count": int(len(scored)),
                "rejected_count": int(max(len(candidates) - len(scored), 0)),
                "weight_sum": float(sum(weight_map.values())),
                "max_weight": float(max(weight_map.values())) if weight_map else 0.0,
                "avg_correlation": float(scored["avg_correlation"].mean()) if "avg_correlation" in scored else 0.0,
                "avg_trading_cost": float(scored["trading_cost"].mean()) if "trading_cost" in scored else 0.0,
            },
        }

    @staticmethod
    def _rank_series(series: pd.Series) -> pd.Series:
        if series.empty:
            return series
        return series.rank(pct=True).fillna(0.0)

    @staticmethod
    def _estimate_turnover(weights: dict[str, float], previous_weights: dict[str, float]) -> float:
        codes = set(weights) | set(previous_weights)
        return float(sum(abs(weights.get(code, 0.0) - previous_weights.get(code, 0.0)) for code in codes))

    def _normalize_with_caps(self, weights: pd.Series) -> pd.Series:
        normalized = weights.copy()
        normalized = normalized / normalized.sum()

        for _ in range(5):
            clipped = normalized.clip(lower=self.config.min_weight, upper=self.config.max_weight)
            if abs(float(clipped.sum()) - 1.0) < 1e-9:
                return clipped
            adjustable = clipped[(clipped > self.config.min_weight) & (clipped < self.config.max_weight)]
            remainder = 1.0 - float(clipped.sum())
            if adjustable.empty:
                return clipped / clipped.sum()
            clipped.loc[adjustable.index] = clipped.loc[adjustable.index] + remainder * (
                clipped.loc[adjustable.index] / clipped.loc[adjustable.index].sum()
            )
            normalized = clipped

        return normalized / normalized.sum()

    @staticmethod
    def _inject_correlation(scored: pd.DataFrame, return_matrix: pd.DataFrame | None) -> pd.Series:
        if return_matrix is None or return_matrix.empty or "code" not in scored.columns:
            return pd.Series(np.zeros(len(scored)), index=scored.index, dtype=float)

        available_codes = [code for code in scored["code"].tolist() if code in return_matrix.columns]
        if len(available_codes) < 2:
            return pd.Series(np.zeros(len(scored)), index=scored.index, dtype=float)

        corr = return_matrix[available_codes].corr().abs()
        corr_values = {}
        for code in available_codes:
            peers = corr.loc[code].drop(labels=[code], errors="ignore")
            corr_values[code] = float(peers.mean()) if not peers.empty else 0.0
        return scored["code"].map(corr_values).fillna(0.0)
