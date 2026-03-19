from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.scoring_model import BaseModel


class LogisticModel(BaseModel):
    def __init__(self, model_path: str = "data/logistic_model.pkl"):
        self.model_path = model_path
        self.model: dict[str, np.ndarray | float] | None = None
        self.is_trained = False
        self.feature_cols = [
            "bias_5",
            "bias_20",
            "bias_60",
            "rsi_6",
            "rsi_14",
            "atr_pct",
            "vol_ratio",
            "vol_ratio_20",
            "macd_norm",
            "macdsignal_norm",
            "macdhist_norm",
            "bb_pos",
            "rs_20d",
            "rel_vol",
            "ret_5",
            "ret_10",
            "ret_20",
            "trend_gap",
            "ma20_slope_5",
            "ma60_slope_10",
            "breakout_20",
            "drawdown_20",
            "drawdown_60",
            "rebound_20",
            "rsi_spread",
            "close_to_ma20",
            "close_to_ma60",
            "intraday_range",
            "market_trend_strength",
            "market_above_ma20",
            "market_above_ma60",
            "market_return_5d",
            "market_return_20d",
            "market_volatility_20d",
            "market_drawdown_60d",
            "market_breadth_proxy",
            "market_volume_ratio",
        ]

    def train(self, df: pd.DataFrame):
        required_cols = self.feature_cols + ["target"]
        train_df = df.dropna(subset=required_cols).copy()
        if train_df.empty:
            print("Logistic training data is empty.")
            return

        X = train_df[self.feature_cols].to_numpy(dtype=float)
        y = train_df["target"].to_numpy(dtype=float)
        sample_weight = train_df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train_df.columns else np.ones(len(train_df), dtype=float)

        feature_mean = X.mean(axis=0)
        feature_std = X.std(axis=0)
        feature_std = np.where(feature_std <= 1e-8, 1.0, feature_std)
        X_scaled = (X - feature_mean) / feature_std

        pos_mask = y >= 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            coefficients = np.zeros(X_scaled.shape[1], dtype=float)
            intercept = float(np.log((y.mean() + 1e-6) / (1.0 - y.mean() + 1e-6)))
        else:
            pos_mean = np.average(X_scaled[pos_mask], axis=0, weights=sample_weight[pos_mask])
            neg_mean = np.average(X_scaled[neg_mask], axis=0, weights=sample_weight[neg_mask])
            coefficients = pos_mean - neg_mean
            base_rate = float(np.clip(np.average(y, weights=sample_weight), 1e-6, 1.0 - 1e-6))
            intercept = float(np.log(base_rate / (1.0 - base_rate)))

        self.model = {
            "coefficients": coefficients.astype(float),
            "intercept": float(intercept),
            "feature_mean": feature_mean.astype(float),
            "feature_std": feature_std.astype(float),
        }
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        scores = self.predict_batch(df.iloc[[-1]])
        return round(float(scores[0]), 4) if len(scores) else 0.0

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained and not self.load_model():
            return np.zeros(len(df), dtype=float)
        if self.model is None:
            return np.zeros(len(df), dtype=float)

        X = df[self.feature_cols].to_numpy(dtype=float)
        feature_mean = np.asarray(self.model["feature_mean"], dtype=float)
        feature_std = np.asarray(self.model["feature_std"], dtype=float)
        coefficients = np.asarray(self.model["coefficients"], dtype=float)
        intercept = float(self.model["intercept"])
        X_scaled = (X - feature_mean) / feature_std
        logits = X_scaled @ coefficients + intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def save_model(self):
        if self.model is None:
            raise ValueError("Model is not trained.")
        path = Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)

    def load_model(self) -> bool:
        path = Path(self.model_path)
        if not path.exists():
            return False
        try:
            with path.open("rb") as handle:
                self.model = pickle.load(handle)
            self.is_trained = True
            return True
        except Exception:
            return False

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        coefficients = np.abs(np.asarray(self.model["coefficients"], dtype=float))
        return pd.DataFrame({"feature": self.feature_cols, "importance": coefficients}).sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)
