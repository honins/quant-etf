from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config.settings import settings
from src.models.scoring_model import BaseModel

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

try:
    import joblib
except Exception:
    joblib = None


class XGBoostModel(BaseModel):
    def __init__(self, model_path: str = "data/xgb_model.json"):
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "seed": 42,
        }
        self.num_boost_round = 300
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
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.best_iteration = None
        self.available = xgb is not None

    def _param_candidates(self, scale_pos_weight: float) -> list[dict]:
        candidate_overrides = [
            {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3, "gamma": 0.0, "lambda": 2.0},
            {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 5, "gamma": 0.0, "lambda": 3.0},
            {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.80, "min_child_weight": 4, "gamma": 0.1, "lambda": 3.0},
        ]

        params = []
        for override in candidate_overrides:
            candidate = dict(self.params)
            candidate.update(override)
            candidate["scale_pos_weight"] = scale_pos_weight
            params.append(candidate)
        return params

    def _validation_score(self, y_val: pd.Series, y_pred_prob: np.ndarray, val_df: pd.DataFrame) -> tuple[float, dict[str, float]]:
        horizon = settings.TRAIN_LABEL_HORIZON
        quality_col = f"quality_{horizon}d"
        upside_col = f"future_max_ret_{horizon}d"
        downside_col = f"future_min_ret_{horizon}d"

        selected = y_pred_prob >= max(0.55, float(np.quantile(y_pred_prob, 0.75)))
        if selected.sum() == 0:
            selected = y_pred_prob >= float(np.max(y_pred_prob))

        predicted_positive = (y_pred_prob > 0.5).astype(int)
        true_positive = float(((y_val.to_numpy() == 1) & (predicted_positive == 1)).sum())
        predicted_positive_count = float(predicted_positive.sum())
        precision = true_positive / predicted_positive_count if predicted_positive_count > 0 else 0.0
        auc = 0.5
        if roc_auc_score is not None and len(np.unique(y_val)) > 1:
            auc = float(roc_auc_score(y_val, y_pred_prob))
        quality_mean = float(val_df.loc[selected, quality_col].mean()) if selected.any() else 0.0
        upside_mean = float(val_df.loc[selected, upside_col].mean()) if selected.any() else 0.0
        downside_mean = float(val_df.loc[selected, downside_col].clip(upper=0).abs().mean()) if selected.any() else 0.0

        score = float(110.0 * quality_mean + 18.0 * upside_mean - 30.0 * downside_mean + 6.0 * precision + 2.0 * auc)
        metrics = {
            "precision": float(precision),
            "auc": float(auc),
            "quality_mean": quality_mean,
            "upside_mean": upside_mean,
            "downside_mean": downside_mean,
        }
        return score, metrics

    def train(self, df: pd.DataFrame):
        if xgb is None:
            print("XGBoost dependency is unavailable.")
            return

        horizon = settings.TRAIN_LABEL_HORIZON
        required_cols = self.feature_cols + ["target", "sample_weight", f"quality_{horizon}d", f"future_max_ret_{horizon}d", f"future_min_ret_{horizon}d"]
        train_df = df.dropna(subset=required_cols).copy()
        if train_df.empty:
            print("Training data is empty.")
            return

        split_idx = int(len(train_df) * 0.8)
        if split_idx <= 0 or split_idx >= len(train_df):
            print("Not enough data for train/validation split.")
            return

        train_part = train_df.iloc[:split_idx].copy()
        val_part = train_df.iloc[split_idx:].copy()

        X_train = train_part[self.feature_cols]
        y_train = train_part["target"]
        w_train = train_part["sample_weight"]
        X_val = val_part[self.feature_cols]
        y_val = val_part["target"]
        w_val = val_part["sample_weight"]

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=self.feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, feature_names=self.feature_cols)

        best_model = None
        best_score = -np.inf
        best_metrics = None

        for idx, params in enumerate(self._param_candidates(scale_pos_weight), start=1):
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train"), (dval, "eval")],
                early_stopping_rounds=25,
                verbose_eval=False,
            )
            y_pred_prob = model.predict(dval)
            score, metrics = self._validation_score(y_val, y_pred_prob, val_part)
            if score > best_score:
                best_model = model
                best_score = score
                best_metrics = metrics
            print(
                f"Candidate {idx}: score={score:.2f} precision={metrics['precision']:.3f} "
                f"auc={metrics['auc']:.3f} quality={metrics['quality_mean']:.4f}"
            )

        self.model = best_model
        self.is_trained = best_model is not None
        self.best_iteration = getattr(best_model, "best_iteration", None) if best_model is not None else None

        if not self.is_trained or best_metrics is None:
            print("Training failed.")
            return

        print(f"Validation score: {best_score:.2f}")
        print(f"Validation precision: {best_metrics['precision']:.3f}")
        print(f"Validation ROC-AUC: {best_metrics['auc']:.3f}")

    def predict(self, df: pd.DataFrame) -> float:
        if not self.is_trained and not self.load_model():
            return 0.0
        if df.empty or self.model is None or xgb is None:
            return 0.0
        current_data = df.iloc[[-1]][self.feature_cols]
        dtest = xgb.DMatrix(current_data, feature_names=self.feature_cols)
        prob = self.model.predict(dtest)[0]
        return round(float(prob), 4)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained and not self.load_model():
            return np.zeros(len(df), dtype=float)
        if self.model is None or xgb is None:
            return np.zeros(len(df), dtype=float)
        data = df[self.feature_cols]
        dtest = xgb.DMatrix(data, feature_names=self.feature_cols)
        return self.model.predict(dtest)

    def save_model(self):
        if self.model is None:
            raise ValueError("Model is not trained.")
        self.model.save_model(self.model_path)
        print(f"XGBoost model saved to {self.model_path}")

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None or not hasattr(self.model, "get_score"):
            return pd.DataFrame(columns=["feature", "importance"])
        scores = self.model.get_score(importance_type="gain")
        rows = [{"feature": feature, "importance": float(scores.get(feature, 0.0))} for feature in self.feature_cols]
        return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)

    def load_model(self) -> bool:
        if xgb is None:
            print("XGBoost dependency is unavailable. Skipping XGBoost model load.")
            return False
        try:
            if not os.path.exists(self.model_path):
                pkl_path = self.model_path.replace(".json", ".pkl")
                if os.path.exists(pkl_path) and joblib is not None:
                    print(f"Loading legacy model from {pkl_path}...")
                    self.model = joblib.load(pkl_path)
                    self.is_trained = True
                    return True
                print(f"Model file {self.model_path} not found.")
                return False

            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
