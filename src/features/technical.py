import numpy as np
import pandas as pd

from config.settings import settings
from src.features.cross_section_features import add_relative_strength_features
from src.features.price_features import calculate_price_features
from src.features.regime_features import attach_regime_features, build_regime_feature_frame
from src.research.labeler import add_legacy_training_labels


class FeatureEngineer:
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df = df.sort_values("trade_date")
        return FeatureEngineer._calc_with_pandas(df)

    @staticmethod
    def _calc_with_pandas(df: pd.DataFrame) -> pd.DataFrame:
        return calculate_price_features(df, atr_period=settings.ATR_PERIOD)

    @staticmethod
    def add_relative_strength(etf_df: pd.DataFrame, index_df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        return add_relative_strength_features(etf_df, index_df, period=period)

    @staticmethod
    def add_regime_features(etf_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        index_feature_df = build_regime_feature_frame(index_df)
        return attach_regime_features(etf_df, index_feature_df)

    @staticmethod
    def add_labels(
        df: pd.DataFrame,
        horizon: int | None = None,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        return add_legacy_training_labels(df, horizon=horizon, threshold=threshold)
