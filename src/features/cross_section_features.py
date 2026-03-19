from __future__ import annotations

import numpy as np
import pandas as pd


def add_relative_strength_features(
    etf_df: pd.DataFrame,
    index_df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    if etf_df.empty or index_df.empty:
        return etf_df.copy()

    result = etf_df.copy()
    index_close = pd.Series(index_df.set_index("trade_date")["close"], dtype=float)
    aligned_index_ret = result["trade_date"].map(index_close.pct_change(period).to_dict())
    result[f"rs_{period}d"] = result["close"].pct_change(period) - aligned_index_ret

    if "atr_pct" in result.columns and "atr_pct" in index_df.columns:
        index_atr_pct = pd.Series(index_df.set_index("trade_date")["atr_pct"], dtype=float)
        aligned_index_atr_pct = result["trade_date"].map(index_atr_pct.to_dict())
        result["rel_vol"] = result["atr_pct"] / aligned_index_atr_pct.replace(0, np.nan)

    return result
