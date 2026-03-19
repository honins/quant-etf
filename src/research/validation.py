from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import pandas as pd


DATE_FMT = "%Y%m%d"


@dataclass(slots=True)
class ValidationWindow:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    mode: str
    purge_days: int = 0

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(str(value), DATE_FMT)


def generate_walk_forward_windows(
    earliest_date: str,
    latest_date: str,
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    mode: str = "rolling",
    purge_days: int = 0,
) -> list[ValidationWindow]:
    mode = mode.lower()
    if mode not in {"rolling", "anchored"}:
        raise ValueError(f"Unsupported validation mode: {mode}")

    start = _parse_date(earliest_date)
    end = _parse_date(latest_date)
    base_train_span = timedelta(days=max(train_window_days - 1, 0))
    test_span = timedelta(days=max(test_window_days - 1, 0))
    step = timedelta(days=max(step_days, 1))
    purge = timedelta(days=max(purge_days, 0))

    windows: list[ValidationWindow] = []
    cursor = start
    fold = 1

    while True:
        train_start = start if mode == "anchored" else cursor
        train_end = cursor + base_train_span
        test_start = train_end + timedelta(days=1) + purge
        test_end = test_start + test_span

        if test_end > end:
            break

        windows.append(
            ValidationWindow(
                fold=fold,
                train_start=train_start.strftime(DATE_FMT),
                train_end=train_end.strftime(DATE_FMT),
                test_start=test_start.strftime(DATE_FMT),
                test_end=test_end.strftime(DATE_FMT),
                mode=mode,
                purge_days=purge_days,
            )
        )

        fold += 1
        cursor += step

    return windows


def generate_purged_walk_forward_windows(
    earliest_date: str,
    latest_date: str,
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    purge_days: int,
    mode: str = "rolling",
) -> list[ValidationWindow]:
    return generate_walk_forward_windows(
        earliest_date=earliest_date,
        latest_date=latest_date,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        step_days=step_days,
        mode=mode,
        purge_days=purge_days,
    )


def split_windows_by_regime(
    windows: list[ValidationWindow],
    market_status_map: dict[str, str],
) -> dict[str, list[ValidationWindow]]:
    grouped: dict[str, list[ValidationWindow]] = {
        "Bull Market": [],
        "Bear Market": [],
        "Volatile Market": [],
        "Unknown Market": [],
    }
    for window in windows:
        regime = market_status_map.get(window.test_end, "Unknown Market")
        grouped.setdefault(regime, []).append(window)
    return grouped


def split_dataframe_by_window(
    df: pd.DataFrame,
    window: ValidationWindow,
    date_col: str = "trade_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Series(df[date_col], dtype=str)
    train_mask = (dates >= window.train_start) & (dates <= window.train_end)
    test_mask = (dates >= window.test_start) & (dates <= window.test_end)
    train_df = df.loc[train_mask.to_numpy()].copy()
    test_df = df.loc[test_mask.to_numpy()].copy()
    return train_df, test_df


def attach_regime_label(
    df: pd.DataFrame,
    market_status_map: dict[str, str],
    date_col: str = "trade_date",
    target_col: str = "regime",
) -> pd.DataFrame:
    labeled = df.copy()
    date_values = pd.Series(labeled[date_col], dtype=str)
    labeled[target_col] = date_values.map(lambda value: market_status_map.get(value))
    return labeled
