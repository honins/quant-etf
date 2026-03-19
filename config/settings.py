import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    REPORTS_DIR = BASE_DIR / "reports"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    DB_PATH = DATA_DIR / "market_data.db"
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
    START_DATE = "20200101"

    ATR_PERIOD = 14
    ATR_MULTIPLIER = 1.3
    ATR_MULTIPLIER_AGGRESSIVE = 2.0
    EXIT_LOOKBACK_PERIOD = 15
    MAX_DRAWDOWN_STOP = 0.10

    BULL_BASE_THRESHOLD = 0.60
    BULL_AGGRESSIVE_THRESHOLD = 0.55
    VOLATILE_THRESHOLD = 0.62
    BEAR_THRESHOLD = 0.75
    SIGNAL_EXIT_THRESHOLD = 0.40

    AGGRESSIVE_TICKERS = [
        "588000.SH",
        "159915.SZ",
        "512480.SH",
        "515030.SH",
        "515070.SH",
    ]
    TICKER_BULL_THRESHOLDS = {
        "588000.SH": 0.60,
        "515070.SH": 0.60,
    }

    USE_DYNAMIC_THRESHOLD = True
    DYNAMIC_THRESHOLD_LOOKBACK = 45
    DYNAMIC_THRESHOLD_QUANTILE = 0.90
    DYNAMIC_THRESHOLD_MIN = 0.55
    DYNAMIC_THRESHOLD_MAX = 0.75

    MARKET_STATE_CONFIRM_DAYS = 3

    TRAIN_LABEL_HORIZON = 7
    TRAIN_LABEL_THRESHOLD = 0.025
    TRAIN_LABEL_END_WEIGHT = 0.30
    TRAIN_LABEL_DRAWDOWN_PENALTY = 1.20

    LIVE_MODEL_FREEZE_DAYS = int(os.getenv("LIVE_MODEL_FREEZE_DAYS", "7"))
    CROSS_SECTION_TOP_K = int(os.getenv("CROSS_SECTION_TOP_K", "6"))
    CROSS_SECTION_MIN_SCORE = float(os.getenv("CROSS_SECTION_MIN_SCORE", "0.58"))
    CROSS_SECTION_CORE_TOP_K = int(os.getenv("CROSS_SECTION_CORE_TOP_K", "0"))
    CROSS_SECTION_SATELLITE_TOP_K = int(os.getenv("CROSS_SECTION_SATELLITE_TOP_K", "0"))


settings = Settings()
