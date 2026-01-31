import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Settings:
    # 基础路径
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    REPORTS_DIR = BASE_DIR / "reports"
    
    # 确保目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 数据库路径
    DB_PATH = DATA_DIR / "market_data.db"

    # Tushare 配置
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
    
    # 回测/分析配置
    START_DATE = "20200101"  # 默认回溯开始时间
    
    # 风险控制参数
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 1.3  # 止损 = 1.3倍ATR
    ATR_MULTIPLIER_AGGRESSIVE = 3.5 # 激进标的止损 = 3.5倍ATR (防止被洗出局)
    EXIT_LOOKBACK_PERIOD = 22 # 吊灯止损回溯周期
    MAX_DRAWDOWN_STOP = 0.05

    # 策略参数
    # 牛市进攻模式标的
    AGGRESSIVE_TICKERS = [
        "588000.SH", # 科创50
        "159915.SZ", # 创业板
        "512480.SH", # 半导体
        "515030.SH", # 新能源车
        "515070.SH"
    ]
    TICKER_BULL_THRESHOLDS = {
        "588000.SH": 0.60,
        "515070.SH": 0.60
    }
    USE_DYNAMIC_THRESHOLD = True
    DYNAMIC_THRESHOLD_LOOKBACK = 60
    DYNAMIC_THRESHOLD_QUANTILE = 0.85
    DYNAMIC_THRESHOLD_MIN = 0.55
    DYNAMIC_THRESHOLD_MAX = 0.75

settings = Settings()
