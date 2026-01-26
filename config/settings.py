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
    ATR_MULTIPLIER = 2.0  # 止损 = 2倍ATR
    EXIT_LOOKBACK_PERIOD = 22 # 吊灯止损回溯周期

    # 策略参数
    # 牛市进攻模式标的
    AGGRESSIVE_TICKERS = [
        "588000.SH", # 科创50
        "159915.SZ", # 创业板
        "512480.SH", # 半导体
        "515030.SH"  # 新能源车
    ]

settings = Settings()
