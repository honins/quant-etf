import tushare as ts
import pandas as pd
from src.core.interfaces import DataProvider
from config.settings import settings
import time

class TushareLoader(DataProvider):
    def __init__(self):
        if not settings.TUSHARE_TOKEN:
            raise ValueError("未配置 TUSHARE_TOKEN，请在 .env 文件中设置。")
        
        self.pro = ts.pro_api(settings.TUSHARE_TOKEN)

    def get_daily_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线行情
        Tushare 接口: daily / fund_daily
        """
        try:
            # 判断是ETF还是股票
            if ts_code.startswith("51") or ts_code.startswith("15") or ts_code.startswith("58"):
                # ETF/Fund
                df = self.pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                # Stock
                df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                return pd.DataFrame()

            # 统一列名格式，确保后续处理一致
            # Tushare 返回: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
            df = df.sort_values('trade_date').reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ts_code}: {e}")
            return pd.DataFrame()

    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                df = df.sort_values('trade_date').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error fetching index data for {ts_code}: {e}")
            return pd.DataFrame()
