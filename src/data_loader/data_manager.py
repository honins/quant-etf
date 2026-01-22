import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from src.core.interfaces import DataProvider
from config.settings import settings

class DataManager:
    """
    负责数据的本地存储、增量更新和读取
    """
    def __init__(self, provider: DataProvider):
        self.provider = provider
        self.db_path = str(settings.DB_PATH)

    def get_latest_date(self, ts_code: str, table_name: str) -> str:
        """获取数据库中某标的的最新日期"""
        conn = sqlite3.connect(self.db_path)
        try:
            # 检查表是否存在
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                return settings.START_DATE

            query = f"SELECT MAX(trade_date) FROM {table_name} WHERE ts_code='{ts_code}'"
            cursor.execute(query)
            result = cursor.fetchone()
            if result and result[0]:
                return result[0]
            return settings.START_DATE
        except Exception as e:
            print(f"DB Error ({table_name}): {e}")
            return settings.START_DATE
        finally:
            conn.close()

    def update_and_get_data(self, ts_code: str, is_index: bool = False) -> pd.DataFrame:
        """
        1. 检查本地最新日期
        2. 从Provider拉取增量数据
        3. 存入本地数据库
        4. 返回完整数据
        """
        table_name = 'index_daily_data' if is_index else 'daily_data'
        last_date = self.get_latest_date(ts_code, table_name)
        today = datetime.now().strftime("%Y%m%d")
        
        # 如果本地最新日期就是今天（或更晚），直接读取
        if last_date >= today:
             return self._read_from_db(ts_code, table_name)

        # 计算起始日期 (last_date + 1 day)
        start_date = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        
        if start_date <= today:
            print(f"[{ts_code}] Fetching data from {start_date} to {today}...")
            
            if is_index:
                new_df = self.provider.get_index_daily(ts_code, start_date, today)
            else:
                new_df = self.provider.get_daily_data(ts_code, start_date, today)
            
            if not new_df.empty:
                self._save_to_db(new_df, table_name)
                print(f"[{ts_code}] Updated {len(new_df)} records.")
            else:
                print(f"[{ts_code}] No new data found.")
        
        return self._read_from_db(ts_code, table_name)

    def _save_to_db(self, df: pd.DataFrame, table_name: str):
        conn = sqlite3.connect(self.db_path)
        try:
            df.to_sql(table_name, conn, if_exists='append', index=False)
        except Exception as e:
            print(f"Save DB Error ({table_name}): {e}")
        finally:
            conn.close()

    def _read_from_db(self, ts_code: str, table_name: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        try:
            # 检查表是否存在，不存在直接返回空
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                return pd.DataFrame()

            query = f"SELECT * FROM {table_name} WHERE ts_code='{ts_code}' ORDER BY trade_date ASC"
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Read DB Error ({table_name}): {e}")
            return pd.DataFrame()
        finally:
            conn.close()
