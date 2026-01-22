from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    """数据提供商接口 (遵循依赖倒置原则)"""
    
    @abstractmethod
    def get_daily_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        pass

    @abstractmethod
    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数日线数据 (用于大势研判)"""
        pass
