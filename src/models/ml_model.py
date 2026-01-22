import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from src.models.scoring_model import BaseModel

class MLModel(BaseModel):
    def __init__(self, model_path="data/rf_model.pkl"):
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            min_samples_leaf=10, 
            random_state=42,
            class_weight="balanced"
        )
        self.feature_cols = [
            'ma5', 'ma20', 'ma60', 
            'rsi_6', 'rsi_14', 
            'atr', 'obv', 
            'macd', 'macdsignal', 'macdhist',
            'upper', 'lower'
        ]
        self.model_path = model_path
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """
        训练模型
        df: 包含技术指标和 target 列的 DataFrame
        """
        # 清洗数据：去除包含 NaN 的行
        train_df = df.dropna(subset=self.feature_cols + ['target'])
        
        if train_df.empty:
            print("Training data is empty!")
            return
            
        X = train_df[self.feature_cols]
        y = train_df['target']
        
        # 训练模型
        self.model.fit(X, y)
        self.is_trained = True
        
        # 评估
        y_pred = self.model.predict(X)
        print(f"Train Precision: {precision_score(y, y_pred):.2f}")
        print(f"Train ROC-AUC: {roc_auc_score(y, self.model.predict_proba(X)[:, 1]):.2f}")

    def predict(self, df: pd.DataFrame) -> float:
        """
        预测最新一天的得分 (0-1)
        """
        if not self.is_trained:
            # 尝试加载
            if not self.load_model():
                return 0.0
                
        if df.empty:
            return 0.0
            
        # 只需要最后一行
        current_data = df.iloc[[-1]][self.feature_cols]
        
        # 填充缺失值 (以防万一)
        current_data = current_data.fillna(0)
        
        # 预测概率
        prob = self.model.predict_proba(current_data)[0][1]
        return round(prob, 2)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        批量预测 (用于回测)
        """
        if not self.is_trained:
            return np.zeros(len(df))
            
        data = df[self.feature_cols].fillna(0)
        return self.model.predict_proba(data)[:, 1]

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        try:
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            return True
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found.")
            return False
