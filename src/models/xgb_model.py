import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score
from src.models.scoring_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, model_path="data/xgb_model.pkl"):
        # XGBoost 参数配置
        self.params = {
            'objective': 'binary:logistic',
            'max_depth': 5,
            'eta': 0.05,            # 学习率
            'subsample': 0.8,       # 随机采样，防止过拟合
            'colsample_bytree': 0.8,
            'eval_metric': 'auc',
            'scale_pos_weight': 3,  # 样本不平衡处理 (假设正样本少)
            'seed': 42
        }
        self.num_boost_round = 200
        
        self.feature_cols = [
            'ma5', 'ma20', 'ma60', 
            'rsi_6', 'rsi_14', 
            'atr', 'obv', 
            'macd', 'macdsignal', 'macdhist',
            'upper', 'lower'
        ]
        self.model_path = model_path
        self.model = None
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """
        训练模型
        """
        train_df = df.dropna(subset=self.feature_cols + ['target'])
        
        if train_df.empty:
            print("Training data is empty!")
            return
            
        X = train_df[self.feature_cols]
        y = train_df['target']
        
        # 转换为 DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # 训练
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        self.is_trained = True
        
        # 训练集评估
        y_pred_prob = self.model.predict(dtrain)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        print(f"XGB Train Precision: {precision_score(y, y_pred):.2f}")
        print(f"XGB Train ROC-AUC: {roc_auc_score(y, y_pred_prob):.2f}")

    def predict(self, df: pd.DataFrame) -> float:
        """
        预测最新一天的得分 (0-1)
        """
        if not self.is_trained:
            if not self.load_model():
                return 0.0
                
        if df.empty:
            return 0.0
            
        current_data = df.iloc[[-1]][self.feature_cols].fillna(0)
        dtest = xgb.DMatrix(current_data)
        
        prob = self.model.predict(dtest)[0]
        return round(float(prob), 2)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        批量预测
        """
        if not self.is_trained:
            return np.zeros(len(df))
            
        data = df[self.feature_cols].fillna(0)
        dtest = xgb.DMatrix(data)
        return self.model.predict(dtest)

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"XGBoost Model saved to {self.model_path}")

    def load_model(self) -> bool:
        try:
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            return True
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found.")
            return False
