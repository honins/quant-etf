import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score
from src.models.scoring_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, model_path="data/xgb_model.json"):
        # XGBoost 参数配置
        self.params = {
            'objective': 'binary:logistic',
            'max_depth': 5,
            # 'learning_rate' and 'scale_pos_weight' will be set dynamically in train()
            'subsample': 0.8,       # 随机采样，防止过拟合
            'colsample_bytree': 0.8,
            'eval_metric': 'auc',
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
        
        # 动态计算 scale_pos_weight
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # 更新参数
        self.params['scale_pos_weight'] = scale_pos_weight
        self.params['learning_rate'] = self.params.pop('eta', 0.05) # Rename eta to learning_rate

        # 划分训练集和验证集 (80% Train, 20% Val)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # 转换为 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)
        
        # 训练 (引入 Early Stopping)
        evals = [(dtrain, 'train'), (dval, 'eval')]
        self.model = xgb.train(
            self.params, 
            dtrain, 
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=20
        )
        self.is_trained = True
        
        # 打印最佳迭代次数
        print(f"Best iteration: {self.model.best_iteration}")
        
        # 验证集评估
        y_pred_prob = self.model.predict(dval)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        print(f"XGB Val Precision: {precision_score(y_val, y_pred):.2f}")
        print(f"XGB Val ROC-AUC: {roc_auc_score(y_val, y_pred_prob):.2f}")

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
        dtest = xgb.DMatrix(current_data, feature_names=self.feature_cols)
        
        prob = self.model.predict(dtest)[0]
        return round(float(prob), 2)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        批量预测
        """
        if not self.is_trained:
            return np.zeros(len(df))
            
        data = df[self.feature_cols].fillna(0)
        dtest = xgb.DMatrix(data, feature_names=self.feature_cols)
        return self.model.predict(dtest)

    def save_model(self):
        # 使用 XGBoost 原生保存方法 (推荐 JSON)
        self.model.save_model(self.model_path)
        print(f"XGBoost Model saved to {self.model_path}")

    def load_model(self) -> bool:
        try:
            # 检查文件是否存在
            import os
            if not os.path.exists(self.model_path):
                # 尝试兼容旧的 pkl 文件 (如果 json 不存在)
                pkl_path = self.model_path.replace('.json', '.pkl')
                if os.path.exists(pkl_path):
                    print(f"Loading legacy model from {pkl_path}...")
                    self.model = joblib.load(pkl_path)
                    self.is_trained = True
                    return True
                print(f"Model file {self.model_path} not found.")
                return False
            
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
