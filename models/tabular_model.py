import xgboost as xgb
import os
import joblib

class TabularXGBoostModel:
    
    def __init__(self, use_gpu=False):
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        if use_gpu:
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cuda'
            
        self.model = None
        
    def train(self, X_train, y_train, X_val, y_val, num_boost_round=100):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        self.model = xgb.train(
            self.params, 
            dtrain, 
            num_boost_round=num_boost_round, 
            evals=evals, 
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        
    def load(self, filepath):
        self.model = xgb.Booster()
        self.model.load_model(filepath)
