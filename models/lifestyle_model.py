import xgboost as xgb
import os

class LifestyleModel:
    
    def __init__(self, use_gpu=False):
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,
            'learning_rate': 0.02,
            'subsample': 0.7
        }
        
        if use_gpu:
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cuda'
            
        self.model = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=100):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
            
        self.model = xgb.train(
            self.params, 
            dtrain, 
            num_boost_round=num_boost_round, 
            evals=evals, 
            early_stopping_rounds=10 if X_val is not None else None,
            verbose_eval=False
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
