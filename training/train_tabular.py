import os
import pandas as pd
from sklearn.model_selection import train_test_split
from models.tabular_model import TabularXGBoostModel
from utils.logging_utils import setup_logger
from utils.config_loader import load_config
import torch

logger = setup_logger("TrainTabular")

def train_tabular_models(config_path="configs/default_config.yaml"):
    config = load_config(config_path)
    processed_csv = os.path.join(config['paths']['processed_dir'], "processed_tabular.csv")
    
    if not os.path.exists(processed_csv):
        logger.warning(f"Processed tabular data not found at {processed_csv}. Skipping.")
        return
        
    df = pd.read_csv(processed_csv)
    
    label_col = config['training'].get('label_col', 'Diagnosis')
    exclude_cols = ['patient_id', label_col]
    features = [c for c in df.columns if c not in exclude_cols]
    
    lifestyle_keywords = ['sleep', 'activity', 'smoking', 'alcohol', 'diet_quality', 'bmi']
    lifestyle_features = [f for f in features if any(k in f.lower() for k in lifestyle_keywords)]
    clinical_features  = [f for f in features if f not in lifestyle_features]
    
    logger.info(f"Clinical/Genetic features: {len(clinical_features)}")
    logger.info(f"Lifestyle features: {len(lifestyle_features)}")
    
    X = df[features]
    y = df[label_col]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config['training']['seed'])
    
    use_gpu = torch.cuda.is_available()
    
    logger.info("Training Clinical/Genetics XGBoost model...")
    clin_model = TabularXGBoostModel(use_gpu=use_gpu)
    clin_model.train(X_train[clinical_features], y_train, X_val[clinical_features], y_val)
    
    clin_save_path = os.path.join(config['paths']['model_save_dir'], "clinical_genetics_xgb.json")
    clin_model.save(clin_save_path)
    logger.info(f"Saved Clinical/Genetics model to {clin_save_path}")
    
    logger.info("Training Lifestyle XGBoost model...")
    from models.lifestyle_model import LifestyleModel
    life_model = LifestyleModel(use_gpu=use_gpu)
    life_model.train(X_train[lifestyle_features], y_train, X_val[lifestyle_features], y_val)
    
    life_save_path = os.path.join(config['paths']['model_save_dir'], "lifestyle_xgb.json")
    life_model.save(life_save_path)
    logger.info(f"Saved Lifestyle model to {life_save_path}")

if __name__ == "__main__":
    train_tabular_models()
