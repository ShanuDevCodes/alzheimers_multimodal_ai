import torch
import xgboost as xgb
import os
import pandas as pd
import numpy as np
from preprocessing.preprocess_mri import get_mri_val_transforms
from preprocessing.preprocess_tabular import TabularPreprocessor
import nibabel as nib
from utils.logging_utils import setup_logger

logger = setup_logger("Predictor")

class AlzheimerPredictor:
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device('cpu') # Enforce CPU for inference as required
        
        logger.info("Loading exported MultiModal models optimized for CPU...")
        
        self.cnn = torch.jit.load(os.path.join(model_dir, "mri_cnn.pt"), map_location=self.device)
        self.cnn.eval()
        
        self.fusion = torch.jit.load(os.path.join(model_dir, "fusion_mlp.pt"), map_location=self.device)
        self.fusion.eval()
        
        self.clin_gen_xgb = xgb.Booster()
        self.clin_gen_xgb.load_model(os.path.join(model_dir, "clinical_genetics_xgb.json"))
        
        self.life_xgb = xgb.Booster()
        self.life_xgb.load_model(os.path.join(model_dir, "lifestyle_xgb.json"))
        
        self.preprocessor = TabularPreprocessor()
        self.preprocessor.load(os.path.join(model_dir, "tabular_preprocessor.joblib"))
        
        self.mri_transform = get_mri_val_transforms()

    def predict(self, mri_path: str, tabular_data: dict):
        
        logger.info(f"Running prediction pipeline for patient...")
        
        if not os.path.exists(mri_path):
            raise FileNotFoundError(f"MRI file {mri_path} not found.")
            
        img = nib.load(mri_path).get_fdata()
        img = np.expand_dims(img, axis=0) # Add channel
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = self.mri_transform(img_tensor).unsqueeze(0) # Add batch dim
        
        with torch.no_grad():
            _, mri_embedding = self.cnn(img_tensor)
            
        df = pd.DataFrame([tabular_data])
        df_processed, _ = self.preprocessor.transform(df)
        
        clinical_features = self.clin_gen_xgb.feature_names
        lifestyle_features = self.life_xgb.feature_names
        
        X_clin = xgb.DMatrix(df_processed[clinical_features])
        X_life = xgb.DMatrix(df_processed[lifestyle_features])
        
        clin_score = self.clin_gen_xgb.predict(X_clin)[0]
        life_score = self.life_xgb.predict(X_life)[0]
        
        genetic_score = clin_score
        
        gen_tensor = torch.tensor([[genetic_score]], dtype=torch.float32)
        life_tensor = torch.tensor([[life_score]], dtype=torch.float32)
        clin_tensor = torch.tensor([[clin_score]], dtype=torch.float32)
        
        with torch.no_grad():
            final_logit = self.fusion(mri_embedding, gen_tensor, life_tensor, clin_tensor)
            probability = torch.sigmoid(final_logit).item()
            
        return {
            "alzheimer_probability": probability,
            "prediction_class": "Alzheimer's Disease" if probability > 0.5 else "No Alzheimer's",
            "component_scores": {
                "clinical_genetic_risk": float(clin_score),
                "lifestyle_risk": float(life_score)
            }
        }
