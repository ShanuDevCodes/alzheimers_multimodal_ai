import os
import argparse
import json
import torch
import warnings
import pandas as pd
from PIL import Image
import xgboost as xgb
from torchvision import transforms

warnings.filterwarnings("ignore")

try:
    from utils.config_loader import load_config
    config = load_config('configs/default_config.yaml')
    mri_weights = config['paths']['cnn_weights']
    clinical_weights = config['paths']['tabular_models']['clinical_genetics']
    lifestyle_weights = config['paths']['tabular_models']['lifestyle']
    fusion_mlp = config['paths']['fusion_model']
except Exception:
    mri_weights = "models/saved/mri_cnn_best.pth"
    clinical_weights = "models/saved/clinical_genetics_xgb.json"
    lifestyle_weights = "models/saved/lifestyle_xgb.json"
    fusion_mlp = "models/saved/fusion_mlp.pth"

from models.cnn_model import MRICNN
from models.tabular_model import TabularXGBoostModel
from models.fusion_model import MultimodalFusionModel

def run_inference(tabular_csv, mri_img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading multimodal models onto {device}...")
    
    clinical_model = TabularXGBoostModel()
    clinical_model.load(clinical_weights)
    
    lifestyle_model = TabularXGBoostModel()
    lifestyle_model.load(lifestyle_weights)
    
    cnn = MRICNN(embedding_size=256).to(device)
    cnn.load_state_dict(torch.load(mri_weights, map_location=device))
    cnn.eval()
    
    fusion = MultimodalFusionModel(mri_embedding_size=256, num_tabular_scores=3).to(device)
    fusion.load_state_dict(torch.load(fusion_mlp, map_location=device))
    fusion.eval()
    
    print("Models loaded successfully. Processing patient data...\n")
    print("-" * 50)
    
    df = pd.read_csv(tabular_csv)
    sample = df.iloc[[0]]
    
    exclude_cols = ['patient_id', 'Diagnosis', 'alzheimer_risk']
    features = [c for c in sample.columns if c not in exclude_cols]
    
    lifestyle_keywords = ['sleep', 'activity', 'smoking', 'alcohol', 'diet_quality', 'bmi']
    lifestyle_features = [f for f in features if any(k in f.lower() for k in lifestyle_keywords)]
    clinical_features  = [f for f in features if f not in lifestyle_features]
    
    X_clin = sample[[c for c in clinical_features if c in sample.columns]]
    X_life = sample[[c for c in lifestyle_features if c in sample.columns]]
    
    p_clin = clinical_model.predict_proba(X_clin)
    p_life = lifestyle_model.predict_proba(X_life)
    
    print(f"🔬 Clinical & Genetic Risk: {p_clin[0]*100:.1f}%")
    print(f"🌿 Lifestyle Risk:         {p_life[0]*100:.1f}%")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(mri_img).convert('RGB') # DenseNet expects 3 channels
    tensor_img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mri_out, cnn_features = cnn(tensor_img)
        p_mri = torch.sigmoid(mri_out).item()
        
    print(f"🖥  MRI Structural Risk:    {p_mri*100:.1f}%")
    
    clin_tensor = torch.tensor([[float(p_clin[0])]], dtype=torch.float32).to(device)
    life_tensor = torch.tensor([[float(p_life[0])]], dtype=torch.float32).to(device)
    gen_tensor = torch.tensor([[0.5]], dtype=torch.float32).to(device) # Mock genetics score
    
    with torch.no_grad():
        fusion_out = fusion(cnn_features, gen_tensor, life_tensor, clin_tensor)
        final_risk = torch.sigmoid(fusion_out).item()
        
    print("-" * 50)
    print(f"🔥 FINAL MULTIMODAL ALZHEIMER'S RISK: {final_risk*100:.1f}% 🔥")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tabular', type=str, required=True, help="Processed CSV with test row")
    parser.add_argument('--mri', type=str, required=True, help="Path to test patient MRI image")
    args = parser.parse_args()
    
    if not os.path.exists(args.tabular):
        print(f"ERROR: Tabular file not found: {args.tabular}")
    elif not os.path.exists(args.mri):
        print(f"ERROR: MRI file not found: {args.mri}")
    else:
        run_inference(args.tabular, args.mri)
