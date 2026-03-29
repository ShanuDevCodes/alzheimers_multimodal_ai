import argparse
import json
import os
from inference.predict import AlzheimerPredictor
from utils.logging_utils import setup_logger
from utils.config_loader import load_config

logger = setup_logger("RunInference")

def run(mri_path: str, tabular_json_path: str, config_path: str = "configs/default_config.yaml"):
    config = load_config(config_path)
    model_dir = config['paths']['model_save_dir']
    
    if not os.path.exists(os.path.join(model_dir, "fusion_mlp.pt")):
        logger.error("TorchScript models not found. Please run scripts/export_models.py first.")
        return
        
    predictor = AlzheimerPredictor(model_dir)
    
    with open(tabular_json_path, 'r') as f:
        tabular_data = json.load(f)
        
    result = predictor.predict(mri_path, tabular_data)
    
    print("\n" + "="*50)
    print("🧠 ALZHEIMER'S RISK PREDICTION REPORT")
    print("="*50)
    print(f"Prediction:   {result['prediction_class']}")
    print(f"Probability:  {result['alzheimer_probability']*100:.2f}%")
    print("-" * 50)
    print("Component Breakdown:")
    print(f"  Clinical/Genetics Risk Score: {result['component_scores']['clinical_genetic_risk']*100:.2f}%")
    print(f"  Lifestyle Risk Score:         {result['component_scores']['lifestyle_risk']*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal inference")
    parser.add_argument("--mri", type=str, required=True, help="Path to NIfTI MRI volume (.nii or .nii.gz)")
    parser.add_argument("--tabular", type=str, required=True, help="Path to JSON file containing patient tabular data")
    
    args = parser.parse_args()
    run(args.mri, args.tabular)
