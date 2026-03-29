import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def create_default_config(save_path: str = "configs/default_config.yaml"):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    default_config = {
        "paths": {
            "mri_data": "data/raw/mri/",
            "tabular_data": "data/raw/tabular/patient_data.csv",
            "processed_dir": "data/processed/",
            "model_save_dir": "models/saved/"
        },
        "training": {
            "batch_size": 16,
            "learning_rate_cnn": 1e-4,
            "learning_rate_tabular": 0.05,
            "epochs": 50,
            "seed": 42
        },
        "model": {
            "cnn_architecture": "densenet121",
            "mri_embedding_size": 256,
            "tabular_model_type": "xgboost"
        }
    }
    
    with open(save_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

if __name__ == "__main__":
    create_default_config()
