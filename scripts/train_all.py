import os
from utils.logging_utils import setup_logger
from utils.seed_utils import set_seed
from utils.config_loader import load_config
from training.train_cnn import train_mri_cnn
from training.train_tabular import train_tabular_models
from training.train_fusion import train_multimodal_fusion

logger = setup_logger("TrainAllPipeline")

def main():
    logger.info("Starting End-to-End Multimodal AI Pipeline")
    
    config = load_config()
    set_seed(config['training']['seed'])
    
    logger.info("=== Phase 1: Training Tabular XGBoost Models ===")
    train_tabular_models()
    
    logger.info("=== Phase 2: Training MRI CNN ===")
    train_mri_cnn()
    
    logger.info("=== Phase 3: Training Multimodal Fusion ===")
    train_multimodal_fusion()
    
    logger.info("Multimodal training pipeline completed successfully!")

if __name__ == "__main__":
    main()
