import torch
import os
from models.cnn_model import MRICNN
from models.fusion_model import MultimodalFusionModel
from utils.logging_utils import setup_logger

logger = setup_logger("ExportModels")

def export_to_torchscript(model, dummy_input, save_path):
    
    model.eval()
    logger.info(f"Exporting model to TorchScript: {save_path}")
    
    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(save_path)
    logger.info(f"Successfully exported to {save_path}")

def export_all_models(config):
    
    save_dir = config['paths']['model_save_dir']
    
    cnn_path = os.path.join(save_dir, "mri_cnn.pth")
    if os.path.exists(cnn_path):
        cnn = MRICNN(embedding_size=config['model']['mri_embedding_size'])
        cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
        cnn.eval()
        
        dummy_mri = torch.randn(1, 1, 128, 128, 128)
        export_to_torchscript(cnn, dummy_mri, os.path.join(save_dir, "mri_cnn.pt"))
        
    fusion_path = os.path.join(save_dir, "fusion_mlp.pth")
    if os.path.exists(fusion_path):
        fusion = MultimodalFusionModel(mri_embedding_size=config['model']['mri_embedding_size'])
        fusion.load_state_dict(torch.load(fusion_path, map_location='cpu'))
        fusion.eval()
        
        dummy_embed = torch.randn(1, config['model']['mri_embedding_size'])
        dummy_gen = torch.randn(1, 1)
        dummy_life = torch.randn(1, 1)
        dummy_clin = torch.randn(1, 1)
        
        export_to_torchscript(
            fusion, 
            (dummy_embed, dummy_gen, dummy_life, dummy_clin), 
            os.path.join(save_dir, "fusion_mlp.pt")
        )

if __name__ == "__main__":
    from utils.config_loader import load_config
    config = load_config()
    export_all_models(config)
