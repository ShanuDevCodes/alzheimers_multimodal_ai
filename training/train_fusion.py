import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models.fusion_model import MultimodalFusionModel
from utils.logging_utils import setup_logger
from utils.config_loader import load_config

logger = setup_logger("TrainFusion")

class FusionDataset(Dataset):
    
    def __init__(self, num_samples, embedding_size):
        self.num_samples = num_samples
        self.mri_embeddings = torch.randn(num_samples, embedding_size)
        self.genetic_scores = torch.rand(num_samples, 1)
        self.lifestyle_scores = torch.rand(num_samples, 1)
        self.clinical_scores = torch.rand(num_samples, 1)
        self.labels = torch.randint(0, 2, (num_samples, 1)).float()
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            'mri': self.mri_embeddings[idx],
            'genetics': self.genetic_scores[idx],
            'lifestyle': self.lifestyle_scores[idx],
            'clinical': self.clinical_scores[idx],
            'label': self.labels[idx]
        }

def train_multimodal_fusion(config_path="configs/default_config.yaml"):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} for Fusion Model")
    
    embedding_size = config['model']['mri_embedding_size']
    
    dataset = FusionDataset(num_samples=100, embedding_size=embedding_size)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    model = MultimodalFusionModel(mri_embedding_size=embedding_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    logger.info("Starting Fusion model training...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            mri = batch['mri'].to(device)
            gen = batch['genetics'].to(device)
            life = batch['lifestyle'].to(device)
            clin = batch['clinical'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(mri, gen, life, clin)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Fusion Loss: {epoch_loss/len(loader):.4f}")
        
    save_path = os.path.join(config['paths']['model_save_dir'], "fusion_mlp.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved Fusion Model to {save_path}")

if __name__ == "__main__":
    train_multimodal_fusion()
