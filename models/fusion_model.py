import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    
    def __init__(self, mri_embedding_size=256, num_tabular_scores=3):
        super(MultimodalFusionModel, self).__init__()
        
        input_size = mri_embedding_size + num_tabular_scores
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) # Binary Classification Logits
        )

    def forward(self, mri_embedding, genetic_score, lifestyle_score, clinical_score):
        
        if genetic_score.dim() == 1:
            genetic_score = genetic_score.unsqueeze(1)
        if lifestyle_score.dim() == 1:
            lifestyle_score = lifestyle_score.unsqueeze(1)
        if clinical_score.dim() == 1:
            clinical_score = clinical_score.unsqueeze(1)
            
        combined_features = torch.cat(
            [mri_embedding, genetic_score, lifestyle_score, clinical_score], 
            dim=1
        )
        
        return self.fusion_mlp(combined_features)
