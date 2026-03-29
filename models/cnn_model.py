import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class MRICNN(nn.Module):
    
    def __init__(self, embedding_size: int = 256, dropout_prob: float = 0.3,
                 freeze_backbone: bool = False):
        super(MRICNN, self).__init__()

        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embed = nn.Sequential(
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(embedding_size, 1),
        )

    def forward(self, x):
        
        feats     = self.backbone(x)
        pooled    = self.pool(feats)
        flat      = pooled.view(pooled.size(0), -1)
        embedding = self.embed(flat)
        logits    = self.fc(embedding)
        return logits, embedding

    def get_embedding(self, x):
        
        feats     = self.backbone(x)
        pooled    = self.pool(feats)
        flat      = pooled.view(pooled.size(0), -1)
        return self.embed(flat)

    def unfreeze_backbone(self):
        
        for param in self.backbone.parameters():
            param.requires_grad = True
