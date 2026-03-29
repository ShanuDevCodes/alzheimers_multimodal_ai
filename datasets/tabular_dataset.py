import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    
    def __init__(self, csv_file: str, features: list, label_col: str = 'alzheimer_risk'):
        self.data_frame = pd.read_csv(csv_file)
        self.features = features
        self.label_col = label_col

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.data_frame.iloc[idx]['patient_id']
        
        features_array = self.data_frame.iloc[idx][self.features].values.astype(np.float32)
        target = self.data_frame.iloc[idx][self.label_col]
        
        return {
            'features': torch.tensor(features_array),
            'label': torch.tensor(target, dtype=torch.long),
            'patient_id': patient_id
        }

def load_data_for_xgboost(csv_file: str, features: list, label_col: str = 'alzheimer_risk'):
    
    df = pd.read_csv(csv_file)
    X = df[features]
    y = df[label_col]
    patient_ids = df['patient_id']
    return X, y, patient_ids
