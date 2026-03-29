import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

class MRIDataset(Dataset):
    
    CLASS_MAP = {"AD": 1, "CN_MCI": 0}
    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    def __init__(self, data_dir: str, labels_df=None, transform=None):
        self.data_dir  = data_dir
        self.transform = transform
        self.files     = []

        if os.path.isdir(data_dir):
            for class_name, class_label in self.CLASS_MAP.items():
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for fname in sorted(os.listdir(class_dir)):
                        if fname.lower().endswith(self.SUPPORTED_EXTS):
                            self.files.append(
                                (os.path.join(class_dir, fname), class_label)
                            )
                else:
                    pass

        if not self.files and os.path.isdir(data_dir):
            all_imgs = []
            for fname in sorted(os.listdir(data_dir)):
                if fname.lower().endswith(self.SUPPORTED_EXTS):
                    all_imgs.append(os.path.join(data_dir, fname))

            if all_imgs and labels_df is not None:
                label_vals = labels_df["Diagnosis"].values
                for i, fp in enumerate(all_imgs):
                    self.files.append((fp, int(label_vals[i % len(label_vals)])))
            elif all_imgs:
                for fp in all_imgs:
                    self.files.append((fp, 0))

        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        img = Image.open(file_path).convert("RGB")
        img_tensor = self.transform(img)
        patient_id = os.path.splitext(os.path.basename(file_path))[0]
        return {
            "image":      img_tensor,
            "label":      torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
        }

    def class_distribution(self):
        
        from collections import Counter
        return dict(Counter(lbl for _, lbl in self.files))
