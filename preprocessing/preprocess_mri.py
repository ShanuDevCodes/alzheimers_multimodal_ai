from torchvision import transforms as T

def get_mri_train_transforms(image_size: int = 224):
    
    return T.Compose([
        T.Resize((image_size + 20, image_size + 20)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def get_mri_val_transforms(image_size: int = 224):
    
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
