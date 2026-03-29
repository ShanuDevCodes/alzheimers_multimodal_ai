import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.cnn_model import MRICNN
from datasets.mri_dataset import MRIDataset
from preprocessing.preprocess_mri import get_mri_train_transforms, get_mri_val_transforms
from utils.logging_utils import setup_logger
from utils.config_loader import load_config

logger = setup_logger("TrainCNN")

def train_mri_cnn(config_path="configs/default_config.yaml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    mri_dir        = config["paths"]["mri_data"]
    embedding_size = config["model"]["mri_embedding_size"]
    batch_size     = config["training"]["batch_size"]
    epochs         = config["training"]["epochs"]
    lr_cnn         = float(config["training"]["learning_rate_cnn"])
    seed           = int(config["training"].get("seed", 42))

    full_dataset = MRIDataset(mri_dir, transform=get_mri_train_transforms())

    if len(full_dataset) == 0:
        logger.warning(f"No MRI images found in '{mri_dir}'.")
        logger.warning("Run: python scripts/download_dataset.py  first.")
        return

    dist = full_dataset.class_distribution()
    logger.info(f"MRI dataset: {len(full_dataset)} images  |  Class distribution: {dist}")

    val_size   = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    torch.manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    val_ds.dataset.transform = get_mri_val_transforms()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    logger.info(f"Train: {train_size} samples  |  Val: {val_size} samples")

    n_pos = dist.get(1, 1)
    n_neg = dist.get(0, 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    logger.info(f"pos_weight for BCELoss: {pos_weight.item():.2f}  (handles AD/CN imbalance)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = MRICNN(embedding_size=embedding_size, freeze_backbone=True).to(device)
    logger.info("Phase 1 — Warm-up: backbone frozen, training head only")

    warmup_epochs = max(3, epochs // 10)
    finetune_epochs = epochs - warmup_epochs

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    def run_epoch(loader, train_mode):
        model.train(train_mode)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train_mode):
            for batch in loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device).float().unsqueeze(1)
                if train_mode:
                    optimizer.zero_grad()
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
                if train_mode:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds.squeeze() == labels.squeeze().long()).sum().item()
                total   += labels.size(0)
        return total_loss / len(loader), correct / total

    best_val_loss = float("inf")

    for epoch in range(warmup_epochs):
        tr_loss, tr_acc = run_epoch(train_loader, train_mode=True)
        va_loss, va_acc = run_epoch(val_loader,   train_mode=False)
        logger.info(
            f"[Warmup {epoch+1}/{warmup_epochs}] "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss:   {va_loss:.4f} Acc: {va_acc:.3f}"
        )

    logger.info(f"Phase 2 — Fine-tune: unfreezing backbone, lr={lr_cnn}")
    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=lr_cnn)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    for epoch in range(finetune_epochs):
        tr_loss, tr_acc = run_epoch(train_loader, train_mode=True)
        va_loss, va_acc = run_epoch(val_loader,   train_mode=False)
        scheduler.step()
        logger.info(
            f"[Finetune {epoch+1}/{finetune_epochs}] "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss:   {va_loss:.4f} Acc: {va_acc:.3f}"
        )
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            save_path = os.path.join(config["paths"]["model_save_dir"], "mri_cnn_best.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved best model (val_loss={va_loss:.4f}) → {save_path}")

    final_path = os.path.join(config["paths"]["model_save_dir"], "mri_cnn.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final CNN model → {final_path}")

if __name__ == "__main__":
    train_mri_cnn()
