# scripts/train_finetune.py

import torch
import timm
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# --- Config ---
ROOT_DIR = Path(__file__).resolve().parents[1]
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 150
NUM_WORKERS = 16
SEED = 52
PATIENCE = 15  # Early stopping

# Paths
DATA_PATH = ROOT_DIR / "datasets/frames"
MODEL_SAVE_PATH = ROOT_DIR / "backend/models"
LOG_FILE_PATH = ROOT_DIR / "150_training_log.txt"
MODEL_SAVE_PATH.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger()

# --- Label smoothing for BCE ---
class SmoothBCEwLogits(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)

# --- Main ---
def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    amp_enabled = (device == "cuda")

    # Model
    logger.info("Setting up XceptionNet for advanced fine-tuning...")
    model = timm.create_model('xception', pretrained=True, num_classes=1)

    # Freeze backbone layers initially
    for name, param in model.named_parameters():
        if "blocks" in name or "conv1" in name or "bn1" in name:
            param.requires_grad = False

    model.to(device)
    model = torch.compile(model)

    # Custom data transforms with augmentation
    data_transform = {
        'train': transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
    }

    train_dataset = datasets.ImageFolder(DATA_PATH / "train", transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(DATA_PATH / "val", transform=data_transform['val'])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    logger.info(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")

    # Optimizer & scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = SmoothBCEwLogits(smoothing=0.05)
    scaler = GradScaler(enabled=amp_enabled)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            with autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                with autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = (val_correct / len(val_dataset)) * 100

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Early stopping & Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # CRITICAL FIX: Save the original, uncompiled model's state dictionary
            torch.save(model._orig_mod.state_dict(), MODEL_SAVE_PATH / "best_detector_finetuned.pth")
            logger.info(f"✅ New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info(f"⚠️ Early stopping triggered at epoch {epoch+1}")
                break

        # Gradual unfreezing
        if epoch == int(EPOCHS * 0.2): # Unfreeze a bit earlier
            logger.info("🔓 Unfreezing backbone for full model fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            # Re-create optimizer to include newly unfrozen parameters
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE / 10, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch)


    logger.info("Fine-tuning complete! ✨")

if __name__ == "__main__":
    main()