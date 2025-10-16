# scripts/train.py

import torch
import timm
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn, optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import time

# --- Configuration & Hyperparameters ---
ROOT_DIR = Path(__file__).resolve().parents[1]
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
NUM_WORKERS = 12

# Paths
DATA_PATH = ROOT_DIR / "datasets/frames"
MODEL_SAVE_PATH = ROOT_DIR / "backend/models"
MODEL_SAVE_PATH.mkdir(exist_ok=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    amp_enabled = (device == "cuda")
    
    # --- Model ---
    print("Setting up the model...")
    model = timm.create_model('xception', pretrained=True, num_classes=1)
    model.to(device)
    model = torch.compile(model)
    
    data_config = timm.data.resolve_model_data_config(model)
    data_transform = timm.data.create_transform(**data_config, is_training=True)
    
    # --- Data ---
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(DATA_PATH / "train", transform=data_transform)
    val_dataset = datasets.ImageFolder(DATA_PATH / "val", transform=data_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # --- Optimizer & Loss ---
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=amp_enabled)
    
    best_val_loss = float('inf')
    print("Starting optimized training...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            with autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()
        
        # --- Validation ---
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
        avg_train_acc = (train_correct / len(train_dataset)) * 100
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = (val_correct / len(val_dataset)) * 100
        
        print(f"\nEpoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Acc {avg_train_acc:.2f}% | "
              f"Val Loss {avg_val_loss:.4f}, Acc {avg_val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model._orig_mod.state_dict(), MODEL_SAVE_PATH / "best_detector.pth")
            print(f"✅ New best model saved (val loss: {best_val_loss:.4f})")
    
    print("\nTraining complete! ✨")

if __name__ == "__main__":
    main()