# scripts/train_master_v9.py
# This script tests our Phase 5 hypothesis: Stronger augmentations.
# 1. (NEW) Added TrivialAugmentWide and RandomErasing
# 2. (NEW) Replaced timm transforms with custom torchvision.transforms
# 3. (KEPT) All v8 features (1e-3 LR, scheduler, resume, grad clip)

import torch
import timm
import logging
import random
import numpy as np
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # (NEW) Import transforms
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

# --- Configuration & Hyperparameters ---
ROOT_DIR = Path("/home/rbk/deepsight-forensics") # Hardcoded
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
NUM_WORKERS = 16
SEED = 42
PATIENCE = 10
GRAD_CLIP_NORM = 1.0
DEVICE_INDEX = 0

# --- Paths ---
DATA_PATH = ROOT_DIR / "datasets/frames"
MODEL_SAVE_PATH = ROOT_DIR / "backend/models"
LOG_FILE_PATH = ROOT_DIR / "training_log_master_v9.txt" # New log file
LAST_CKPT_PATH = MODEL_SAVE_PATH / "last_checkpoint.pth"

# --- Set up Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
main_logger = logging.getLogger('main_training')

# --- Reproducibility Function ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main_logger.info(f"Setting global seed to {seed}")

def main(args):
    # --- 1. Setup ---
    set_seed(args.seed)
    device = f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu"
    
    amp_enabled = (device != "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    main_logger.info(f"--- DeepSight Master v9 Training Run ---")
    main_logger.info(f"HYPOTHESIS: Stronger augmentations (TrivialAugment, RandomErasing)")
    main_logger.info(f"Using device: {device}")
    main_logger.info(f"AMP enabled: {amp_enabled} | AMP dtype: {amp_dtype}")
    main_logger.info(f"Num workers: {args.num_workers} | Batch size: {args.batch_size}")
    main_logger.info(f"Learning rate: {args.lr}")

    # --- 2. Model & Config Setup ---
    main_logger.info("Setting up XceptionNet model...")
    model = timm.create_model('xception', pretrained=True, num_classes=1)
    model.to(device)
    model = torch.compile(model)

    # (NEW) Get model config, but build transforms manually
    data_config = timm.data.resolve_model_data_config(model)
    input_size = data_config['input_size'][-2:] # (299, 299)
    mean = data_config['mean']
    std = data_config['std']
    
    main_logger.info(f"Applying new augmentations: TrivialAugmentWide, RandomErasing")

    # (NEW) Custom training transform pipeline
    data_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
    ])
    
    # (NEW) Clean validation transform pipeline
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_f1 = BinaryF1Score().to(device)
    val_precision = BinaryPrecision().to(device)
    val_recall = BinaryRecall().to(device)

    # --- 3. Data Loading ---
    main_logger.info("Loading datasets...")
    train_dataset = datasets.ImageFolder(DATA_PATH / "train", transform=data_transform)
    val_dataset = datasets.ImageFolder(DATA_PATH / "val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    main_logger.info(f"Found {len(train_dataset)} train and {len(val_dataset)} val images.")

    # --- 4. Optimizer, Loss, Scaler, and Scheduler ---
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=amp_enabled)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- 5. Resume Support ---
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    if args.resume:
        if LAST_CKPT_PATH.exists():
            try:
                main_logger.info(f"Resuming from checkpoint: {LAST_CKPT_PATH}")
                checkpoint = torch.load(LAST_CKPT_PATH, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['best_val_loss']
                epochs_no_improve = checkpoint['epochs_no_improve']
                main_logger.info(f"Resumed successfully. Starting from epoch {start_epoch}.")
            except Exception as e:
                main_logger.warning(f"Could not load checkpoint: {e}. Starting from scratch.")
        else:
            main_logger.warning(f"Resume flag set, but no checkpoint found at {LAST_CKPT_PATH}. Starting from scratch.")
            
    main_logger.info("--- Starting training ---")

    # --- 6. The Training Loop ---
    checkpoint_data = {} 
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            # --- Training Phase ---
            model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                with autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            # --- Validation Phase ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_f1.reset(); val_precision.reset(); val_recall.reset()
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                    
                    with autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                    
                    val_loss += loss.item()
                    probs = torch.sigmoid(outputs)
                    preds = probs > 0.5
                    val_correct += (preds.float() == labels).sum().item()
                    
                    val_f1.update(probs, labels)
                    val_precision.update(probs, labels)
                    val_recall.update(probs, labels)

            # --- Calculate & Log Epoch Results ---
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = (val_correct / len(val_dataset)) * 100
            final_f1 = val_f1.compute().item()
            final_precision = val_precision.compute().item()
            final_recall = val_recall.compute().item()
            epoch_duration = time.time() - epoch_start_time

            log_message = (
                f"\nEpoch {epoch+1}/{args.epochs} | Duration: {epoch_duration:.1f}s | LR: {scheduler.get_last_lr()[0]:.1e}\n"
                f"  Train:     Loss: {avg_train_loss:.4f}\n"
                f"  Val:       Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2f}% | F1: {final_f1:.4f}, P: {final_precision:.4f}, R: {final_recall:.4f}"
            )
            main_logger.info(log_message)
            
            scheduler.step()

            # --- Checkpointing & Early Stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                save_path = MODEL_SAVE_PATH / f"best_model_loss_{best_val_loss:.4f}_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), save_path)
                main_logger.info(f"  ✅ New best model saved to {save_path.name}\n")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    main_logger.info(f"  ⚠️ Early stopping triggered at epoch {epoch+1}\n")
                    break
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve,
            }
            torch.save(checkpoint_data, LAST_CKPT_PATH)

    except KeyboardInterrupt:
        main_logger.info("\nTraining interrupted by user. Saving last completed epoch state...")
        if checkpoint_data:
            torch.save(checkpoint_data, LAST_CKPT_PATH)
            main_logger.info("Last state saved successfully.")
        else:
            main_logger.warning("No epoch data to save.")
            
    finally:
        pass

    main_logger.info(f"\nTraining complete! Best validation loss: {best_val_loss:.4f} ✨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSight Master Training Script v9 (Augmentations)")
    
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the last checkpoint')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Initial learning rate')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Total number of epochs to train')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Epochs for early stopping patience')
    parser.add_argument('--device-index', type=int, default=DEVICE_INDEX,
                        help='GPU device index to use')
    
    args = parser.parse_args()
    main(args)