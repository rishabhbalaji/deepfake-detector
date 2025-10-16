# scripts/train.py

import torch
import timm
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm

# --- Configuration & Hyperparameters ---
# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32  # Adjust this based on your GPU memory
EPOCHS = 50      # We can stop early if the model stops improving

# Data paths
DATA_PATH = ROOT_DIR / "datasets/frames"
MODEL_SAVE_PATH = ROOT_DIR / "models"
MODEL_SAVE_PATH.mkdir(exist_ok=True)

# --- Main Training Logic ---
def main():
    # Set the device (use GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Model & Config Setup ---
    # First, create the model to get its specific configuration
    print("Setting up the model...")
    model = timm.create_model('xception', pretrained=True, num_classes=1)
    model.to(device)

    # Now, resolve the data config FROM the model object itself
    data_config = timm.data.resolve_model_data_config(model)
    data_transform = timm.data.create_transform(**data_config, is_training=True)

    # --- 2. Data Loading ---
    print("Loading datasets...")
    # Create datasets using ImageFolder with the new transform
    train_dataset = datasets.ImageFolder(DATA_PATH / "train", transform=data_transform)
    val_dataset = datasets.ImageFolder(DATA_PATH / "val", transform=data_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # --- 3. Optimizer & Loss Function ---
    # Define Loss Function and Optimizer
    # BCEWithLogitsLoss is perfect for binary classification as it's numerically stable
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 4. The Training Loop ---
    best_val_loss = float('inf')
    print("Starting model training...")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train() # Set model to training mode
        train_loss, train_correct = 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_loss, val_correct = 0, 0
        
        with torch.no_grad(): # No need to calculate gradients during validation
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                
                # Calculate metrics
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()

        # --- Print Epoch Results ---
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = (train_correct / len(train_dataset)) * 100
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = (val_correct / len(val_dataset)) * 100
        
        print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")

        # --- Save the Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH / "best_detector.pth")
            print(f"✅ New best model saved with validation loss: {best_val_loss:.4f}")

    print("\nTraining complete! ✨ The best model is saved as 'best_detector.pth'.")


if __name__ == "__main__":
    main()