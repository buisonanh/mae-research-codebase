import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import datetime
import json
from tqdm import tqdm
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from src.config import (
    DEVICE, LEARNING_RATE, AUTOENCODER_NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH, DATASET_PATHS, MEAN, STD, BATCH_SIZE, NUM_WORKERS,
    PRETRAIN_DATASET_NAME, MULTI_GPU, GPU_IDS
)
from src.models.vit_autoencoder import MaskedAutoencoder
from src.utils.visualization import plot_loss_curve, format_config_params

def get_data_transforms():
    """Get data transforms for training and evaluation."""
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),  # ViT requires 224x224 input
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),  # ViT requires 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    }

def create_data_loaders():
    """Create data loaders for training and validation."""
    data_transforms = get_data_transforms()
    
    # Load dataset
    dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_PATHS[PRETRAIN_DATASET_NAME]),
        transform=data_transforms['train']
    )
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    return train_loader, val_loader


def train_vit_autoencoder(train_loader, val_loader, model, num_epochs):
    """Train the ViT autoencoder model."""
    # Check if we should use multiple GPUs
    if MULTI_GPU and torch.cuda.is_available() and len(GPU_IDS) > 1:
        print(f"Using {len(GPU_IDS)} GPUs: {GPU_IDS}")
        model = torch.nn.DataParallel(model, device_ids=GPU_IDS)
    
    model.to(DEVICE)
    # Use MSE loss for token embedding reconstruction
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_loss = float('inf')
    stagnant_epochs = 0
    train_loss_values = []
    val_loss_values = []
    
    # Create checkpoint and plot directories
    mae_checkpoint_dir = os.path.join(MODEL_SAVE_PATH, 'vit_pretrain_checkpoints', 'vit_mae_checkpoints')
    plots_dir = os.path.join(MODEL_SAVE_PATH, 'vit_pretrain_checkpoints', 'plots')
    os.makedirs(mae_checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Running epoch {epoch+1}/{num_epochs}...")
        
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        
        for batch_idx, (imgs, _) in enumerate(tqdm(train_loader, desc="Training")):
            imgs = imgs.to(DEVICE)
            
            # Convert grayscale to RGB if needed
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
                
            # Forward pass
            reconstructed_embeddings, original_embeddings, mask = model(imgs)
            
            # Compute loss only on masked tokens
            loss = 0
            for i in range(imgs.shape[0]):
                if mask[i].sum() > 0:  # Ensure there are masked tokens
                    loss += criterion(
                        reconstructed_embeddings[i, mask[i]], 
                        original_embeddings[i, mask[i]]
                    )
            
            # Average loss over batch
            loss = loss / imgs.shape[0]
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_epoch_loss / len(train_loader)
        train_loss_values.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        
        with torch.no_grad():
            for imgs, _ in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(DEVICE)
                
                # Convert grayscale to RGB if needed
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                
                # Forward pass
                reconstructed_embeddings, original_embeddings, mask = model(imgs)
                
                # Compute loss only on masked tokens
                loss = 0
                for i in range(imgs.shape[0]):
                    if mask[i].sum() > 0:  # Ensure there are masked tokens
                        loss += criterion(
                            reconstructed_embeddings[i, mask[i]], 
                            original_embeddings[i, mask[i]]
                        )
                
                # Average loss over batch
                loss = loss / imgs.shape[0]
                val_epoch_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_loss_values.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping and model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            stagnant_epochs = 0
            torch.save(model.autoencoder.state_dict(), 
                      os.path.join(mae_checkpoint_dir, f'vit_mae_autoencoder_epoch{epoch+1}.pth'))
            print("Saved best model!")
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                break

    # Save final model
    torch.save(model.autoencoder.state_dict(), 
              os.path.join(mae_checkpoint_dir, 'vit_mae_autoencoder_final.pth'))
    
    # Save final loss and configuration
    config_str = format_config_params()
    metrics_file = os.path.join(MODEL_SAVE_PATH, 'vit_pretrain_checkpoints', 'final_vit_pretrain_loss.txt')
    with open(metrics_file, 'w') as f:
        f.write(config_str)
        f.write("\n=== Final Training Metrics ===\n\n")
        f.write(f"Final Train Loss: {train_loss_values[-1]:.4f}\n")
        f.write(f"Final Val Loss: {val_loss_values[-1]:.4f}\n")
        f.write(f"Best Val Loss: {best_loss:.4f}\n")
    
    # Save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_values, label='Train Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'vit_mae_training_loss.png'))
    plt.close()
    
    return train_loss_values, val_loss_values, best_loss

def main():
    # Create save directories
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'vit_pretrain_checkpoints', 'vit_mae_checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'vit_pretrain_checkpoints', 'plots'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'vit_classification_checkpoints'), exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders()
    
    # Create masked autoencoder model
    print("Creating Masked Autoencoder model...")
    masked_autoencoder = MaskedAutoencoder()
    
    # Train ViT autoencoder
    print("Training ViT Masked Autoencoder...")
    train_losses, val_losses, best_loss = train_vit_autoencoder(
        train_loader, val_loader, masked_autoencoder, AUTOENCODER_NUM_EPOCHS
    )
    
    print(f"Training complete! Best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
