import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import json

from src.config import *
from src.models.autoencoder import Autoencoder
from src.data.dataset import create_pretrain_data_loaders
from src.utils.masking import partial_jigsaw_mask_keypoints, partial_jigsaw_mask, random_mask
from src.utils.visualization import plot_loss_curve, save_reconstruction_samples, save_training_results, format_config_params
from src.utils.train_keypoints import load_and_train_keypoints
    
def train_autoencoder(train_loader, val_loader, test_loader, model_keypoints, model, num_epochs, masking_strategy):
    """Train the autoencoder model."""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    train_loss_values = []
    val_loss_values = []
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(PRETRAIN_FOLDER, 'logs'), exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Training)', leave=False)
        for imgs, _ in train_progress:
            imgs = imgs.to(DEVICE)
            
            # Convert to grayscale for keypoints prediction
            imgs_gray = imgs.mean(dim=1, keepdim=True)
            
            if masking_strategy == "keypoints-jigsaw":
                # Predict keypoints
                with torch.no_grad():
                    keypoints_flat = model_keypoints(imgs_gray)
                    predicted_keypoints = keypoints_flat.view(-1, 15, 2)
                
                # Apply masking
                masked_imgs = partial_jigsaw_mask_keypoints(
                    imgs.clone(), 
                    predicted_keypoints, 
                    patch_size=PATCH_SIZE
                )
            elif masking_strategy == "random-jigsaw":
                masked_imgs = partial_jigsaw_mask(
                    imgs.clone(),
                    patch_size=PATCH_SIZE,
                    shuffle_ratio=MASK_RATIO
                )
            elif masking_strategy == "random":
                masked_imgs = random_mask(
                    imgs.clone(),
                    patch_size=PATCH_SIZE,
                    mask_ratio=MASK_RATIO
                )
            
            # Convert back to 3-channel format for the autoencoder
            if masked_imgs.shape[1] == 1:
                masked_imgs = masked_imgs.repeat(1, 3, 1, 1)
            
            # Forward pass
            outputs = model(masked_imgs)

            # Ensure target has same number of channels as output
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            
            loss = criterion(outputs, imgs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Update progress bar with current loss
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_loss_values.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Validation)', leave=False)
        with torch.no_grad():
            for imgs, _ in val_progress:
                imgs = imgs.to(DEVICE)
                
                # Convert to grayscale for keypoints prediction
                imgs_gray = imgs.mean(dim=1, keepdim=True)
                
                if masking_strategy == "keypoints-jigsaw":
                    # Predict keypoints
                    keypoints_flat = model_keypoints(imgs_gray)
                    predicted_keypoints = keypoints_flat.view(-1, 15, 2)
                    
                    # Apply masking
                    masked_imgs = partial_jigsaw_mask_keypoints(
                        imgs.clone(), 
                        predicted_keypoints, 
                        patch_size=PATCH_SIZE
                    )
                elif masking_strategy == "random-jigsaw":
                    masked_imgs = partial_jigsaw_mask(
                        imgs.clone(),
                        patch_size=PATCH_SIZE,
                        shuffle_ratio=MASK_RATIO
                    )
                elif masking_strategy == "random":
                    masked_imgs = random_mask(
                        imgs.clone(),
                        patch_size=PATCH_SIZE,
                        mask_ratio=MASK_RATIO
                    )
                
                if masked_imgs.shape[1] == 1:
                    masked_imgs = masked_imgs.repeat(1, 3, 1, 1)
                
                outputs = model(masked_imgs)
                
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                
                loss = criterion(outputs, imgs)
                val_loss += loss.item()
                
                # Update progress bar with current loss
                val_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)
                
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(PRETRAIN_FOLDER, 'mae_checkpoints', 'best_model.pth'))
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Best)')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save loss values to JSON file
        loss_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        # Save to JSON file
        results_file = os.path.join(PRETRAIN_FOLDER, 'logs', 'losses.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(loss_dict)
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=4)
    
    return train_loss_values, val_loss_values


def main():
    # Create save directories
    os.makedirs(os.path.join(PRETRAIN_FOLDER, 'mae_checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(PRETRAIN_FOLDER, 'plots'), exist_ok=True)

    print(f"Start pretraining with {MASKING_STRATEGY} strategy on {PRETRAIN_DATASET_NAME} dataset.")

    # Load data and train keypoint model if using keypoints-jigsaw
    if MASKING_STRATEGY == "keypoints-jigsaw":
        os.makedirs(os.path.join(PRETRAIN_FOLDER, 'keypoints_checkpoint'), exist_ok=True)
        print("Loading keypoints dataset and training keypoint model...")
        keypoint_model = load_and_train_keypoints()
    else:
        keypoint_model = None


    # For random masking strategies, we don't need keypoints data
    print("Loading image dataset for autoencoder...")
    train_loader, val_loader, test_loader = create_pretrain_data_loaders()

    # Train autoencoder
    print("Training autoencoder...")
    autoencoder = Autoencoder()
    train_loss_values, val_loss_values = train_autoencoder(
        train_loader, 
        val_loader, 
        test_loader, 
        keypoint_model, 
        autoencoder, 
        AUTOENCODER_NUM_EPOCHS, 
        MASKING_STRATEGY
    )
    
    # Save training results
    config = format_config_params()

    save_training_results(
        train_loss_values, 
        val_loss_values, 
        config, 
        os.path.join(PRETRAIN_FOLDER, 'training_results.txt')
    )
    
    # Plot loss curve
    plot_loss_curve(
        train_loss_values, 
        val_loss_values, 
        os.path.join(PRETRAIN_FOLDER, 'plots', 'autoencoder_training_loss.png')
    )

    # Save reconstruction samples
    save_reconstruction_samples(
        autoencoder,
        keypoint_model,
        test_loader,
        DEVICE,
        os.path.join(PRETRAIN_FOLDER, 'reconstruction_samples.png'),
        PATCH_SIZE
    )

if __name__ == "__main__":
    main()