import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

from src.config import (
    DEVICE, KEYPOINT_NUM_EPOCHS, AUTOENCODER_NUM_EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH, PATCH_SIZE, DATASET_PATHS
)
from src.models.keypoint_model import KeypointCNN
from src.models.autoencoder import Autoencoder
from src.data.dataset import load_keypoints_data, create_data_loaders, get_image_dataset
from src.utils.masking import partial_jigsaw_mask_keypoints
from src.utils.visualization import plot_training_results, plot_loss_curve, save_reconstruction_samples, format_config_params

def train_keypoint_model(train_loader, val_loader, model, num_epochs):
    """Train the keypoint detection model."""
    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create checkpoint directory
    keypoints_checkpoint_dir = os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'keypoints_checkpoint')
    os.makedirs(keypoints_checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for images, keypoints in train_loader:
            images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, keypoints in val_loader:
                images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, keypoints)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(keypoints_checkpoint_dir, 'best_model_keypoints.pth'))
            print("Saved best model!")
    
    return train_losses, val_losses

def train_autoencoder(train_loader, test_loader, model_keypoints, model, num_epochs):
    """Train the autoencoder model."""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    stagnant_epochs = 0
    loss_values = []
    
    # Create checkpoint and plot directories
    mae_checkpoint_dir = os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'mae_checkpoints')
    plots_dir = os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'plots')
    os.makedirs(mae_checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Running epoch {epoch}...")
        epoch_loss = 0.0
        
        for batch_idx, (imgs, _) in enumerate(tqdm(train_loader)):
            imgs = imgs.to(DEVICE)
            imgs_gray = imgs.mean(dim=1, keepdim=True)
            imgs_3c = imgs.repeat(1, 3, 1, 1)

            # Predict keypoints
            with torch.no_grad():
                keypoints_flat = model_keypoints(imgs_gray)
                predicted_keypoints = keypoints_flat.view(-1, 15, 2)

            # Apply masking
            masked_imgs = partial_jigsaw_mask_keypoints(
                imgs.clone(), 
                keypoints=predicted_keypoints, 
                patch_size=PATCH_SIZE
            )

            optimizer.zero_grad()
            masked_imgs_3c = masked_imgs.repeat(1, 3, 1, 1)
            output = model(masked_imgs_3c)
            loss = criterion(output, imgs.repeat(1, 3, 1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Visualize first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                plot_training_results(
                    imgs, masked_imgs, output,
                    keypoints=predicted_keypoints,
                    save_path=os.path.join(plots_dir, 'first_batch_visualization.png')
                )

        avg_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_loss)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            stagnant_epochs = 0
            torch.save(model.state_dict(), 
                      os.path.join(mae_checkpoint_dir, f'mae_autoencoder_epoch{epoch}.pth'))
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
                break

    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(mae_checkpoint_dir, 'mae_autoencoder_final.pth'))
    
    # Save final loss and configuration
    config_str = format_config_params()
    metrics_file = os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'final_pretrain_loss.txt')
    with open(metrics_file, 'w') as f:
        f.write(config_str)
        f.write("\n=== Final Training Metrics ===\n\n")
        f.write(f"Final Loss: {avg_loss:.4f}\n")
    
    # Save loss curve
    plot_loss_curve(loss_values, os.path.join(plots_dir, 'mae_training_loss.png'))
    
    # Save reconstruction samples from test set
    save_reconstruction_samples(
        model=model,
        model_keypoints=model_keypoints,
        test_loader=train_loader,
        device=DEVICE,
        save_path=os.path.join(plots_dir, 'test_set_reconstructions.png'),
        patch_size=PATCH_SIZE
    )
    
    return loss_values

def main():
    # Create save directories
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'keypoints_checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'mae_checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'plots'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'classification_checkpoints'), exist_ok=True)
    
    # Load and split keypoints data
    df = load_keypoints_data(DATASET_PATHS["keypoints"])
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_df, val_df, test_df)
    
    # Train keypoint model
    print("Training keypoint model...")
    keypoint_model = KeypointCNN().to(DEVICE)
    train_losses, val_losses = train_keypoint_model(
        train_loader, val_loader, keypoint_model, KEYPOINT_NUM_EPOCHS
    )
    plot_loss_curve(
        train_losses, 
        os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'plots', 'keypoint_training_loss.png')
    )
    
    # Load best keypoint model for autoencoder training
    keypoint_model.load_state_dict(
        torch.load(os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'keypoints_checkpoint', 'best_model_keypoints.pth'))
    )
    keypoint_model.eval()
    
    # Train autoencoder
    print("Training autoencoder...")
    autoencoder = Autoencoder()
    image_loader = get_image_dataset()
    loss_values = train_autoencoder(image_loader, test_loader, keypoint_model, autoencoder, AUTOENCODER_NUM_EPOCHS)

if __name__ == "__main__":
    main() 