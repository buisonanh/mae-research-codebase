import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

from src.config import *
from src.models.keypoint_model import KeypointCNN
from src.data.dataset import load_keypoints_data, create_keypoints_data_loaders
from src.utils.visualization import plot_loss_curve

def train_keypoint_model(train_loader, val_loader, model, num_epochs):
    """Train the keypoint detection model."""
    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create checkpoint directory
    keypoints_checkpoint_dir = os.path.join(SAVE_PATH, 'pretrain_checkpoints', 'keypoints_checkpoint')
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
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(keypoints_checkpoint_dir, 'best_model_keypoints.pth'))
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Best)')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def load_and_train_keypoints():
    """Load keypoints data and train the keypoint model."""
    print("Loading keypoints dataset...")
    df = load_keypoints_data(DATASET_PATHS["keypoints"])
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_keypoints_data_loaders(train_df, val_df, test_df)
    
    # Train keypoint model
    print("Training keypoint model...")
    keypoint_model = KeypointCNN().to(DEVICE)
    train_losses, val_losses = train_keypoint_model(
        train_loader, val_loader, keypoint_model, KEYPOINT_NUM_EPOCHS
    )
    
    # Plot loss curve
    plot_loss_curve(
        train_losses, 
        val_losses,
        os.path.join(SAVE_PATH, 'pretrain_checkpoints', 'plots', 'keypoint_training_loss.png')
    )
    
    # Load best keypoint model
    keypoint_model.load_state_dict(
        torch.load(os.path.join(SAVE_PATH, 'pretrain_checkpoints', 'keypoints_checkpoint', 'best_model_keypoints.pth'))
    )
    keypoint_model.eval()
    
    return keypoint_model
