import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json

from src.config import *
from src.models.models_vit_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
from src.data.dataset import create_pretrain_data_loaders
from src.utils.train_keypoints import load_and_train_keypoints
from src.utils.visualization import plot_loss_curve, save_reconstruction_samples, save_training_results, format_config_params

def train_mae(train_loader, val_loader, test_loader, keypoint_model, model, num_epochs, masking_strategy):
    print(f"Using masking strategy: {masking_strategy}")
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Using optimizer: {optimizer.__class__.__name__}")

    best_val_loss = float('inf')
    train_loss_values = []
    val_loss_values = []

    os.makedirs(os.path.join(PRETRAIN_FOLDER, 'logs'), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Training)', leave=False)
        for imgs, _ in train_progress:
            imgs = imgs.to(DEVICE)
            imgs_gray = imgs.mean(dim=1, keepdim=True)
            if imgs_gray.shape[1] == 1:
                imgs_gray = imgs_gray.repeat(1, 3, 1, 1)

            keypoints = None
            if masking_strategy == "keypoints-jigsaw":
                with torch.no_grad():
                    keypoints_flat = keypoint_model(imgs_gray)
                    keypoints = keypoints_flat.view(imgs.shape[0], NUM_KEYPOINTS, 2)
            elif masking_strategy == "combined-keypoints-jigsaw-random-mask":
                with torch.no_grad():
                    keypoints_flat = keypoint_model(imgs_gray)
                    keypoints = keypoints_flat.view(imgs.shape[0], NUM_KEYPOINTS, 2)

            # Forward pass with masking strategy
            if masking_strategy == "random-masking":
                loss, _, _ = model(imgs, mask_ratio=MASK_RATIO)
            elif masking_strategy == "random-jigsaw":
                loss, _, _ = model(imgs, mask_ratio=0.0, shuffle_ratio=MASK_RATIO)  # No masking, only shuffle
            elif masking_strategy == "keypoints-jigsaw":
                loss, _, _ = model(imgs, mask_ratio=0.0, shuffle_ratio=0.0, keypoints=keypoints)
            elif masking_strategy == "combined-keypoints-jigsaw-random-mask":
                loss, _, _ = model(imgs, mask_ratio=MASK_RATIO, shuffle_ratio=0.0, keypoints=keypoints)
            else:
                raise ValueError(f"Unknown masking strategy: {masking_strategy}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
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
                imgs_gray = imgs.mean(dim=1, keepdim=True)
                if imgs_gray.shape[1] == 1:
                    imgs_gray = imgs_gray.repeat(1, 3, 1, 1)

                keypoints = None
                if masking_strategy == "keypoints-jigsaw":
                    keypoints_flat = keypoint_model(imgs_gray)
                    keypoints = keypoints_flat.view(imgs.shape[0], NUM_KEYPOINTS, 2)
                elif masking_strategy == "combined-keypoints-jigsaw-random-mask":
                    keypoints_flat = keypoint_model(imgs_gray)
                    keypoints = keypoints_flat.view(imgs.shape[0], NUM_KEYPOINTS, 2)

                if masking_strategy == "random-masking":
                    loss, _, _ = model(imgs, mask_ratio=MASK_RATIO)
                elif masking_strategy == "random-jigsaw":
                    loss, _, _ = model(imgs, mask_ratio=0.0, shuffle_ratio=MASK_RATIO)
                elif masking_strategy == "keypoints-jigsaw":
                    loss, _, _ = model(imgs, mask_ratio=0.0, shuffle_ratio=0.0, keypoints=keypoints)
                elif masking_strategy == "combined-keypoints-jigsaw-random-mask":
                    loss, _, _ = model(imgs, mask_ratio=MASK_RATIO, shuffle_ratio=0.0, keypoints=keypoints)

                val_loss += loss.item()
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
        loss_dict = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss}
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
    os.makedirs(os.path.join(PRETRAIN_FOLDER, 'mae_checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(PRETRAIN_FOLDER, 'plots'), exist_ok=True)
    print(f"Start pretraining with {MASKING_STRATEGY} strategy on {PRETRAIN_DATASET_NAME} dataset.")

    # Load data and train keypoint model if needed
    if MASKING_STRATEGY in ["keypoints-jigsaw", "combined-keypoints-jigsaw-random-mask"]:
        os.makedirs(os.path.join(PRETRAIN_FOLDER, 'keypoints_checkpoint'), exist_ok=True)
        print("Loading keypoints dataset and training keypoint model...")
        keypoint_model = load_and_train_keypoints()
    else:
        keypoint_model = None

    print(f"Masking ratio: {MASK_RATIO}")
    print("Loading image dataset for MAE...")
    train_loader, val_loader, test_loader = create_pretrain_data_loaders()

    # Instantiate MAE model
    if ENCODER_MODEL == "vit_base_p16":
        model = mae_vit_base_patch16()
    elif ENCODER_MODEL == "vit_large_p16":
        model = mae_vit_large_patch16()
    elif ENCODER_MODEL == "vit_huge_p14":
        model = mae_vit_huge_patch14()
    else:
        raise ValueError(f"Unknown encoder model: {ENCODER_MODEL}")

    print("Training MAE...")
    train_loss_values, val_loss_values = train_mae(
        train_loader,
        val_loader,
        test_loader,
        keypoint_model,
        model,
        AUTOENCODER_NUM_EPOCHS,
        MASKING_STRATEGY
    )

    config = format_config_params()
    save_training_results(
        train_loss_values,
        val_loss_values,
        config,
        os.path.join(PRETRAIN_FOLDER, 'training_results.txt')
    )

    plot_loss_curve(
        train_loss_values,
        val_loss_values,
        os.path.join(PRETRAIN_FOLDER, 'plots', 'mae_training_loss.png')
    )

    save_reconstruction_samples(
        model,
        keypoint_model,
        test_loader,
        DEVICE,
        os.path.join(PRETRAIN_FOLDER, 'reconstruction_samples.png'),
        num_samples=7,
        masking_strategy=MASKING_STRATEGY
    )

    # Print the path to the best model for the runner script
    checkpoint_path = os.path.join(PRETRAIN_FOLDER, 'mae_checkpoints', 'best_model.pth')
    print(f"Best model saved at: {checkpoint_path}")

if __name__ == "__main__":
    main()
