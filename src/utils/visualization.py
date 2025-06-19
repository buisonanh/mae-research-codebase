import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial
from src.utils.masking import random_jigsaw_mask_keypoints, random_mask, random_jigsaw_mask, combined_keypoints_jigsaw_random_mask
from src.config import MASK_RATIO, NUM_KEYPOINTS
import os

def display_image_with_keypoints(image_path, keypoints):
    """Display an image with its keypoints overlaid."""
    # Load the original image for display
    image = plt.imread(image_path)
    if len(image.shape) == 3:  # If RGB, convert to grayscale
        image = np.mean(image, axis=-1)
    plt.imshow(image, cmap='gray')
    
    # Denormalize keypoints (scale back to image size)
    keypoints = keypoints.cpu().detach().numpy().reshape(-1, 2)
    keypoints[:, 0] *= image.shape[1]  # Scale x-coordinates
    keypoints[:, 1] *= image.shape[0]  # Scale y-coordinates

    # Plot the keypoints
    for (x, y) in keypoints:
        plt.scatter(x, y, s=10, c='red', marker='o')
    plt.show()


def plot_loss_curve(train_loss_values, val_loss_values, save_path=None):
    """Plot training and validation loss curves."""
    epochs = range(1, len(train_loss_values) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_values, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss_values, label='Validation Loss', color='orange')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def save_reconstruction_samples(model, model_keypoints, test_loader, device, save_path, patch_size, num_samples=7, masking_strategy="random-jigsaw"):
    """Save visualization of reconstruction samples.
    
    Args:
        model: The trained autoencoder model
        model_keypoints: The trained keypoint detection model (can be None if not using keypoints)
        test_loader: DataLoader for test set
        device: Device to run inference on
        save_path: Path to save the visualization
        patch_size: Patch size for masking
        num_samples: Number of samples to visualize (default: 7)
    """
    # Put models in eval mode
    model.eval()
    if model_keypoints is not None:
        model_keypoints.eval()

    # Get one batch from test set
    data_iter = iter(test_loader)
    data, _ = next(data_iter)  # Ignore labels since they might be keypoints
    
    # Get image names if available
    image_names = [name.split('/')[-1] for name, _ in test_loader.dataset.samples] if hasattr(test_loader.dataset, 'samples') else None
    
    data = data.to(device)

    with torch.no_grad():
        # Predict keypoints if needed for the strategy
        predicted_keypoints = None
        if masking_strategy in ["keypoints-jigsaw", "combined-keypoints-jigsaw-random-mask"]:
            if model_keypoints is None:
                raise ValueError("Keypoint model is required for this masking strategy but was not provided.")
            data_gray = data.mean(dim=1, keepdim=True)
            keypoints_flat = model_keypoints(data_gray)
            predicted_keypoints = keypoints_flat.view(-1, NUM_KEYPOINTS, 2)

        # Apply masking based on the strategy
        if masking_strategy == "keypoints-jigsaw":
            masked_img = random_jigsaw_mask_keypoints(
                data.clone(),
                keypoints=predicted_keypoints,
                patch_size=patch_size
            )
        elif masking_strategy == "combined-keypoints-jigsaw-random-mask":
            masked_img = combined_keypoints_jigsaw_random_mask(
                data.clone(),
                keypoints=predicted_keypoints,
                patch_size=patch_size,
                random_mask_ratio=MASK_RATIO
            )
        elif masking_strategy == "random":
            masked_img = random_mask(
                data.clone(),
                patch_size=patch_size,
                mask_ratio=MASK_RATIO
            )
        elif masking_strategy == "random-jigsaw":
            masked_img = random_jigsaw_mask(
                data.clone(),
                patch_size=patch_size,
                shuffle_ratio=MASK_RATIO
            )
        else:
            raise ValueError(f"Unknown masking strategy for visualization: {masking_strategy}")
            
        # Convert to 3 channels for model input if needed
        if masked_img.shape[1] == 1:
            masked_img = masked_img.repeat(1, 3, 1, 1)

        # 4) Generate reconstruction
        output = model(masked_img)

    # Move tensors to CPU for plotting
    data_cpu = data.cpu().numpy()
    masked_cpu = masked_img.cpu().numpy()
    recon_cpu = output.cpu().numpy()

    # Plot samples
    num_show = min(num_samples, data_cpu.shape[0])
    fig, ax = plt.subplots(3, num_show, figsize=(15, 6))

    for i in range(num_show):
        img_name = image_names[i] if image_names and i < len(image_names) else f"Image {i}"

        # Original image with keypoints
        orig_2d = data_cpu[i, 0, :, :]
        ax[0, i].imshow(orig_2d, cmap='gray')
        ax[0, i].set_title(f"Original\n{img_name}", fontsize=8)
        ax[0, i].axis('off')

        if model_keypoints is not None:
            # Plot keypoints
            kps = predicted_keypoints[i].cpu().numpy()
            H_, W_ = data[i].shape[1], data[i].shape[2]
            kps[:, 0] *= W_
            kps[:, 1] *= H_
            ax[0, i].scatter(kps[:, 0], kps[:, 1], s=10, c='red', marker='o')

        # Masked image
        masked_2d = masked_cpu[i, 0, :, :]
        ax[1, i].imshow(masked_2d, cmap='gray')
        ax[1, i].set_title(f"Masked\n{img_name}", fontsize=8)
        ax[1, i].axis('off')

        # Reconstructed image
        recon_3d = recon_cpu[i].transpose(1, 2, 0)
        ax[2, i].imshow(recon_3d)
        ax[2, i].set_title(f"Reconstructed\n{img_name}", fontsize=8)
        ax[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def format_config_params():
    """Format all configuration parameters as a string."""
    from src.config import (
        DEVICE, TARGET_SIZE, BATCH_SIZE, NUM_WORKERS,
        PRETRAIN_DATASET_NAME, CLASSIFY_DATASET_NAME, NUM_CLASSES,
        PATCH_SIZE, NUM_KEYPOINTS, KEYPOINT_NUM_EPOCHS,
        AUTOENCODER_NUM_EPOCHS, CLASSIFIER_NUM_EPOCHS,
        CLASSIFIER_LEARNING_RATE, LEARNING_RATE,
        EARLY_STOPPING_PATIENCE, MEAN, STD, DATASET_PATHS,
        SAVE_PATH
    )
    
    config_str = "=== Configuration Parameters ===\n\n"
    
    # Device and basic parameters
    config_str += "Basic Parameters:\n"
    config_str += f"Device: {DEVICE}\n"
    config_str += f"Target Size: {TARGET_SIZE}\n"
    config_str += f"Batch Size: {BATCH_SIZE}\n"
    config_str += f"Number of Workers: {NUM_WORKERS}\n\n"
    
    # Dataset parameters
    config_str += "Dataset Parameters:\n"
    config_str += f"Pretraining Dataset: {PRETRAIN_DATASET_NAME}\n"
    config_str += f"Classification Dataset: {CLASSIFY_DATASET_NAME}\n"
    config_str += "Number of Classes:\n"
    for dataset, num_classes in NUM_CLASSES.items():
        config_str += f"  - {dataset}: {num_classes}\n"
    config_str += "\nDataset Paths:\n"
    for dataset, path in DATASET_PATHS.items():
        config_str += f"  - {dataset}: {path}\n"
    config_str += "\n"
    
    # Model parameters
    config_str += "Model Parameters:\n"
    config_str += f"Patch Size: {PATCH_SIZE}\n"
    config_str += f"Number of Keypoints: {NUM_KEYPOINTS}\n\n"
    
    # Training parameters
    config_str += "Training Parameters:\n"
    config_str += f"Keypoint Training Epochs: {KEYPOINT_NUM_EPOCHS}\n"
    config_str += f"Autoencoder Training Epochs: {AUTOENCODER_NUM_EPOCHS}\n"
    config_str += f"Classifier Training Epochs: {CLASSIFIER_NUM_EPOCHS}\n"
    config_str += f"Classifier Learning Rate: {CLASSIFIER_LEARNING_RATE}\n"
    config_str += f"Base Learning Rate: {LEARNING_RATE}\n"
    config_str += f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n\n"
    
    # Normalization parameters
    config_str += "Image Normalization Parameters:\n"
    config_str += f"Mean: {MEAN}\n"
    config_str += f"Standard Deviation: {STD}\n\n"
    
    # Save paths
    config_str += "Save Paths:\n"
    config_str += f"Model Save Path: {SAVE_PATH}\n"
    
    return config_str


def save_training_results(train_loss_values, val_loss_values, config, save_path):
    """Save training results and configurations to a text file."""
    with open(save_path, 'w') as f:
        # Write configuration
        f.write("Training Configuration:\n")
        f.write("----------------------\n")
        f.write(config)
        
        # Write final metrics
        f.write("\nFinal Metrics:\n")
        f.write("--------------\n")
        f.write(f"Final Train Loss: {train_loss_values[-1]:.6f}\n")
        f.write(f"Final Validation Loss: {val_loss_values[-1]:.6f}\n")
        f.write(f"Best Validation Loss: {min(val_loss_values):.6f}\n")
        
        # Write loss history
        f.write("\nTraining Loss History:\n")
        f.write("---------------------\n")
        for epoch, loss in enumerate(train_loss_values):
            f.write(f"Epoch {epoch+1}: {loss:.6f}\n")
        
        f.write("\nValidation Loss History:\n")
        f.write("-----------------------\n")
        for epoch, loss in enumerate(val_loss_values):
            f.write(f"Epoch {epoch+1}: {loss:.6f}\n")