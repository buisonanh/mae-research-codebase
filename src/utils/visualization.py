import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial
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

# --- Masking functions are methods on MaskedAutoencoderViT ---
# Masking is performed via model.patchify, model.random_masking, model.random_jigsaw_masking, model.keypoint_jigsaw_masking, etc.


def save_reconstruction_samples(model, model_keypoints, test_loader, device, save_path, num_samples=7, masking_strategy="random-jigsaw"):
    """
    Save visualization of reconstruction samples by using the model's forward pass.

    Args:
        model: The trained autoencoder model (MaskedAutoencoderViT).
        model_keypoints: The trained keypoint detection model (can be None).
        test_loader: DataLoader for the test set.
        device: Device to run inference on.
        save_path: Path to save the visualization.
        num_samples: Number of samples to visualize.
        masking_strategy: The masking strategy used during training.
    """
    model.eval()
    if model_keypoints is not None:
        model_keypoints.eval()

    # Get one batch from the test set
    try:
        batch = next(iter(test_loader))
    except StopIteration:
        raise RuntimeError("Test loader is empty!")

    # Unpack data and move to device
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        data, keypoints_or_labels = batch
    else:
        data = batch
        keypoints_or_labels = None
    data = data.to(device)

    # Get keypoints if needed for the masking strategy
    predicted_keypoints = None
    if masking_strategy == "keypoints-jigsaw":
        if model_keypoints is not None:
            predicted_keypoints = model_keypoints(data).view(-1, NUM_KEYPOINTS, 2)
        elif keypoints_or_labels is not None and keypoints_or_labels.shape[-1] == NUM_KEYPOINTS * 2:
            predicted_keypoints = keypoints_or_labels.view(-1, NUM_KEYPOINTS, 2).to(device)
        else:
            raise ValueError("Keypoints required for keypoint-jigsaw strategy but none provided.")

    # --- Model Forward Pass ---
    with torch.no_grad():
        # The model's forward pass returns loss, prediction, and mask
        loss, pred, mask = model(data, mask_ratio=MASK_RATIO, keypoints=predicted_keypoints)

    # --- Visualization ---
    recon_img = model.unpatchify(pred)

    # Create masked/shuffled image for visualization
    if masking_strategy in ["keypoints-jigsaw", "random-jigsaw"]:
        # For jigsaw, the input is shuffled. We create a representative shuffled image for visualization.
        # Note: This shuffle is random and may not be the exact one used in the forward pass.
        patches = model.patchify(data)
        if masking_strategy == "keypoints-jigsaw":
            shuffled_patches, _, _ = model.keypoint_jigsaw_masking(patches, predicted_keypoints)
        else:  # random-jigsaw
            shuffled_patches, _, _ = model.random_jigsaw_masking(patches, MASK_RATIO)
        masked_img = model.unpatchify(shuffled_patches)
        masked_title = "Shuffled Image"
    else:  # Default to random-masking visualization
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
        masked_patches = model.patchify(data) * (1 - mask)  # Apply mask
        masked_img = model.unpatchify(masked_patches)
        masked_title = "Masked Image"

    # Move tensors to CPU for plotting
    data_cpu = data.cpu()
    recon_cpu = recon_img.cpu()
    masked_cpu = masked_img.cpu()

    # Plotting
    num_show = min(num_samples, data.shape[0])
    fig, ax = plt.subplots(3, num_show, figsize=(15, 6))

    for i in range(num_show):
        # Original Image
        orig_display = data_cpu[i].permute(1, 2, 0).numpy()
        ax[0, i].imshow(orig_display)
        ax[0, i].set_title(f"Original Image {i+1}")
        ax[0, i].axis('off')

        # Masked/Shuffled Image
        masked_display = masked_cpu[i].permute(1, 2, 0).numpy()
        ax[1, i].imshow(masked_display)
        ax[1, i].set_title(f"{masked_title} {i+1}")
        ax[1, i].axis('off')

        # Reconstructed Image
        recon_display = recon_cpu[i].permute(1, 2, 0).numpy()
        ax[2, i].imshow(recon_display)
        ax[2, i].set_title(f"Reconstructed Image {i+1}")
        ax[2, i].axis('off')

    plt.tight_layout()
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Reconstruction samples saved to {save_path}")

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

if __name__ == '__main__':
    # Example usage: Print the formatted configuration parameters
    config_string = format_config_params()
    print(config_string)

    # To test other functions, you would need to provide appropriate data,
    # for example, dummy loss values for plot_loss_curve:
    #
    # train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    # val_losses = [0.6, 0.5, 0.4, 0.3, 0.2]
    # plot_loss_curve(train_losses, val_losses, save_path="sample_loss_curve.png")
    # print("Sample loss curve saved to sample_loss_curve.png")