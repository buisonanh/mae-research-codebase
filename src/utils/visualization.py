import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial
from src.utils.masking import partial_jigsaw_mask_keypoints

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

def plot_training_results(original_images, masked_images, reconstructed_images, 
                         keypoints=None, image_names=None, labels=None, num_images=7,
                         save_path=None):
    """Plot original, masked, and reconstructed images in a grid.
    
    Args:
        original_images: Tensor of original images
        masked_images: Tensor of masked images
        reconstructed_images: Tensor of reconstructed images
        keypoints: Optional tensor of keypoints to plot
        image_names: Optional list of image names
        labels: Optional tensor of labels
        num_images: Number of images to show (default: 7)
        save_path: Optional path to save the plot
    """
    num_show = min(num_images, original_images.shape[0])
    fig, ax = plt.subplots(3, num_show, figsize=(15, 6))

    for i in range(num_show):
        img_name = image_names[i] if image_names is not None else f"Image {i}"
        img_class = labels[i].item() if labels is not None else "Unknown"

        # Original images
        orig_img = original_images[i].detach().cpu().numpy().transpose(1, 2, 0)
        ax[0, i].imshow(orig_img, cmap='gray')
        ax[0, i].set_title(f"Original\n{img_name}\nClass: {img_class}", fontsize=8)
        ax[0, i].axis('off')

        # Plot keypoints if provided
        if keypoints is not None:
            kps = keypoints[i].detach().cpu().numpy()
            H_, W_ = original_images[i].shape[1:3]
            kps[:, 0] *= W_
            kps[:, 1] *= H_
            ax[0, i].scatter(kps[:, 0], kps[:, 1], s=10, c='red', marker='o')

        # Masked images
        masked_img = masked_images[i].detach().cpu().numpy().transpose(1, 2, 0)
        ax[1, i].imshow(masked_img, cmap='gray')
        ax[1, i].set_title(f"Masked\n{img_name}", fontsize=8)
        ax[1, i].axis('off')

        # Reconstructed images
        recon_img = reconstructed_images[i].detach().cpu().numpy().transpose(1, 2, 0)
        ax[2, i].imshow(recon_img)
        ax[2, i].set_title(f"Reconstructed\n{img_name}", fontsize=8)
        ax[2, i].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    return fig

def plot_loss_curve(loss_values, save_path=None):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_values) + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def save_reconstruction_samples(model, model_keypoints, test_loader, device, save_path, patch_size, num_samples=7):
    """Save visualization of original-masked-reconstructed samples from test set.
    
    Args:
        model: The trained autoencoder model
        model_keypoints: The trained keypoint detection model
        test_loader: DataLoader for test set
        device: Device to run inference on
        save_path: Path to save the visualization
        patch_size: Patch size for masking
        num_samples: Number of samples to visualize (default: 7)
    """
    # Put models in eval mode
    model.eval()
    model_keypoints.eval()

    # Get one batch from test set
    data_iter = iter(test_loader)
    data, _ = next(data_iter)  # Ignore labels since they might be keypoints
    
    # Get image names if available
    image_names = [name.split('/')[-1] for name, _ in test_loader.dataset.samples] if hasattr(test_loader.dataset, 'samples') else None
    
    data = data.to(device)

    with torch.no_grad():
        # 1) Generate keypoints
        data_gray = data.mean(dim=1, keepdim=True)
        keypoints_flat = model_keypoints(data_gray)
        num_keypoints = 15
        predicted_keypoints = keypoints_flat.view(-1, num_keypoints, 2)

        # 2) Apply masking
        masked_img = partial_jigsaw_mask_keypoints(
            data.clone(),
            keypoints=predicted_keypoints,
            patch_size=patch_size
        )
        
        # 3) Convert to 3 channels for model input
        masked_img_3c = masked_img.repeat(1, 3, 1, 1)

        # 4) Generate reconstruction
        output = model(masked_img_3c)

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
        MODEL_SAVE_PATH
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
    config_str += f"Model Save Path: {MODEL_SAVE_PATH}\n"
    
    return config_str 