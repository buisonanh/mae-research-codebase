import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import FacialKeypointsDataset
from src.models.models_vit_mae import MaskedAutoencoderViT
import src.config as config


def load_and_preprocess_dataframe(csv_path):
    """Loads and preprocesses the facial keypoints dataframe."""
    print(f"Loading data from {csv_path}...")
    try:
        dataframe = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Dataset CSV file not found at {csv_path}")
        print("Please ensure the path is correct in src/config.py and the file exists.")
        return None

    # Drop rows with any missing keypoints, as they are not used for image data itself
    dataframe.dropna(inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    # Convert 'Image' column from space-separated string to NumPy array
    def string_to_array(img_str):
        pixels = np.array(img_str.split(), dtype=np.float32)
        side = int(np.sqrt(pixels.shape[0]))
        return pixels.reshape(side, side)

    if isinstance(dataframe['Image'].iloc[0], str):
        print("Converting 'Image' column from string to NumPy array...")
        dataframe['Image'] = dataframe['Image'].apply(string_to_array)
    
    # Normalize pixel values to [0, 1]
    dataframe['Image'] = dataframe['Image'] / 255.0
    print("Data loaded and preprocessed.")
    return dataframe

def denormalize_image(tensor, mean, std):
    """Denormalizes a tensor image."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def show_image(tensor_img, title="", mean=config.MEAN, std=config.STD, ax=None):
    """Displays a tensor image using matplotlib."""
    if tensor_img.is_cuda:
        tensor_img = tensor_img.cpu()
    
    img_to_show = denormalize_image(tensor_img.squeeze(0), mean, std) # Remove batch dim for single image
    
    if ax is None:
        plt.figure()
        plt.imshow(img_to_show.permute(1, 2, 0).numpy()) # C, H, W -> H, W, C
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(img_to_show.permute(1, 2, 0).numpy()) # C, H, W -> H, W, C
        ax.set_title(title)
        ax.axis('off')

def main():
    # --- 1. Load Data --- 
    dataframe = load_and_preprocess_dataframe(config.DATASET_PATHS["keypoints"])
    if dataframe is None:
        return

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # Model expects 3 channels
        transforms.Resize((config.TARGET_SIZE, config.TARGET_SIZE)), # Ensure consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    dataset = FacialKeypointsDataset(dataframe, transform=transform)
    # Use a batch size of 1 for visualization of a single image
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0) 

    # --- 2. Instantiate Model --- 
    # Ensure img_size in model matches TARGET_SIZE
    # The model's default in_chans=3 matches our Grayscale(3) transform
    model = MaskedAutoencoderViT(
        img_size=config.TARGET_SIZE,
        patch_size=config.PATCH_SIZE,
        embed_dim=768, # Example, adjust if needed or load from a config
        decoder_embed_dim=512, # Example
        norm_pix_loss=False
    ).to(config.DEVICE)
    model.eval() # Set model to evaluation mode

    # --- 3. Get a Sample --- 
    try:
        imgs_batch, keypoints_batch = next(iter(dataloader)) # Now also get keypoints
    except StopIteration:
        print("ERROR: DataLoader is empty. Check dataset path and content.")
        return
        
    img_original = imgs_batch[0:1].to(config.DEVICE) # Keep batch dim, select first image
    keypoints = keypoints_batch[0].reshape(-1, 2) # (NUM_KEYPOINTS, 2)

    # --- 4. Prepare Patches --- 
    # (N, L, P*P*C) - Pixel values for patches
    patches_pix = model.patchify(img_original) 
    # (N, L, D_embed) - Embeddings of patches
    x_embed = model.patch_embed(img_original) 
    # Add positional encoding (excluding CLS token position)
    x_embed_pos = x_embed + model.pos_embed[:, 1:, :]

    N, L, D_embed = x_embed_pos.shape
    _, _, PPC = patches_pix.shape # P*P*C

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # --- 5. Original Image --- 
    show_image(img_original.cpu(), title="Original Image", ax=axs[0])

    # --- 6. Random Masking Visualization --- 
    print(f"Applying Random Masking (mask_ratio={config.MASK_RATIO})...")
    _, mask_rm, _ = model.random_masking(x_embed_pos.clone(), config.MASK_RATIO)
    
    viz_patches_rm = patches_pix.clone()
    # Gray out masked patches (mask_rm == 1 means remove/mask)
    gray_val_for_patches = 0.5 # Mid-gray for normalized [0,1] pixel values
    for i in range(N): # Should be 1 for this script
        for l_idx in range(L):
            if mask_rm[i, l_idx] == 1:
                viz_patches_rm[i, l_idx, :] = gray_val_for_patches 
                
    img_viz_rm = model.unpatchify(viz_patches_rm)
    show_image(img_viz_rm.cpu(), title=f"Random Masking (Input to Encoder, MR={config.MASK_RATIO})", ax=axs[1])

    # --- 7. Random Jigsaw Masking Visualization --- 
    shuffle_ratio_viz = 0.5 # Define a shuffle ratio for visualization
    print(f"Applying Random Jigsaw Masking (shuffle_ratio={shuffle_ratio_viz})...")
    
    # Replicate jigsaw logic for pixel patches to visualize
    viz_patches_rjm = patches_pix.clone()
    len_shuffle = int(L * shuffle_ratio_viz)

    if len_shuffle > 1:
        for i in range(N): # Should be 1 for this script
            noise = torch.rand(L, device=patches_pix.device)
            ids_shuffle_subset = torch.argsort(noise)[:len_shuffle]
            new_order = torch.randperm(len_shuffle, device=patches_pix.device)
            
            # Apply the same shuffle to pixel patches
            shuffled_pixel_patches = viz_patches_rjm[i, ids_shuffle_subset, :][new_order]
            viz_patches_rjm[i, ids_shuffle_subset, :] = shuffled_pixel_patches
            
    img_viz_rjm = model.unpatchify(viz_patches_rjm)
    show_image(img_viz_rjm.cpu(), title=f"Random Jigsaw (Input to Encoder, SR={shuffle_ratio_viz})", ax=axs[2])

    # --- 8. Keypoints-Guided Jigsaw Masking Visualization ---
    print("Applying Keypoints-Guided Jigsaw Masking...")
    viz_patches_kpjm = patches_pix.clone()
    N, L, PPC = viz_patches_kpjm.shape
    grid_size = int(L ** 0.5)
    patch_size = config.PATCH_SIZE
    # Compute which patch each keypoint falls into
    keypoints_norm = keypoints.clone()
    if keypoints_norm.max() > 2:  # If not normalized, assume pixel coords
        keypoints_norm[:, 0] = keypoints_norm[:, 0] / config.TARGET_SIZE
        keypoints_norm[:, 1] = keypoints_norm[:, 1] / config.TARGET_SIZE
    keypoints_norm = keypoints_norm.clamp(0, 0.999)
    patch_indices_set = set()
    for kp in keypoints_norm:
        col = int(kp[0].item() * grid_size)
        row = int(kp[1].item() * grid_size)
        idx = row * grid_size + col
        patch_indices_set.add(idx)
    patch_indices = torch.tensor(list(patch_indices_set), dtype=torch.long, device=viz_patches_kpjm.device)
    if len(patch_indices) > 1:
        shuffled_order = torch.randperm(len(patch_indices), device=viz_patches_kpjm.device)
        shuffled_patches = viz_patches_kpjm[0, patch_indices, :][shuffled_order]
        viz_patches_kpjm[0, patch_indices, :] = shuffled_patches
    img_viz_kpjm = model.unpatchify(viz_patches_kpjm)
    show_image(img_viz_kpjm.cpu(), title="Keypoints-Guided Jigsaw", ax=axs[3])

    plt.tight_layout()
    fig.savefig("visualize_masking_results.png")
    plt.show()

if __name__ == '__main__':
    main()
