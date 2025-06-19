import torch
import numpy as np # Retaining for other functions, though not used in the new one directly

from torch import nn
from src.config import PATCH_SIZE # Used as default in random_jigsaw_mask_keypoints


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x -> B c h w
        bs, c, h, w = x.shape
        
        x = self.unfold(x)
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        a = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        return a


def random_jigsaw_mask_keypoints(image, keypoints, patch_size=PATCH_SIZE):
    # If keypoints is 2D, add a batch dimension
    if keypoints.ndim == 2:
        keypoints = keypoints.unsqueeze(0)

    # Clone keypoints before modification to avoid side effects
    keypoints_proc = keypoints.clone()
    # Clamp to avoid exactly 1.0, which can cause out-of-bounds for patch calculation
    keypoints_proc = keypoints_proc.clamp(min=0.0, max=1.0 - 1e-6) 
        
    B, C, H, W = image.shape
    
    # Multiply normalized coords by width & height
    keypoints_proc[..., 0] = keypoints_proc[..., 0] * W  # x-coords
    keypoints_proc[..., 1] = keypoints_proc[..., 1] * H  # y-coords
    
    if not (H % patch_size == 0 and W % patch_size == 0):
        raise ValueError(f"Image height {H} and width {W} must be divisible by patch_size {patch_size}")
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w

    # Create patches view
    # unfold: (B, C, H, W) -> (B, C*patch_size*patch_size, num_patches_total)
    # permute & view: (B, num_patches_total, C, patch_size, patch_size)
    patches = (
        image.unfold(2, patch_size, patch_size) # Patches along H
             .unfold(3, patch_size, patch_size) # Patches along W
             .permute(0, 2, 3, 1, 4, 5) # B, num_patches_h, num_patches_w, C, patch_size, patch_size
             .contiguous()
             .view(B, total_patches, C, patch_size, patch_size)
    )
    
    for b in range(B):
        patch_indices_set = set()
        # Use keypoints_proc for this batch item
        current_keypoints = keypoints_proc[b] 
        for kp in current_keypoints:
            x, y = kp[0].item(), kp[1].item()
            # Ensure col/row are within valid range
            col = min(int(x // patch_size), num_patches_w - 1)
            row = min(int(y // patch_size), num_patches_h - 1)
            idx = row * num_patches_w + col
            patch_indices_set.add(idx)

        if patch_indices_set:
            patch_indices_tensor = torch.tensor(list(patch_indices_set), device=image.device, dtype=torch.long)
            if patch_indices_tensor.numel() > 0: # Ensure there are patches to shuffle
                shuffled_indices = patch_indices_tensor[torch.randperm(len(patch_indices_tensor))]
                # This modifies `patches` in place for the current batch item
                patches[b, patch_indices_tensor] = patches[b, shuffled_indices]
    
    # Reconstruct the image from (potentially) shuffled patches
    patches_reconstructed = patches.view(B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches_reconstructed = patches_reconstructed.permute(0, 3, 1, 4, 2, 5).contiguous() # B, C, num_patches_h, patch_size, num_patches_w, patch_size
    shuffled_image = patches_reconstructed.view(B, C, H, W)
    return shuffled_image


def random_jigsaw_mask(image, patch_size=32, shuffle_ratio=0.4):
    """
    Partially shuffle patches in a batch of images.
    Args:
        image (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (int): Size of each patch.
        shuffle_ratio (float): Fraction of patches to shuffle (0 <= shuffle_ratio <= 1).
    Returns:
        torch.Tensor: Tensor with partially shuffled patches, same shape as input.
    """
    B, C, H, W = image.shape
    if not (H % patch_size == 0 and W % patch_size == 0):
        raise ValueError(f"Image height {H} and width {W} must be divisible by patch_size {patch_size}")

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w

    # Reshape the image into patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # Shape: (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches = patches.view(B, -1, C, patch_size, patch_size)  # Shape: (B, num_patches, C, patch_size, patch_size)

    # Shuffle a subset of patches
    for b in range(B):
        if total_patches == 0: continue # Avoid error if image is smaller than patch size

        num_shuffled = int(total_patches * shuffle_ratio)
        if num_shuffled == 0:
            continue

        patch_indices_all = torch.arange(total_patches, device=image.device)
        # Indices of patches that will be part of the shuffle
        shuffle_indices_perm = patch_indices_all[torch.randperm(total_patches, device=image.device)[:num_shuffled]]
        
        # Permutation of these selected indices to define the new order
        target_indices_for_shuffled_data = shuffle_indices_perm[torch.randperm(num_shuffled, device=image.device)]

        # Apply the shuffle
        patches[b, shuffle_indices_perm] = patches[b, target_indices_for_shuffled_data]

    # Reconstruct the image
    patches_reconstructed = patches.view(B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches_reconstructed = patches_reconstructed.permute(0, 3, 1, 4, 2, 5).contiguous()  # Shape: (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    shuffled_image = patches_reconstructed.view(B, C, H, W)

    return shuffled_image

def random_mask(img, patch_size, mask_ratio):
    """
    Apply patch-based random masking to the input image.
    Masked patches are set to zero.

    Parameters:
    img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
    patch_size (int): Size of each patch (patch_size x patch_size).
    mask_ratio (float): Proportion of patches to mask (0 < mask_ratio < 1).

    Returns:
    torch.Tensor: Masked image.
    """
    batch_size, channels, height, width = img.size()

    if not (height % patch_size == 0 and width % patch_size == 0):
        raise ValueError(f"Image height {height} and width {width} must be divisible by patch_size {patch_size}")

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_patches = num_patches_h * num_patches_w
    num_masked = int(mask_ratio * total_patches)

    masked_img = img.clone() # Start with a copy of the image

    if num_masked == 0: # No masking if mask_ratio is 0 or too small
        return masked_img
    if num_masked >= total_patches: # All masked if mask_ratio is 1 or more
        return torch.zeros_like(img)

    for b in range(batch_size):
        if total_patches == 0: continue
        perm = torch.randperm(total_patches, device=img.device)
        indices_to_mask = perm[:num_masked]

        for idx_val in indices_to_mask:
            idx = idx_val.item()
            row_start = (idx // num_patches_w) * patch_size
            col_start = (idx % num_patches_w) * patch_size
            masked_img[b, :, row_start:row_start + patch_size, col_start:col_start + patch_size] = 0.0
            
    return masked_img


def combined_keypoints_jigsaw_random_mask(image, keypoints, patch_size, random_mask_ratio):
    """
    Apply a combined masking strategy:
    1. Keypoint-based Jigsaw: Patches containing keypoints are shuffled amongst themselves.
    2. Random Masking: A portion of the *remaining* (non-keypoint) patches are masked (set to zero).

    Parameters:
    image (torch.Tensor): Input image tensor of shape (B, C, H, W).
    keypoints (torch.Tensor): Keypoints tensor, expected to be normalized (0-1) (B, N_kp, 2) or (N_kp, 2).
    patch_size (int): Size of each patch.
    random_mask_ratio (float): Proportion of non-keypoint patches to mask (0 <= random_mask_ratio <= 1).

    Returns:
    torch.Tensor: Masked image tensor.
    """
    B, C, H, W = image.shape
    if not (H % patch_size == 0 and W % patch_size == 0):
        raise ValueError(f"Image height {H} and width {W} must be divisible by patch_size {patch_size}")

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w

    # --- Part 1: Identify keypoint patches (for later random masking exclusion) ---
    _keypoints_for_id = keypoints.clone()
    if _keypoints_for_id.ndim == 2:
        _keypoints_for_id_batched = _keypoints_for_id.unsqueeze(0)
    else:
        _keypoints_for_id_batched = _keypoints_for_id
    
    # Handle cases where a single set of keypoints might be provided for a batch
    if _keypoints_for_id_batched.shape[0] == 1 and B > 1:
        _keypoints_for_id_batched = _keypoints_for_id_batched.repeat(B, 1, 1)
    elif _keypoints_for_id_batched.shape[0] != B:
        raise ValueError(f"Batch size mismatch: image has {B}, keypoints have {_keypoints_for_id_batched.shape[0]})")

    _keypoints_for_id_scaled = _keypoints_for_id_batched.clamp(min=0.0, max=1.0 - 1e-6)
    _keypoints_for_id_scaled[..., 0] = _keypoints_for_id_scaled[..., 0] * W
    _keypoints_for_id_scaled[..., 1] = _keypoints_for_id_scaled[..., 1] * H

    batch_keypoint_patch_indices_sets = []
    for b_idx in range(B):
        current_kp_patch_indices = set()
        for kp in _keypoints_for_id_scaled[b_idx]:
            x, y = kp[0].item(), kp[1].item()
            col = min(int(x // patch_size), num_patches_w - 1)
            row = min(int(y // patch_size), num_patches_h - 1)
            idx = row * num_patches_w + col
            current_kp_patch_indices.add(idx)
        batch_keypoint_patch_indices_sets.append(current_kp_patch_indices)

    # --- Part 2: Apply keypoint jigsaw masking ---
    # random_jigsaw_mask_keypoints returns a new tensor with shuffled patches.
    jigsawed_image = random_jigsaw_mask_keypoints(image.clone(), keypoints, patch_size=patch_size)

    # --- Part 3: Apply random masking to non-keypoint patches on the jigsawed_image ---
    output_image = jigsawed_image.clone() # Work on a copy of the jigsawed image

    for b_idx in range(B):
        keypoint_patches_for_this_item = batch_keypoint_patch_indices_sets[b_idx]
        
        non_keypoint_indices_list = [i for i in range(total_patches) if i not in keypoint_patches_for_this_item]
        
        if not non_keypoint_indices_list: # All patches are keypoint patches
            continue 
            
        non_keypoint_indices = torch.tensor(non_keypoint_indices_list, dtype=torch.long, device=image.device)
        num_non_keypoint_patches = non_keypoint_indices.shape[0]
        num_to_mask_randomly = int(num_non_keypoint_patches * random_mask_ratio)

        if num_to_mask_randomly == 0:
            continue
        
        if num_non_keypoint_patches == 0: # Should be caught by `if not non_keypoint_indices_list`
            continue

        perm = torch.randperm(num_non_keypoint_patches, device=image.device)
        indices_to_mask = non_keypoint_indices[perm[:num_to_mask_randomly]]

        for patch_idx_val in indices_to_mask:
            patch_idx = patch_idx_val.item()
            patch_row = (patch_idx // num_patches_w) * patch_size
            patch_col = (patch_idx % num_patches_w) * patch_size
            # Mask by setting to zero
            output_image[b_idx, :, patch_row:patch_row + patch_size, patch_col:patch_col + patch_size] = 0.0
            
    return output_image
