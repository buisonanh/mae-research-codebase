import torch
import numpy as np

from torch import nn
from src.config import PATCH_SIZE


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

    keypoints = keypoints.clamp(min=0.0, max=0.999)
        
    B, C, H, W = image.shape
    
    # Multiply normalized coords by width & height
    keypoints[:, :, 0] = keypoints[:, :, 0] * W  # x-coords
    keypoints[:, :, 1] = keypoints[:, :, 1] * H  # y-coords
    
    assert H % patch_size == 0 and W % patch_size == 0
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w

    patches = (
        image.unfold(2, patch_size, patch_size)
             .unfold(3, patch_size, patch_size)
             .permute(0, 2, 3, 1, 4, 5)
             .contiguous()
             .view(B, total_patches, C, patch_size, patch_size)
    )
    
    for b in range(B):
        patch_indices_set = set()
        for kp in keypoints[b]:
            x, y = kp[0].item(), kp[1].item()
            row = int(y // patch_size)
            col = int(x // patch_size)
            idx = row * num_patches_w + col
            patch_indices_set.add(idx)

        if patch_indices_set:
            patch_indices = torch.tensor(list(patch_indices_set), device=image.device)
            shuffled_indices = patch_indices[torch.randperm(len(patch_indices))]
            patches[b, patch_indices] = patches[b, shuffled_indices]
    
    patches = patches.view(B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled_image = patches.view(B, C, H, W)
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
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch_size"

    # Number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w

    # Reshape the image into patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # Shape: (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches = patches.view(B, -1, C, patch_size, patch_size)  # Shape: (B, num_patches, C, patch_size, patch_size)

    # Shuffle a subset of patches
    for b in range(B):
        patch_indices = torch.arange(total_patches)
        num_shuffled = int(total_patches * shuffle_ratio)
        shuffle_indices = patch_indices[torch.randperm(total_patches)[:num_shuffled]]
        shuffled_subset = shuffle_indices[torch.randperm(num_shuffled)]

        # Swap patches in the subset
        patches[b, shuffle_indices] = patches[b, shuffled_subset]

    # Reconstruct the image
    patches = patches.view(B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()  # Shape: (B, C, num_patches_h * patch_size, num_patches_w * patch_size)
    shuffled_image = patches.view(B, C, H, W)

    return shuffled_image

def random_mask(img, patch_size, mask_ratio):
    """
    Apply patch-based random masking to the input image.

    Parameters:
    img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
    patch_size (int): Size of each patch (patch_size x patch_size).
    mask_ratio (float): Proportion of patches to mask (0 < mask_ratio < 1).

    Returns:
    torch.Tensor: Masked image with only selected patches kept.
    """
    batch_size, channels, height, width = img.size()
    # print("Height:", height)
    # print("Width:", width)

    # Ensure the image dimensions are divisible by patch_size
    assert height % patch_size == 0 and width % patch_size == 0, \
        "Image dimensions must be divisible by the patch size."

    # Calculate the number of patches along each dimension
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_patches = num_patches_h * num_patches_w
    num_keep = int((1 - mask_ratio) * total_patches)

    np.random.seed(64)  # Seed for reproducibility

    # Initialize output tensor
    masked_img = torch.zeros_like(img)

    # Apply patchify and masking for each image in the batch
    patchify = Patchify(patch_size=patch_size)
    patches = patchify(img)  # Extract patches

    for b in range(batch_size):
        # Randomly select patches to keep
        keep_indices = np.random.choice(total_patches, num_keep, replace=False)

        # Keep only selected patches
        for idx in keep_indices:
            patch = patches[b, idx]  # Get the patch
            row = (idx // num_patches_w) * patch_size
            col = (idx % num_patches_w) * patch_size

            # Place the patch back in the corresponding location
            masked_img[b, :, row:row + patch_size, col:col + patch_size] = patch

    return masked_img
