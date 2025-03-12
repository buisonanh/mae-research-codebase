import torch
from src.config import PATCH_SIZE

def partial_jigsaw_mask_keypoints(image, keypoints, patch_size=PATCH_SIZE):
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