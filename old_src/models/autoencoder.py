import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm

class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.model = timm.create_model(
            'vit_small_patch16_224.augreg_in21k_ft_in1k',
            pretrained=True,
            num_classes=0
        )
    def forward(self, x):
        # ViT expects 3-channel input; repeat if grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        latent = self.model.forward_features(x)  # (B, num_tokens, embed_dim)
        return latent

class ViTDecoder(nn.Module):
    def __init__(self, embed_dim=384, decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6, mlp_ratio=4.0):
        super(ViTDecoder, self).__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * mlp_ratio),
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim)
        nn.init.normal_(self.mask_token, std=0.02)
    def forward(self, x, mask):
        # x: encoded tokens, mask: boolean mask (True for masked tokens)
        B, N, D = x.shape
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(B, N, 1)
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

class VitAutoencoder(nn.Module):
    def __init__(self):
        super(VitAutoencoder, self).__init__()
        self.encoder = ViTEncoder()
        self.decoder = ViTDecoder(embed_dim=384)
        # Patchify/Unpatchify settings
        self.img_size = 224  # match ViT and config
        self.patch_size = 16
        self.num_channels = 3
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.reconstruct_linear = nn.Linear(384, self.patch_size * self.patch_size * self.num_channels)
        self.unfold = None  # not used, see below
    def forward(self, x):
        # Patchify
        from src.utils.masking import Patchify
        B, C, H, W = x.shape
        patchify = Patchify(self.patch_size)
        patches = patchify(x)  # (B, num_patches, C, patch_size, patch_size)
        patches = patches.view(B, self.num_patches, -1)  # (B, num_patches, patch_dim)

        # Encode
        latent = self.encoder(x)  # (B, num_patches, embed_dim)

        # Mask: all False (no masking during reconstruction)
        mask = torch.zeros((B, self.num_patches), dtype=torch.bool, device=x.device)

        # Decode
        decoded = self.decoder(latent, mask)  # (B, num_patches, embed_dim)

        # Reconstruct patches
        rec_patches = self.reconstruct_linear(decoded)  # (B, num_patches, patch_size*patch_size*num_channels)
        rec_patches = rec_patches.view(B, self.num_patches, self.num_channels, self.patch_size, self.patch_size)

        # Unpatchify
        # Arrange patches back to image
        num_patches_per_row = self.img_size // self.patch_size
        rows = []
        for i in range(num_patches_per_row):
            row_patches = rec_patches[:, i*num_patches_per_row:(i+1)*num_patches_per_row]
            row = torch.cat([p for p in row_patches[0]], dim=-1)  # (C, patch_size, img_size)
            for b in range(1, B):
                row = torch.cat((row, torch.cat([p for p in row_patches[b]], dim=-1)), dim=0)
            rows.append(row)
        # Stack rows to form the image
        img_recon = torch.cat([
            torch.cat([rec_patches[b, i*num_patches_per_row:(i+1)*num_patches_per_row].permute(0,2,3,1).reshape(self.num_channels, self.patch_size, self.img_size)
                       for i in range(num_patches_per_row)], dim=1)
            for b in range(B)
        ], dim=0)
        img_recon = img_recon.view(B, self.num_channels, self.img_size, self.img_size)
        return img_recon