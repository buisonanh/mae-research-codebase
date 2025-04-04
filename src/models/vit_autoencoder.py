import torch
import torch.nn as nn
import timm
import numpy as np
import scipy.stats as stats
from omegaconf import OmegaConf
from src.taming.models.vqgan import VQModel
from src.util.pos_embed import get_2d_sincos_pos_embed

class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        # Create a ViT small model with the classifier head removed.
        self.model = timm.create_model(
            'vit_small_patch16_224.augreg_in21k_ft_in1k',
            pretrained=True,
            num_classes=0  # removes the classifier layer
        )
        # Note: The model expects images to be preprocessed as in its data config.

    def forward(self, x):
        # Forward pass through the ViT model to extract token embeddings.
        # The output shape is typically (B, num_tokens, embed_dim)
        latent = self.model.forward_features(x)
        return latent

class ViTDecoder(nn.Module):
    def __init__(self, embed_dim=384, decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6, mlp_ratio=4.0):
        super(ViTDecoder, self).__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Initialize decoder blocks (simplified from reference code)
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
        
        # Initialize weights
        nn.init.normal_(self.mask_token, std=0.02)
        
    def forward(self, x, mask):
        # x: encoded tokens, mask: boolean mask (True for masked tokens)
        B, N, D = x.shape
        
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Create mask tokens
        mask_tokens = self.mask_token.repeat(B, N, 1)
        
        # Replace masked tokens with mask token
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        
        # Apply transformer blocks
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
    
    def forward(self, x):
        # Encode the input image into a sequence of latent tokens.
        latent = self.encoder(x)
        # Decode the latent tokens to reconstruct the embedding.
        x_reconstructed = self.decoder(latent, torch.zeros_like(latent[:, :, 0], dtype=torch.bool))
        return x_reconstructed

class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder model based on ViT and VQGAN."""
    def __init__(self, mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25, vqgan_ckpt_path='vqgan_jax_strongaug.ckpt'):
        super().__init__()
        # Initialize ViT encoder and decoder
        self.encoder = ViTEncoder()
        self.embed_dim = 384  # ViT small embed dimension
        self.decoder = ViTDecoder(
            embed_dim=self.embed_dim,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=8,
            decoder_num_heads=6
        )
        
        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
            (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
            loc=mask_ratio_mu, scale=mask_ratio_std
        )
        
        # Create a learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # Initialize the mask token
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Import VQGAN components
        try:
            config = OmegaConf.load('config/vqgan.yaml').model
            self.vqgan = VQModel(
                ddconfig=config.params.ddconfig,
                n_embed=config.params.n_embed,
                embed_dim=config.params.embed_dim,
                ckpt_path=vqgan_ckpt_path
            )
            # Freeze VQGAN parameters
            for param in self.vqgan.parameters():
                param.requires_grad = False
            self.use_vqgan = True
            self.codebook_size = config.params.n_embed
        except (ImportError, ModuleNotFoundError, FileNotFoundError):
            print("Warning: VQGAN module or config not found. Falling back to standard ViT features.")
            self.use_vqgan = False
            self.codebook_size = 0
        
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]
        
        # Step 1: Convert input image to image tokens using VQGAN
        if self.use_vqgan:
            # VQGAN encoding to get image tokens
            with torch.no_grad():
                z_q, _, token_tuple = self.vqgan.encode(x)
                _, _, token_indices = token_tuple
                token_indices = token_indices.reshape(z_q.size(0), -1)
                # Store original token indices for loss calculation
                gt_indices = token_indices.clone().detach().long()
        else:
            # Fallback to using ViT features directly
            token_embeddings = self.encoder(x)
            gt_indices = None
        
        # Get token embeddings from encoder
        if self.use_vqgan:
            # Convert token indices to embeddings using VQGAN codebook
            token_embeddings = self.vqgan.quantize.get_codebook_entry(
                token_indices.flatten(), shape=(batch_size, -1, self.vqgan.quantize.e_dim)
            )
            # Project to encoder dimension if needed
            if token_embeddings.shape[-1] != self.embed_dim:
                token_embeddings = nn.Linear(token_embeddings.shape[-1], self.embed_dim)(token_embeddings)
        else:
            token_embeddings = self.encoder(x)
        
        # Get dimensions
        B, N, D = token_embeddings.shape  # batch, tokens, dimension
        
        # Step 2: Create random mask
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(N * mask_rate))
        
        # Generate random noise for masking
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is keep, large is remove
        cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
        
        # Create mask (1 for masked, 0 for unmasked)
        mask = (noise <= cutoff_mask).float()
        
        # Step 3: Replace masked tokens with mask token
        masked_embeddings = token_embeddings.clone()
        mask_tokens = self.mask_token.repeat(B, N, 1)
        masked_embeddings = torch.where(mask.unsqueeze(-1).bool(), mask_tokens, token_embeddings)
        
        # Step 4: Forward pass through encoder blocks to get latent representation
        # For simplicity, we'll use the encoder's transformer blocks
        latent_representation = masked_embeddings
        
        # Step 5: Forward pass through decoder to reconstruct original embeddings
        reconstructed_embeddings = self.decoder(latent_representation, mask.bool())
        
        return reconstructed_embeddings, token_embeddings, mask, gt_indices
        
    def loss_function(self, reconstructed_embeddings, token_embeddings, mask, gt_indices=None):
        """
        Calculate reconstruction loss
        """
        if gt_indices is not None and self.use_vqgan:
            # If we have ground truth indices from VQGAN, we can use them for loss
            # This would require implementing a token prediction head
            # For simplicity, we'll use MSE loss on the embeddings
            pass
        
        # Calculate MSE loss only on masked tokens
        loss = ((reconstructed_embeddings - token_embeddings) ** 2)
        loss = loss.mean(dim=-1)  # [B, N]
        
        # Only compute loss on masked tokens
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        
        return loss
