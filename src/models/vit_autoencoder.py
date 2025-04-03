import torch
import torch.nn as nn
import timm

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
    def __init__(self, embed_dim=384):
        super(ViTDecoder, self).__init__()
        # A simple decoder that operates on each token embedding independently.
        # This decoder "reconstructs" the embedding space of each token.
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, z):
        # z shape: (B, num_tokens, embed_dim)
        # Apply the MLP to each token (the linear layers work on the last dimension).
        reconstructed = self.decoder(z)
        return reconstructed

class VitAutoencoder(nn.Module):
    def __init__(self):
        super(VitAutoencoder, self).__init__()
        self.encoder = ViTEncoder()
        self.decoder = ViTDecoder(embed_dim=384)
    
    def forward(self, x):
        # Encode the input image into a sequence of latent tokens.
        latent = self.encoder(x)
        # Decode the latent tokens to reconstruct the embedding.
        x_reconstructed = self.decoder(latent)
        return x_reconstructed

# Print the ViTAutoencoder model architecture
# print(VitAutoencoder())
