import torch
import torch.nn as nn
import timm
from timm.models.convnext import ConvNeXtBlock

class Encoder(nn.Module):
    def __init__(self, model_name="resnet18"):
        super(Encoder, self).__init__()
        self.encoder = timm.create_model(
            model_name=model_name,
            pretrained=False,
        )
        # print(self.encoder)
        self.encoder.fc = nn.Identity()
        self.encoder.global_pool = nn.Identity()

    def forward(self, x):
        x = self.encoder.forward_features(x)
        return x

class ConvNeXtv2TinyDecoder(nn.Module):
    def __init__(self, decoder_embed_dim=512, decoder_depth=4): # Added parameters
        super(ConvNeXtv2TinyDecoder, self).__init__()
        
        encoder_output_channels = 768 
        decoder_output_channels = 3

        # 1. Projection layer to map encoder output to decoder_embed_dim
        self.proj = nn.Conv2d(
            in_channels=encoder_output_channels, 
            out_channels=decoder_embed_dim, 
            kernel_size=1)

        # 2. Sequence of ConvNeXt Blocks for decoding
        # These blocks will operate at the feature resolution of the encoder output
        # and use decoder_embed_dim channels.
        # Using ConvNeXtBlock from timm.models.convnext
        decoder_blocks_list = [ConvNeXtBlock(in_chs=decoder_embed_dim) for _ in range(decoder_depth)]
        self.decoder_blocks = nn.Sequential(*decoder_blocks_list)
        
        # Upsampling layers to reconstruct the image
        # Using nn.Upsample + nn.Conv2d for more stable training vs. ConvTranspose2d
        self.upsampler = nn.Sequential(
            # Input: [B, decoder_embed_dim, H_feat, W_feat] (e.g., [B, 512, 3, 3])
            # Upsample 3x3 -> 6x6
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(decoder_embed_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),

            # Upsample 6x6 -> 12x12
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # Upsample 12x12 -> 24x24
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Upsample 24x24 -> 48x48
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Upsample 48x48 -> 96x96
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Final layer to get to 3 channels for the image
            nn.Conv2d(64, decoder_output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(), # To ensure pixel values are in [0, 1]
        )
    
    def forward(self, x):
        # Project encoder features
        x = self.proj(x)
        
        # Pass through decoder blocks
        # Note: The paper's example includes mask token handling here for MAE.
        # If this is an MAE, that logic would be added around here.
        x = self.decoder_blocks(x)
        
        # Upsample to reconstruct image
        x = self.upsampler(x)
        return x

class Resnet18Decoder(nn.Module):
    def __init__(self):
        super(Resnet18Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, model_name="resnet18"):
        super().__init__()
        print(f"Using model {model_name} for encoder")
        self.encoder = Encoder(model_name=model_name)
        if model_name == "resnet18":
            self.decoder = Resnet18Decoder()
        elif model_name == "convnextv2_tiny":
            self.decoder = ConvNeXtv2TinyDecoder()

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat 