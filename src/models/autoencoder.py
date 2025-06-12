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
    def __init__(self):
        super(ConvNeXtv2TinyDecoder, self).__init__()
        
        encoder_output_channels = 768 
        decoder_output_channels = 3

        # A single ConvNeXt block from timm.models.convnext
        # It processes features at the same spatial resolution and can transform channel depth if configured.
        # Here, in_chs=encoder_output_channels means it expects 768 channels and will output 768 channels by default.
        self.convnext_block = ConvNeXtBlock(in_chs=encoder_output_channels) # Uses default parameters for the block

        # Upsampling layers to reconstruct the image from the features processed by ConvNeXtBlock
        self.upsampler = nn.Sequential(
            # Input to upsampler is [B, 768, H_feat, W_feat] (e.g., [B, 768, 3, 3])
            # Upsample 3x3 -> 6x6, 768ch -> 512ch
            nn.ConvTranspose2d(encoder_output_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample 6x6 -> 12x12, 512ch -> 256ch
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample 12x12 -> 24x24, 256ch -> 128ch
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample 24x24 -> 48x48, 128ch -> 64ch
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample 48x48 -> 96x96, 64ch -> 64ch
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Final layer to get to 3 channels for the image, no spatial change from 96x96
            nn.ConvTranspose2d(64, decoder_output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(), # To ensure pixel values are in [0, 1]
        )
    
    def forward(self, x):
        # Pass input through the ConvNeXt block first
        x = self.convnext_block(x)
        # Then through the upsampler
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