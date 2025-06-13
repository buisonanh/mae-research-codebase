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


# class ConvNeXtv2NanoDecoder(nn.Module):
#     def __init__(self, decoder_embed_dim=640, decoder_depth=4):
#         super(ConvNeXtv2NanoDecoder, self).__init__()

class ConvNeXtv2TinyDecoder(nn.Module):
    def __init__(self, decoder_embed_dim=768, decoder_depth=4): # Added parameters
        super(ConvNeXtv2TinyDecoder, self).__init__()
        
        encoder_output_channels = 768 
        decoder_output_channels = 3

        # 2. Sequence of ConvNeXt Blocks for decoding
        # These blocks will operate at the feature resolution of the encoder output
        # and use decoder_embed_dim channels.
        # Using ConvNeXtBlock from timm.models.convnext
        # Forcing a single block for diagnosis, ignoring decoder_depth for this part
        # Initialize the single ConvNeXt Block, operating with 'decoder_embed_dim' channels.
        # Assumes encoder output or preceding projection layer provides 'decoder_embed_dim' channels.
        self.single_convnext_block = ConvNeXtBlock(in_chs=decoder_embed_dim)
        
        # Upsampling layers to reconstruct the image from a small spatial dimension (e.g., 3x3)
        # up to the original image size (e.g., 96x96).
        # The input to the upsampler comes from self.single_convnext_block,
        # which outputs features with 'decoder_embed_dim' (768) channels.

        upsample_in_channels = decoder_embed_dim  # Starting channels for upsampling (768)

        # Define channel dimensions for each upsampling stage.
        # This sequence upsamples 5 times, e.g., 3x3 -> 6x6 -> 12x12 -> 24x24 -> 48x48 -> 96x96.
        # Channel progression: 768 -> 384 -> 192 -> 96 -> 48 -> 24
        upsample_stage_channels = [upsample_in_channels, 384, 192, 96, 48, 24]

        upsampling_modules = []
        for i in range(len(upsample_stage_channels) - 1):
            upsampling_modules.extend([
                nn.ConvTranspose2d(upsample_stage_channels[i], upsample_stage_channels[i+1],
                                   kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(upsample_stage_channels[i+1])
            ])
        self.upsample_layers = nn.Sequential(*upsampling_modules)

        # Final prediction layer to map the upsampled features to the desired number of output channels
        # (e.g., 3 for an RGB image).
        self.pred = nn.Conv2d(
            in_channels=upsample_stage_channels[-1],  # Input channels from the last upsampling stage (24)
            out_channels=decoder_output_channels,     # Target output channels (e.g., 3)
            kernel_size=1                             # 1x1 convolution for channel mapping
        )
    
    def forward(self, x):
        # Project encoder features
        
        # Pass through decoder blocks
        # Note: The paper's example includes mask token handling here for MAE.
        # If this is an MAE, that logic would be added around here.
        x = self.single_convnext_block(x)
        
        # Pass through upsampling layers to reconstruct spatial dimensions
        x = self.upsample_layers(x)
        
        # Final prediction layer to map to output channels
        x = self.pred(x)
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