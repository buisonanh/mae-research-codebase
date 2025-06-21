import torch
import torch.nn as nn

class ViTClassificationHead(nn.Module):
    """
    Classification head for Vision Transformer (ViT) models.
    Selects the [CLS] token and applies a linear classifier.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: [batch_size, num_tokens, in_features]
        x_cls = x[:, 0, :]  # select [CLS] token
        return self.classifier(x_cls)
