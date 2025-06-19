import torch
import torch.nn as nn
from src.config import NUM_KEYPOINTS

class KeypointCNN(nn.Module):
    def __init__(self):
        super(KeypointCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 512), nn.ReLU(),
            nn.Linear(512, NUM_KEYPOINTS * 2),  # Predict all keypoint coordinates
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x 