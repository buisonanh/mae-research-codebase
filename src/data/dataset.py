import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import pandas as pd
from src.config import (
    TARGET_SIZE, BATCH_SIZE, NUM_WORKERS, DATASET_PATHS,
    PRETRAIN_DATASET_NAME
)

class FacialKeypointsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.keypoint_columns = self.dataframe.columns[:-1]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the image (already a NumPy array of shape (96, 96))
        image = self.dataframe.iloc[idx]['Image']

        # If shape is (1,96,96), squeeze out channel first
        if image.shape[0] == 1 and len(image.shape) == 3:
            image = image[0]

        # Convert [0..1] float -> [0..255] uint8
        image_8bit = (image * 255).astype(np.uint8)
        
        # Turn into a PIL grayscale image
        img_pil = Image.fromarray(image_8bit, mode='L')

        # Apply TorchVision transforms if needed
        if self.transform:
            img_pil = self.transform(img_pil)

        # Convert keypoints
        keypoints = self.dataframe.iloc[idx][self.keypoint_columns].values.astype('float32')
        keypoints_torch = torch.tensor(keypoints, dtype=torch.float32)

        return img_pil, keypoints_torch

def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
    ])

def load_keypoints_data(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Preprocessing the images
    def preprocess_image(image_str):
        image = np.array(image_str.split(), dtype=np.float32).reshape(96, 96) / 255.0
        return image

    df['Image'] = df['Image'].apply(preprocess_image)
    
    # Normalize keypoints to [0, 1]
    keypoint_columns = df.columns[:-1]
    df[keypoint_columns] = df[keypoint_columns] / 96.0
    
    # Handle missing values
    df = df.dropna()
    
    return df

def create_data_loaders(train_df, val_df, test_df):
    transform = get_transform()
    
    # Create datasets
    train_dataset = FacialKeypointsDataset(train_df, transform=transform)
    val_dataset = FacialKeypointsDataset(val_df, transform=transform)
    test_dataset = FacialKeypointsDataset(test_df, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, val_loader, test_loader

def get_image_dataset():
    transform = get_transform()
    dataset = datasets.ImageFolder(
        root=DATASET_PATHS[PRETRAIN_DATASET_NAME],
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    ) 