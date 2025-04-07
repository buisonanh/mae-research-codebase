import torch
import os
import json

# Device configuration
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Dataset parameters
TARGET_SIZE = 96
BATCH_SIZE = 512
NUM_WORKERS = 4
PRETRAIN_DATASET_NAME = "affectnet"  # Dataset for pretraining
CLASSIFY_DATASET_NAME = "rafdb"  # Dataset for classification
NUM_CLASSES = {
    "rafdb": 7,  # For RAF-DB dataset
    "affectnet": 8  # For AffectNet dataset
}

# Maskng strategy
MASKING_STRATEGY = "random" # "keypoints-jigsaw", "random-jigsaw", "random"

# Model parameters
PATCH_SIZE = 16

# For keypoints-based jigsaw masking
NUM_KEYPOINTS = 15

# For random jigsaw masking and jigsaw masking only
MASK_RATIO = 0.4

# Training parameters
AUTOENCODER_NUM_EPOCHS = 50
CLASSIFIER_NUM_EPOCHS = 120
KEYPOINT_NUM_EPOCHS = 20

LEARNING_RATE = 0.001
CLASSIFIER_LEARNING_RATE = 0.01
EARLY_STOPPING_PATIENCE = 5

# Image normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Dataset paths
DATASET_PATHS = {
    "rafdb": "datasets/raf-db-dataset/DATASET/train",
    "affectnet": "datasets/affectnet/AffectNet/train",
    "keypoints": "datasets/keypoints/training_data/training.csv"
}

# Model save paths

SAVE_PATH = f"results_{PRETRAIN_DATASET_NAME}_{CLASSIFY_DATASET_NAME}_{MASKING_STRATEGY}"

PRETRAIN_FOLDER = os.path.join(SAVE_PATH, f'pretrain_checkpoints')
CLASSIFICATION_FOLDER = os.path.join(SAVE_PATH, f'classification_checkpoints')