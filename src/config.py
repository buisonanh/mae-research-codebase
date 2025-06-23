import torch
import os
import json

# Device configuration
DEVICE = torch.device(os.environ.get('DEVICE', 'cuda:2'))

# Dataset parameters
TARGET_SIZE = 96
BATCH_SIZE = 512
NUM_WORKERS = 4
import os

PRETRAIN_DATASET_NAME = os.environ.get('PRETRAIN_DATASET_NAME', 'affectnet')
CLASSIFY_DATASET_NAME = os.environ.get('CLASSIFY_DATASET_NAME', 'rafdb')
NUM_CLASSES = {
    "rafdb": 7,  # For RAF-DB dataset
    "affectnet": 8  # For AffectNet dataset
}

ENCODER_MODEL = os.environ.get('ENCODER_MODEL', 'resnet18') # [resnet18, convnextv2_tiny]

# Masking strategy
MASKING_STRATEGY = os.environ.get('MASKING_STRATEGY', 'random') # "keypoints-jigsaw", "random-jigsaw", "random", "combined-keypoints-jigsaw-random-mask"

# Model parameters
PATCH_SIZE = int(os.environ.get('PATCH_SIZE', 16))

# For keypoints-based jigsaw masking
NUM_KEYPOINTS = int(os.environ.get('NUM_KEYPOINTS', 15))

# For random jigsaw masking and jigsaw masking only
MASK_RATIO = float(os.environ.get('MASK_RATIO', 0.4))

# Training parameters
AUTOENCODER_NUM_EPOCHS = int(os.environ.get('AUTOENCODER_NUM_EPOCHS', 50))
CLASSIFIER_NUM_EPOCHS = int(os.environ.get('CLASSIFIER_NUM_EPOCHS', 120))
KEYPOINT_NUM_EPOCHS = int(os.environ.get('KEYPOINT_NUM_EPOCHS', 20))

LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.001))
CLASSIFIER_LEARNING_RATE = float(os.environ.get('CLASSIFIER_LEARNING_RATE', 0.001))
EARLY_STOPPING_PATIENCE = int(os.environ.get('EARLY_STOPPING_PATIENCE', 5))

# Image normalization parameters
# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Dataset paths
DATASET_PATHS = {
    "rafdb": "datasets/raf-db-dataset/DATASET",
    "affectnet": "datasets/affectnet/AffectNet",
    "keypoints": "datasets/keypoints/training_data/training.csv"
}

# Model save paths

SAVE_PATH = f"results_{ENCODER_MODEL}_{PRETRAIN_DATASET_NAME}_{CLASSIFY_DATASET_NAME}_{MASKING_STRATEGY}_mr{MASK_RATIO}_lr{LEARNING_RATE}"

PRETRAIN_FOLDER = os.path.join(SAVE_PATH, f'pretrain_checkpoints')
CLASSIFICATION_FOLDER = os.path.join(SAVE_PATH, f'classification_checkpoints')
