import torch

# Device configuration
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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

# Model parameters
PATCH_SIZE = 16
NUM_KEYPOINTS = 15

# Training parameters
KEYPOINT_NUM_EPOCHS = 2
AUTOENCODER_NUM_EPOCHS = 2
CLASSIFIER_NUM_EPOCHS = 2

CLASSIFIER_LEARNING_RATE = 0.01
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# Image normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Dataset paths
DATASET_PATHS = {
    "rafdb": "/home/sonanhbui/projects/mae-research/dataset/raf-db-dataset/DATASET/train",
    "affectnet": "/home/sonanhbui/projects/mae-research/dataset/AffectNet/train"
}

# Model save paths
MODEL_SAVE_PATH = "checkpoints_affectnet_rafdb_normalize_sanity_check" 