import torch

# Device configuration
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Dataset parameters
TARGET_SIZE = 96
BATCH_SIZE = 128
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
KEYPOINT_NUM_EPOCHS = 20
AUTOENCODER_NUM_EPOCHS = 50
CLASSIFIER_NUM_EPOCHS = 10
CLASSIFIER_LEARNING_RATE = 0.01
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# Image normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Dataset paths
DATASET_PATHS = {
    "rafdb": "/home/sonanhbui/projects/mae-research-codebase/datasets/raf-db-dataset/DATASET/train",
    "affectnet": "/home/sonanhbui/projects/mae-research-codebase/datasets/affectnet/AffectNet/train",
    "keypoints": "/home/sonanhbui/projects/mae-research-codebase/datasets/keypoints/training_data/training.csv"
}

# Model save paths
MODEL_SAVE_PATH = "checkpoints_affectnet_rafdb" 

# Pretrained checkpoint path (set to None to train from scratch)
PRETRAINED_CHECKPOINT_PATH = None