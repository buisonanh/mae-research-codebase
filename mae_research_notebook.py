# %% [markdown]
# # MAE Research Notebook
# This notebook implements the Masked Autoencoder (MAE) research project, combining keypoint detection,
# autoencoding, and classification tasks. The notebook is organized into sections for configuration,
# data loading, model training, and evaluation.

# %% [markdown]
# ## 1. Import Dependencies and Configuration

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os

# %% [markdown]
# ## 2. Configuration Parameters

# %%
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
KEYPOINT_NUM_EPOCHS = 20
AUTOENCODER_NUM_EPOCHS = 50
CLASSIFIER_NUM_EPOCHS = 120
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
MODEL_SAVE_PATH = "checkpoints_affectnet_rafdb"

# %% [markdown]
# ## 3. Data Loading and Preprocessing Functions

# %%
def get_data_transforms():
    """Get data transforms for training and evaluation."""
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    }

def create_data_loaders():
    """Create data loaders for training, validation and testing."""
    data_transforms = get_data_transforms()
    
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_PATHS[CLASSIFY_DATASET_NAME]),
        transform=data_transforms['train']
    )
    testset = datasets.ImageFolder(
        root=os.path.join(DATASET_PATHS[CLASSIFY_DATASET_NAME].replace('train', 'test')),
        transform=data_transforms['test']
    )
    
    # Split training into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    seed = torch.Generator().manual_seed(64)
    trainset, valset = random_split(train_dataset, [train_size, val_size], generator=seed)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader

# %% [markdown]
# ## 4. Model Architecture

# %%
def create_model(weights_path, num_classes=None):
    """Create and initialize the classification model."""
    if num_classes is None:
        num_classes = NUM_CLASSES[CLASSIFY_DATASET_NAME]
        
    # Load pretrained weights
    weights = torch.load(os.path.join(MODEL_SAVE_PATH, 'pretrain_checkpoints', 'mae_checkpoints', 'mae_autoencoder_final.pth'), map_location=DEVICE)
    
    # Extract encoder weights
    encoder_weights = {k.replace("encoder.", ""): v for k, v in weights.items() if k.startswith("encoder.")}
    
    # Create model
    model = timm.create_model(
        model_name="resnet18",
        pretrained=False,
    )
    model.fc = nn.Identity()
    model.global_pool = nn.Identity()
    
    # Load encoder weights
    model.load_state_dict(encoder_weights)
    
    # Add classification head
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=512, out_features=num_classes)
    )
    
    return model

# %% [markdown]
# ## 5. Training Functions

# %%
def train_model(model, train_loader, val_loader, num_epochs=CLASSIFIER_NUM_EPOCHS):
    """Train the classification model."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_loss = float('inf')
    
    # Create classification checkpoint directory
    classification_dir = os.path.join(MODEL_SAVE_PATH, 'classification_checkpoints')
    os.makedirs(classification_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for images, labels in train_loader_tqdm:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc="Validation")
            for images, labels in val_loader_tqdm:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Update scheduler
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            torch.save(
                model.state_dict(),
                os.path.join(classification_dir, f'classifier_{CLASSIFY_DATASET_NAME}_epoch{epoch}.pth')
            )
            best_val_loss = epoch_val_loss
        
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_acc:.2f}%")
        print(f"  Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.2f}%")
    
    return train_losses, train_accuracies, val_losses, val_accuracies

# %% [markdown]
# ## 6. Evaluation Functions

# %%
def evaluate_model(model, test_loader):
    """Evaluate the model on test set and generate confusion matrix."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Testing")
        for images, labels in test_loader_tqdm:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Get class names
    if CLASSIFY_DATASET_NAME == "rafdb":
        classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    elif CLASSIFY_DATASET_NAME == "affectnet":
        classes = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    else:
        classes = test_loader.dataset.classes
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {CLASSIFY_DATASET_NAME} classification (%)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'classification_checkpoints', "confusion_matrix.png"))
    plt.close()
    
    return test_loss, test_accuracy

# %% [markdown]
# ## 7. Main Execution

# %%
if __name__ == "__main__":
    # Create directories
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'classification_checkpoints'), exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Create and train model
    model = create_model(weights_path=None)
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader
    )
    
    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_loader)
    
    print("Training and evaluation completed!") 