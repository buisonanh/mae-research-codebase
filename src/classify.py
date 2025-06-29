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
import os
from torch.utils.data import random_split
import json

from src.config import (
    DEVICE, TARGET_SIZE, BATCH_SIZE, NUM_WORKERS, CLASSIFY_DATASET_NAME,
    MEAN, STD, DATASET_PATHS, NUM_CLASSES,
    CLASSIFIER_NUM_EPOCHS, CLASSIFIER_LEARNING_RATE, PRETRAIN_FOLDER, CLASSIFICATION_FOLDER, ENCODER_MODEL
)

def create_model(weights_path=None, num_classes=None):
    """Create and initialize the classification model.

    Args:
        weights_path: Optional path to a pretrained checkpoint. If "default", uses PyTorch default
                      pretrained weights. If None, the model is not pretrained.
        num_classes: Number of output classes. If None, uses the value from config.

    Returns:
        A tuple of (model, weights_loaded_flag).
    """
    if num_classes is None:
        num_classes = NUM_CLASSES[CLASSIFY_DATASET_NAME]

    weights_loaded = False

    # Case 1: Use "default" pretrained weights from PyTorch
    if weights_path == "default":
        print(f"Loading PyTorch pretrained {ENCODER_MODEL} with {num_classes} classes.")
        model = timm.create_model(model_name=ENCODER_MODEL, pretrained=True, num_classes=num_classes)
        weights_loaded = True
        return model, weights_loaded

    # Case 2: No weights path provided, create a non-pretrained model
    if weights_path is None:
        print(f"Loading PyTorch non-pretrained {ENCODER_MODEL} with {num_classes} classes.")
        model = timm.create_model(model_name=ENCODER_MODEL, pretrained=False, num_classes=num_classes)
        weights_loaded = False
        return model, weights_loaded

    # Case 3: A specific checkpoint path is provided
    if not os.path.exists(weights_path):
        print(f"Warning: Checkpoint file not found at {weights_path}. Creating a new model with random weights.")
        model = timm.create_model(model_name=ENCODER_MODEL, pretrained=False, num_classes=num_classes)
        weights_loaded = False
        return model, weights_loaded

    # If we reach here, the file exists. Load it.
    print(f"Creating base model {ENCODER_MODEL} for feature extraction (loading from path: {weights_path})")
    feature_extractor = timm.create_model(
        model_name=ENCODER_MODEL, pretrained=False, num_classes=0, global_pool=''
    )

    try:
        weights = torch.load(weights_path, map_location=DEVICE)
        if any(k.startswith("encoder.") for k in weights.keys()):
            print("Detected MAE-style encoder weights. Extracting and loading.")
            encoder_weights = {k.replace("encoder.", ""): v for k, v in weights.items() if k.startswith("encoder.")}
            feature_extractor.load_state_dict(encoder_weights, strict=False)
        elif 'model' in weights:
            feature_extractor.load_state_dict(weights['model'], strict=False)
        elif 'state_dict' in weights:
            feature_extractor.load_state_dict(weights['state_dict'], strict=False)
        else:
            feature_extractor.load_state_dict(weights, strict=False)
        weights_loaded = True
    except Exception as e:
        print(f"Error loading weights from {weights_path}: {e}. Creating a new model with random weights.")
        model = timm.create_model(model_name=ENCODER_MODEL, pretrained=False, num_classes=num_classes)
        weights_loaded = False
        return model, weights_loaded

    try:
        in_features = feature_extractor.num_features
    except AttributeError:
        temp_model = timm.create_model(ENCODER_MODEL, pretrained=False, num_classes=1)
        in_features = temp_model.num_features
        del temp_model

    model = nn.Sequential(
        feature_extractor,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features, num_classes)
    )

    return model, weights_loaded

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
    
    # Correctly point to the 'train' and 'test' subdirectories for ImageFolder
    base_path = DATASET_PATHS[CLASSIFY_DATASET_NAME]
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    # Check if these paths exist to avoid errors with different dataset structures.
    if not os.path.isdir(train_path):
        print(f"Warning: 'train' subdirectory not found in {base_path}. Using root directory for training.")
        train_path = base_path
    
    if not os.path.isdir(test_path):
        print(f"Warning: 'test' subdirectory not found in {base_path}. Using root directory for testing.")
        test_path = base_path

    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=data_transforms['train']
    )
    testset = datasets.ImageFolder(
        root=test_path,
        transform=data_transforms['test']
    )

    # Sanity check: ensure the number of classes matches the configuration
    num_found_classes = len(train_dataset.classes)
    num_expected_classes = NUM_CLASSES[CLASSIFY_DATASET_NAME]
    if num_found_classes != num_expected_classes:
        print(f"Warning: Found {num_found_classes} classes, but expected {num_expected_classes}.")
        print(f"Found classes: {train_dataset.classes}")
    
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

def train_model(model, train_loader, val_loader, classification_dir, num_epochs=CLASSIFIER_NUM_EPOCHS):
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
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], ")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, ")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Save model if validation loss improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(classification_dir, 'best_model.pth'))
            print("Best model saved!")
        
        # Save metrics to JSON file
        metrics_dict = {
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        }
        
        # Save to JSON file
        results_file = os.path.join(classification_dir, 'training_metrics.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(metrics_dict)
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        # Step the scheduler
        scheduler.step(epoch_val_loss)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, test_loader, classification_dir):
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
    plt.savefig(os.path.join(classification_dir, "confusion_matrix.png"))
    plt.close()
    
    return test_loss, test_accuracy

def save_metrics(final_pretrain_loss, test_loss, test_accuracy, classification_dir):
    """Save all training metrics to a text file."""
    from src.utils.visualization import format_config_params
    metrics_file = os.path.join(classification_dir, 'final_metrics.txt')
    
    with open(metrics_file, 'w') as f:
        # Write configuration parameters
        f.write(format_config_params())
        f.write("\n=== Final Training Metrics ===\n\n")
        if final_pretrain_loss is not None:
            f.write("Pretraining:\n")
            f.write(f"Final Loss: {final_pretrain_loss:.4f}\n\n")
        f.write("Classification:\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, classification_dir):
    """Plot and save training metrics."""
    metrics_dir = os.path.join(classification_dir, "metrics_plots")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "training_validation.png"))
    plt.close()

def main():
    """Main function to run the classification training."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a classifier with optional pretrained checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to a pretrained checkpoint to use for initialization')
    args = parser.parse_args()
    
    # Load model and determine output directory
    model, weights_loaded = create_model(weights_path=args.checkpoint)

    if weights_loaded and args.checkpoint != "default":
        # Use the predefined folder from config if a custom checkpoint was loaded
        classification_dir = CLASSIFICATION_FOLDER
    else:
        # Create a new folder for training from scratch or from default pytorch weights
        training_type = "from_pytorch_pretrain" if args.checkpoint == "default" else "from_scratch"
        classification_dir = f"results_{ENCODER_MODEL}_{CLASSIFY_DATASET_NAME}_{training_type}_lr{CLASSIFIER_LEARNING_RATE}"

    # Create save directories
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'metrics_plots'), exist_ok=True)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders()

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, classification_dir, num_epochs=CLASSIFIER_NUM_EPOCHS
    )

    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, test_loader, classification_dir)

    # Save final metrics
    pretrain_loss = None  # No pretraining loss available in this script
    save_metrics(pretrain_loss, test_loss, test_accuracy, classification_dir)

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, classification_dir)

if __name__ == "__main__":
    main()