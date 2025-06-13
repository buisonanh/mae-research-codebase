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
        weights_path: Optional path to a pretrained checkpoint. If None, uses PyTorch default
                      pretrained weights for the ENCODER_MODEL.
        num_classes: Number of output classes. If None, uses the value from config.
    """
    if num_classes is None:
        num_classes = NUM_CLASSES[CLASSIFY_DATASET_NAME]

    if weights_path is None:
        # Load PyTorch default pretrained weights and let timm handle the classifier head
        print(f"Loading PyTorch default pretrained weights for {ENCODER_MODEL} with {num_classes} classes.")
        model = timm.create_model(
            model_name=ENCODER_MODEL,
            pretrained=True,
            num_classes=num_classes  # This ensures the model has the correct output shape
        )
        # No need to manually replace fc or global_pool if num_classes is set appropriately.
        # timm handles creating a suitable head.

    else:
        # Load weights from a specified checkpoint file (e.g., MAE encoder)
        # First, create the base model without a classification head (num_classes=0 for feature extraction)
        print(f"Creating base model {ENCODER_MODEL} for feature extraction (loading from path: {weights_path})")
        feature_extractor = timm.create_model(
            model_name=ENCODER_MODEL,
            pretrained=False,  # We are loading weights from a file
            num_classes=0,     # We want the feature extractor
            global_pool=''     # Return a 4D feature map
        )

        print(f"Loading pretrained weights from checkpoint: {weights_path}")
        weights = torch.load(weights_path, map_location=DEVICE)

        # Check if these are MAE encoder weights or full model weights
        # MAE pretraining often saves only the encoder part.
        if any(k.startswith("encoder.") for k in weights.keys()):
            print("Detected MAE-style encoder weights (prefixed with 'encoder.'). Extracting and loading.")
            # Filter and rename keys for the encoder
            encoder_weights = {k.replace("encoder.", ""): v for k, v in weights.items() if k.startswith("encoder.")}
            missing_keys, unexpected_keys = feature_extractor.load_state_dict(encoder_weights, strict=False)
        elif 'model' in weights:
            print("Detected checkpoint with 'model' key. Attempting to load state_dict from weights['model'].")
            missing_keys, unexpected_keys = feature_extractor.load_state_dict(weights['model'], strict=False)
        elif 'state_dict' in weights:
            print("Detected checkpoint with 'state_dict' key. Attempting to load state_dict from weights['state_dict'].")
            missing_keys, unexpected_keys = feature_extractor.load_state_dict(weights['state_dict'], strict=False)
        else:
            print("Attempting to load weights directly (assuming full model or compatible encoder checkpoint).")
            missing_keys, unexpected_keys = feature_extractor.load_state_dict(weights, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys when loading weights: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading weights: {unexpected_keys}")

        # Now, add the classification head to the feature_extractor
        try:
            in_features = feature_extractor.num_features
        except AttributeError:
            # Fallback if num_features is not directly available after num_classes=0
            # This might happen if timm model with num_classes=0 doesn't expose num_features directly.
            # We can try to infer it by running a dummy input or by checking the last layer's output channels.
            # For simplicity, we'll try to get it from a temporary model instance with a head.
            temp_model_for_features = timm.create_model(ENCODER_MODEL, pretrained=False, num_classes=1)
            in_features = temp_model_for_features.num_features
            del temp_model_for_features
            print(f"Inferred in_features as {in_features} for {ENCODER_MODEL}.")

        # Define the full model with the loaded feature_extractor and a new head
        model = nn.Sequential(
            feature_extractor,
            nn.AdaptiveAvgPool2d((1, 1)), # Global average pooling
            nn.Flatten(),                 # Flatten to [batch_size, features]
            nn.Linear(in_features, num_classes) # Classifier
        )

    return model

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
    classification_dir = CLASSIFICATION_FOLDER
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
    plt.savefig(os.path.join(CLASSIFICATION_FOLDER, "confusion_matrix.png"))
    plt.close()
    
    return test_loss, test_accuracy

def save_metrics(final_pretrain_loss, test_loss, test_accuracy):
    """Save all training metrics to a text file."""
    from src.utils.visualization import format_config_params
    metrics_file = os.path.join(CLASSIFICATION_FOLDER, 'final_metrics.txt')
    
    with open(metrics_file, 'w') as f:
        # Write configuration parameters
        f.write(format_config_params())
        f.write("\n=== Final Training Metrics ===\n\n")
        f.write("Pretraining:\n")
        f.write(f"Final Loss: {final_pretrain_loss:.4f}\n\n")
        f.write("Classification:\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot and save training metrics."""
    metrics_dir = os.path.join(CLASSIFICATION_FOLDER, "metrics_plots")
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
    
    # Create save directories
    classification_dir = CLASSIFICATION_FOLDER
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'metrics_plots'), exist_ok=True)
    
    # Load model with pretrained weights if specified
    model = create_model(weights_path=args.checkpoint)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Train model
    print("Training classifier...")
    metrics = train_model(model, train_loader, val_loader)
    train_losses, train_accuracies, val_losses, val_accuracies = metrics
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader)
    
    # Try to read final pretraining loss
    try:
        with open(os.path.join(PRETRAIN_FOLDER, 'final_pretrain_loss.txt'), 'r') as f:
            final_pretrain_loss = float(f.read().strip())
    except (FileNotFoundError, ValueError):
        final_pretrain_loss = float('nan')
    
    # Save all metrics
    save_metrics(final_pretrain_loss, test_loss, test_accuracy)

if __name__ == "__main__":
    main()