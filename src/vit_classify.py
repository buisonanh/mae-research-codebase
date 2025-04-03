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
import argparse
from torch.utils.data import random_split

from src.config import (
    DEVICE, TARGET_SIZE, BATCH_SIZE, NUM_WORKERS, CLASSIFY_DATASET_NAME,
    MEAN, STD, DATASET_PATHS, MODEL_SAVE_PATH, NUM_CLASSES,
    CLASSIFIER_NUM_EPOCHS, CLASSIFIER_LEARNING_RATE
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a ViT classifier with different pretrained weights')
    parser.add_argument('--weights', type=str, default='timm', 
                        choices=['timm', 'mae', 'combined', 'none'],
                        help='Which pretrained weights to use: timm (default), mae (your pretrained), combined, or none')
    parser.add_argument('--mae_checkpoint', type=str, default=None,
                        help='Path to custom MAE checkpoint (default: use the final checkpoint)')
    parser.add_argument('--epochs', type=int, default=CLASSIFIER_NUM_EPOCHS,
                        help=f'Number of epochs to train (default: {CLASSIFIER_NUM_EPOCHS})')
    parser.add_argument('--lr', type=float, default=CLASSIFIER_LEARNING_RATE,
                        help=f'Learning rate (default: {CLASSIFIER_LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    return parser.parse_args()

def create_model(weights_type='timm', mae_checkpoint_path=None, num_classes=None):
    """Create and initialize the ViT classification model.
    
    Args:
        weights_type: Which pretrained weights to use: 'timm', 'mae', 'combined', or 'none'
        mae_checkpoint_path: Path to custom MAE checkpoint
        num_classes: Number of output classes. If None, uses the value from config.
    """
    if num_classes is None:
        num_classes = NUM_CLASSES[CLASSIFY_DATASET_NAME]
    
    # Create the ViT model
    if weights_type == 'timm':
        print("Using pretrained weights from timm")
        model = timm.create_model(
            'vit_small_patch16_224.augreg_in21k_ft_in1k',
            pretrained=True,
            num_classes=0  # removes the classifier layer
        )
    elif weights_type == 'none':
        print("Training from scratch (no pretrained weights)")
        model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=0  # removes the classifier layer
        )
    else:
        # For 'mae' or 'combined', we'll start with the model structure
        if weights_type == 'combined':
            print("Using combined weights (timm + MAE)")
            model = timm.create_model(
                'vit_small_patch16_224.augreg_in21k_ft_in1k',
                pretrained=True,
                num_classes=0  # removes the classifier layer
            )
        else:  # 'mae'
            print("Using only MAE pretrained weights")
            model = timm.create_model(
                'vit_small_patch16_224',
                pretrained=False,
                num_classes=0  # removes the classifier layer
            )
        
        # Load MAE weights
        if mae_checkpoint_path is None:
            # Use default MAE checkpoint
            mae_checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'vit_pretrain_checkpoints', 'vit_mae_checkpoints', 'vit_mae_autoencoder_final.pth')
        
        if os.path.exists(mae_checkpoint_path):
            print(f"Loading MAE weights from: {mae_checkpoint_path}")
            weights = torch.load(mae_checkpoint_path, map_location=DEVICE)
            
            # Extract encoder weights from autoencoder
            encoder_weights = {}
            for k, v in weights.items():
                # Handle both direct encoder weights and weights inside autoencoder
                if k.startswith("encoder."):
                    encoder_weights[k.replace("encoder.", "")] = v
                elif k.startswith("autoencoder.encoder."):
                    encoder_weights[k.replace("autoencoder.encoder.", "")] = v
                elif "encoder" in k and "model" in k:
                    # Handle nested model structure
                    new_key = k.split("encoder.model.")[-1] if "encoder.model." in k else k
                    encoder_weights[new_key] = v
            
            # Try to load weights
            if encoder_weights:
                try:
                    # For 'combined', we'll load weights selectively to avoid overwriting timm weights
                    if weights_type == 'combined':
                        # Get current model state dict
                        model_dict = model.state_dict()
                        # Filter encoder weights to only include those that match model structure
                        filtered_encoder_weights = {k: v for k, v in encoder_weights.items() 
                                                if k in model_dict and model_dict[k].shape == v.shape}
                        # Update model dict with filtered weights
                        model_dict.update(filtered_encoder_weights)
                        model.load_state_dict(model_dict)
                        print(f"Successfully loaded {len(filtered_encoder_weights)} layers from MAE weights")
                    else:
                        # For 'mae', try to load all weights
                        model.load_state_dict(encoder_weights, strict=False)
                        print("Successfully loaded MAE encoder weights")
                except RuntimeError as e:
                    print(f"Warning: Could not load all weights: {e}")
                    # Try to load weights that match
                    model_dict = model.state_dict()
                    filtered_encoder_weights = {k: v for k, v in encoder_weights.items() 
                                            if k in model_dict and model_dict[k].shape == v.shape}
                    if filtered_encoder_weights:
                        model_dict.update(filtered_encoder_weights)
                        model.load_state_dict(model_dict)
                        print(f"Loaded {len(filtered_encoder_weights)} matching layers")
            else:
                print("No encoder weights found in checkpoint")
        else:
            print(f"Warning: MAE checkpoint not found at {mae_checkpoint_path}")
    
    # Add classification head
    model.head = nn.Linear(model.embed_dim, num_classes)
    
    return model

def get_data_transforms():
    """Get data transforms for training and evaluation."""
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),  # ViT requires 224x224 input
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),  # ViT requires 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),  # ViT requires 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    }

def create_data_loaders(batch_size=BATCH_SIZE):
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
        trainset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=CLASSIFIER_NUM_EPOCHS, learning_rate=CLASSIFIER_LEARNING_RATE):
    """Train the classification model."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for different parts of the model
    # Smaller learning rate for pretrained backbone, larger for the new head
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params = [p for n, p in model.named_parameters() if "head" in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},
        {'params': head_params, 'lr': learning_rate}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_loss = float('inf')
    
    # Create classification checkpoint directory
    classification_dir = os.path.join(MODEL_SAVE_PATH, 'vit_classification_checkpoints')
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
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Save model if it's the best so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(classification_dir, 'best_vit_classifier.pth'))
            print("  Saved best model!")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(classification_dir, 'final_vit_classifier.pth'))
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'vit_classification_checkpoints', 'confusion_matrix.png'))
    
    return test_loss, test_accuracy

def save_metrics(weights_type, final_pretrain_loss, test_loss, test_accuracy):
    """Save metrics to a file."""
    metrics_file = os.path.join(MODEL_SAVE_PATH, 'vit_classification_checkpoints', f'final_metrics_{weights_type}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"=== ViT Classification Results (Weights: {weights_type}) ===\n\n")
        f.write(f"Pretrain Final Loss: {final_pretrain_loss:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, weights_type):
    """Plot and save training metrics."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'vit_classification_checkpoints', f'training_metrics_{weights_type}.png'))
    plt.close()

def main():
    """Main function to run the classification training."""
    # Parse command line arguments
    args = parse_args()
    
    # Create save directories
    os.makedirs(os.path.join(MODEL_SAVE_PATH, 'vit_classification_checkpoints'), exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=args.batch_size)
    
    # Create model with specified weights
    model = create_model(
        weights_type=args.weights,
        mae_checkpoint_path=args.mae_checkpoint
    )
    
    # Train model
    print(f"Training ViT classifier with {args.weights} weights...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, args.weights)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, 'vit_classification_checkpoints', 'best_vit_classifier.pth')))
    
    # Evaluate model
    print("Evaluating ViT classifier...")
    test_loss, test_accuracy = evaluate_model(model, test_loader)
    
    # Try to read the pretrain loss
    try:
        with open(os.path.join(MODEL_SAVE_PATH, 'vit_pretrain_checkpoints', 'final_vit_pretrain_loss.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Best Val Loss:" in line:
                    final_pretrain_loss = float(line.split(":")[1].strip())
                    break
            else:
                final_pretrain_loss = 0.0
    except:
        final_pretrain_loss = 0.0
    
    # Save metrics
    save_metrics(args.weights, final_pretrain_loss, test_loss, test_accuracy)
    
    print(f"Classification training and evaluation complete with {args.weights} weights!")

if __name__ == "__main__":
    main()
