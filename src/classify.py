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
    if num_classes is None:
        num_classes = NUM_CLASSES[CLASSIFY_DATASET_NAME]

    if ENCODER_MODEL == "vit_base_p16":
        encoder_model = "vit_base_patch16_224"
    elif ENCODER_MODEL == "vit_large_p16":
        encoder_model = "vit_large_patch16_224"
    elif ENCODER_MODEL == "vit_huge_p14":
        encoder_model = "vit_huge_patch14_224"
    else:
        raise ValueError(f"Unknown encoder model: {ENCODER_MODEL}")

    if weights_path is None:
        print(f"Loading PyTorch non-pretrained {ENCODER_MODEL} with {num_classes} classes.")
        model = timm.create_model(
            model_name=encoder_model,
            pretrained=False,
            num_classes=num_classes
        )
    else:
        print(f"Creating base model {ENCODER_MODEL} for feature extraction (loading from path: {weights_path})")
        feature_extractor = timm.create_model(
            model_name=encoder_model,
            pretrained=False,
            num_classes=0,
            global_pool=''
        )
        print(f"Loading pretrained weights from checkpoint: {weights_path}")
        weights = torch.load(weights_path, map_location=DEVICE)
        if any(k.startswith("encoder.") for k in weights.keys()):
            print("Detected MAE-style encoder weights (prefixed with 'encoder.'). Extracting and loading.")
            encoder_weights = {k.replace("encoder.", ""): v for k, v in weights.items() if k.startswith("encoder.")}
            feature_extractor.load_state_dict(encoder_weights, strict=False)
        elif 'model' in weights:
            feature_extractor.load_state_dict(weights['model'], strict=False)
        elif 'state_dict' in weights:
            feature_extractor.load_state_dict(weights['state_dict'], strict=False)
        else:
            feature_extractor.load_state_dict(weights, strict=False)
        try:
            in_features = feature_extractor.num_features
        except AttributeError:
            temp_model_for_features = timm.create_model(encoder_model, pretrained=False, num_classes=1)
            in_features = temp_model_for_features.num_features
            del temp_model_for_features
            print(f"Inferred in_features as {in_features} for {encoder_model}.")
        model = nn.Sequential(
            feature_extractor,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )
    return model

def get_data_transforms():
    return {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    }

def create_data_loaders():
    data_transforms = get_data_transforms()
    base_path = DATASET_PATHS[CLASSIFY_DATASET_NAME]
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
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
    num_found_classes = len(train_dataset.classes)
    num_expected_classes = NUM_CLASSES[CLASSIFY_DATASET_NAME]
    if num_found_classes != num_expected_classes:
        print(f"Warning: Found {num_found_classes} classes, but expected {num_expected_classes}.")
        print(f"Found classes: {train_dataset.classes}")
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    seed = torch.Generator().manual_seed(64)
    trainset, valset = random_split(train_dataset, [train_size, val_size], generator=seed)
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
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_loss = float('inf')
    classification_dir = CLASSIFICATION_FOLDER
    os.makedirs(classification_dir, exist_ok=True)
    for epoch in range(num_epochs):
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
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
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
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], ")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, ")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(classification_dir, 'best_model.pth'))
            print("Best model saved!")
        metrics_dict = {
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        }
        results_file = os.path.join(classification_dir, 'training_metrics.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        data.append(metrics_dict)
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=4)
        scheduler.step(epoch_val_loss)
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, test_loader):
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
    cm = confusion_matrix(all_labels, all_predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    if CLASSIFY_DATASET_NAME == "rafdb":
        classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    elif CLASSIFY_DATASET_NAME == "affectnet":
        classes = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    else:
        classes = test_loader.dataset.classes
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
    from src.utils.visualization import format_config_params
    metrics_file = os.path.join(CLASSIFICATION_FOLDER, 'final_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(format_config_params())
        f.write("\n=== Final Training Metrics ===\n\n")
        f.write("Pretraining:\n")
        f.write(f"Final Loss: {final_pretrain_loss:.4f}\n\n")
        f.write("Classification:\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    metrics_dir = os.path.join(CLASSIFICATION_FOLDER, "metrics_plots")
    os.makedirs(metrics_dir, exist_ok=True)
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
    import argparse
    parser = argparse.ArgumentParser(description='Train a classifier with optional pretrained checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to a pretrained checkpoint to use for initialization')
    args = parser.parse_args()
    classification_dir = CLASSIFICATION_FOLDER
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(os.path.join(classification_dir, 'metrics_plots'), exist_ok=True)
    model = create_model(weights_path=args.checkpoint)
    train_loader, val_loader, test_loader = create_data_loaders()
    print("Training classifier...")
    metrics = train_model(model, train_loader, val_loader)
    train_losses, train_accuracies, val_losses, val_accuracies = metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = evaluate_model(model, test_loader)
    try:
        with open(os.path.join(PRETRAIN_FOLDER, 'final_pretrain_loss.txt'), 'r') as f:
            final_pretrain_loss = float(f.read().strip())
    except (FileNotFoundError, ValueError):
        final_pretrain_loss = float('nan')
    save_metrics(final_pretrain_loss, test_loss, test_accuracy)

if __name__ == "__main__":
    main()
