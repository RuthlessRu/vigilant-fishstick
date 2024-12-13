from torchvision import models
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T
import kagglehub
from multiprocessing import Lock
import time
import logging
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def create_baseline_model(num_classes):
    # load a pre-trained resnet
    model = models.resnet18(pretrained=True)
    # modify so we can accept a mel-spec
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # obvs modify so its the number of our classes (which is 6)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device="cuda", epochs=30):
    """Training our model
    """
    model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = 0.0, 0
        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_accuracy = correct / total
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_accuracy)

            scheduler.step(val_loss / len(val_loader))

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict()

        print(f"epoch {epoch + 1}/{epochs}: train Loss: {train_losses[-1]:.4f}, train Acc: {train_accuracies[-1]:.4f}")
        if val_loader:
            print(f"  val loss: {val_losses[-1]:.4f}, val acc: {val_accuracies[-1]:.4f}")

    # load best model
    if best_model_state is not None:
        print(f"loading model weights from epoch w/ best validation acc: {best_val_accuracy:.4f}")
        model.load_state_dict(best_model_state)

    # plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    if val_loader:
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    if val_loader:
        plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, criterion, device="cuda"):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"test loss: {test_loss / len(test_loader):.4f}, test accuracy: {accuracy:.4f}")

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing preprocessed .pt files.
        """
        self.data_dir = data_dir
        self.file_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        return data["mel_spectrogram"], data["label"]

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"current device: {device}")

    # load up the preprocessed data
    train_dataset = PreprocessedDataset("./preprocessed_data/train")
    val_dataset = PreprocessedDataset("./preprocessed_data/val")
    test_dataset = PreprocessedDataset("./preprocessed_data/test")

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # model
    model = create_baseline_model(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device=device, epochs=30)

    # eval
    evaluate_model(model, test_loader, criterion, device=device)