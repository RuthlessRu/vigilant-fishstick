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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset


class PreprocessedDataset(Dataset):
    def __init__(self, data_dir):
        """
        takes in .pt files that are preprocessed.
        """
        self.data_dir = data_dir
        self.file_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        return data["mel_spectrogram"], data["label"]

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim):
        """
        attention mechanism.
        """
        super(AttentionMechanism, self).__init__()
        self.attention_layer = nn.Linear(input_dim, 1) 
        self.tanh = nn.Tanh()
        
    def forward(self, inputs):
        """
        Forward pass
        inputs.shape (batch_size, sequence_length, input_dim).
        """
        attention = self.tanh(self.attention_layer(inputs))  # shape: (batch_size, sequence_length, 1)
        attention = F.softmax(attention.squeeze(-1), dim=-1)  # shape: (batch_size, sequence_length)
        attention = attention.unsqueeze(-1)  # shape: (batch_size, sequence_length, 1)
        weighted_inputs = inputs * attention  # element-wise
        return weighted_inputs
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ConvNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetWithAttention, self).__init__()
        self.layer1 = ResidualBlock(1, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = AttentionMechanism(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # conv layers w residual connections
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # global pooling (avg)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.attention(x.unsqueeze(1))
        x = torch.mean(x, dim=1)

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device="cuda", epochs=30):
    """ train the model and display confusion matrix for highest validation accuracy """
    model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0
    best_model_state = None
    best_val_preds = []
    best_val_labels = []

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
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_accuracy = correct / total
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_accuracy)

            scheduler.step(val_loss / len(val_loader))

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict()
                best_val_preds = all_preds
                best_val_labels = all_labels

        print(f"epoch {epoch + 1}/{epochs}: train loss: {train_losses[-1]:.4f}, train acc: {train_accuracies[-1]:.4f}")
        if val_loader:
            print(f"  val loss: {val_losses[-1]:.4f}, val acc: {val_accuracies[-1]:.4f}")

    if best_model_state is not None:
        print(f"Loading model weights from epoch with highest validation accuracy: {best_val_accuracy:.4f}")
        model.load_state_dict(best_model_state)

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

    if best_val_preds and best_val_labels:
        cm = confusion_matrix(best_val_labels, best_val_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(best_val_labels))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix for Highest Validation Accuracy")
        plt.show()


def evaluate_model(model, test_loader, criterion, device="cuda"):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.4f}")

    # compute and display confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_labels))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"current device: {device}")

    train_crema_tess = PreprocessedDataset("./preprocessed_data/train")
    test_ravdess = PreprocessedDataset("./preprocessed_data/test")

    # combine
    combined_dataset = ConcatDataset([train_crema_tess, test_ravdess])

    # Determine new split sizes
    total_size = len(combined_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # shuffle
    new_train_dataset, new_val_dataset, new_test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

    batch_size = 32
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(new_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(new_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = ConvNetWithAttention(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device="cuda", epochs=30)

    # evaluate
    evaluate_model(model, test_loader, criterion, device=device)