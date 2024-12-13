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
        attention = attention.unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)
        weighted_inputs = inputs * attention  # element-wise
        return weighted_inputs
    
class ConvRNNWithAttention(nn.Module):
    """
    Convolutional Recurrent Neural Network

    Here, we use 2 convolutional layers
    2 rnn layers that are bidirectional LSTMs
    we have an attention layer near the end
    as well as a fully connected layer
    """
    def __init__(self, num_classes):
        super(ConvRNNWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        self.rnn1 = None #dynamically fill out in forwar
        self.rnn2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.attention = AttentionMechanism(128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # cnn part
        x = (self.conv1(x))
        x = F.relu(self.batch_norm1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = (self.conv2(x))
        x = F.relu(self.batch_norm2(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        batch_size, channels, height, width = x.size()
        reshape_dim = channels * height * width
        x = x.view(batch_size, -1, reshape_dim)

        if self.rnn1 is None: #here we dynamically shape this
            self.rnn1 = nn.LSTM(reshape_dim, 128, bidirectional=True, batch_first=True).to(x.device)

        # lstms
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)

        # attention part
        x = self.attention(x)

        # fully connected part
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device="cuda", epochs=30):
    """ training the model as well as graphs.
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

        print(f"epoch {epoch + 1}/{epochs}: train loss: {train_losses[-1]:.4f}, train acc: {train_accuracies[-1]:.4f}")
        if val_loader:
            print(f"  val loss: {val_losses[-1]:.4f}, val acc: {val_accuracies[-1]:.4f}")

    # load best model
    if best_model_state is not None:
        print(f"loading model weights from epoch w/ best validation acc: {best_val_accuracy:.4f}")
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
    print(f"test loss: {test_loss / len(test_loader):.4f}, test acc: {accuracy:.4f}")

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
    model = ConvRNNWithAttention(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device=device, epochs=30)

    # evaluate
    evaluate_model(model, test_loader, criterion, device=device)