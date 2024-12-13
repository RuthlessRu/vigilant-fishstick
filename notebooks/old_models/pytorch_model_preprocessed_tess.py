import os
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
#import torchlibrosa as tl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T
import kagglehub
from multiprocessing import Lock
import time
import logging
from torch.utils.data import Dataset, DataLoader


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

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim):
        """
        Initializes the attention mechanism.

        Args:
            input_dim (int): The dimensionality of the input features.
        """
        super(AttentionMechanism, self).__init__()
        self.attention_layer = nn.Linear(input_dim, 1) 
        self.tanh = nn.Tanh()
        
    def forward(self, inputs):
        """
        Forward pass for the attention mechanism.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Weighted input tensor of the same shape as inputs.
        """
        # Apply linear transformation and tanh activation
        attention = self.tanh(self.attention_layer(inputs))  # Shape: (batch_size, sequence_length, 1)
        
        # Flatten and apply softmax along the sequence dimension
        attention = F.softmax(attention.squeeze(-1), dim=-1)  # Shape: (batch_size, sequence_length)
        
        # Expand dimensions to match inputs for element-wise multiplication
        attention = attention.unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)
        
        # Compute the weighted inputs
        weighted_inputs = inputs * attention  # Element-wise multiplication
        return weighted_inputs
    
class ConvRNNWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(ConvRNNWithAttention, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)

        # Placeholder for LSTMs
        self.rnn1 = None
        self.rnn2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)

        # Attention mechanism
        self.attention = AttentionMechanism(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)

        # Compute reshape_dim dynamically
        batch_size, channels, height, width = x.size()
        reshape_dim = channels * height * width

        # Reshape for RNN input
        x = x.view(batch_size, -1, reshape_dim)

        # Dynamically initialize the first RNN layer
        if self.rnn1 is None:
            self.rnn1 = nn.LSTM(reshape_dim, 128, bidirectional=True, batch_first=True).to(x.device)

        # Bi-directional LSTMs
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)

        # Attention mechanism
        x = self.attention(x)

        # Fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)


def train_model(model, train_loader, val_loader, criterion, optimizer, device="cuda", epochs=30):
    model.to(device)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for i, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()
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

            logging.info(f"Epoch {epoch + 1}, Batch {i + 1}: Took {time.time() - batch_start_time:.4f}s")

        train_accuracy = correct / total
        if val_loader:
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
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
        else:
            val_loss, val_accuracy = None, None
        logging.info(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.4f}s")
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}")
        if val_loader:
            print(f"  Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")

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
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Ensure proper multiprocessing
    mp.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set device
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Current device: {device}")

    # Load preprocessed datasets
    train_dataset = PreprocessedDataset("./preprocessed_data/train")
    val_dataset = PreprocessedDataset("./preprocessed_data/val")
    test_dataset = PreprocessedDataset("./preprocessed_data/test")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize and train model
    model = ConvRNNWithAttention(num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, device=device, epochs=30)

    # Evaluate the model
    evaluate_model(model, test_loader, criterion, device=device)