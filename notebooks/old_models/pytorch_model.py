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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, preprocess_fn, device="cuda"):
        self.file_paths = file_paths
        self.labels = labels
        self.preprocess_fn = preprocess_fn
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Preprocess waveform on CPU
        waveform, _ = self.preprocess_fn(file_path, device="cpu")  # Resampling, silence trimming, etc.
        waveform = waveform.to(self.device)  # Move waveform to GPU

        # Extract mel spectrogram on GPU
        mel_spectrogram = extract_mel_spectrogram(waveform, device=self.device)

        return mel_spectrogram, torch.tensor(label, device=self.device)

def create_pytorch_dataloader(file_paths, labels, preprocess_fn, batch_size=32, shuffle=True, device="cuda"):
    dataset = AudioDataset(file_paths, labels, preprocess_fn, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=False)
    return dataloader

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

def trim_silence(waveform, threshold=1e-4):
    start_time = time.time()
    if waveform.dim() > 1:
        reduced_waveform = waveform.mean(dim=0)  # Take the mean of channels
    else:
        reduced_waveform = waveform

    non_silent_indices = torch.where(reduced_waveform.abs() > threshold)[0]

    if len(non_silent_indices) == 0:
        logging.info(f"Trimming silence: All silent. Took {time.time() - start_time:.4f}s")
        return waveform

    start, end = non_silent_indices[0], non_silent_indices[-1] + 1
    logging.info(f"Trimming silence: Took {time.time() - start_time:.4f}s")
    return waveform[:, start:end] if waveform.dim() > 1 else waveform[start:end]

def preprocess_audio(file_path, target_sr=16000, duration=2.5, device="cuda"):
    start_time = time.time()
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.to(device)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
        waveform = resampler(waveform)
    
    logging.info(f"Resampling {file_path}: Took {time.time() - start_time:.4f}s")

    waveform = trim_silence(waveform)
    waveform = waveform / waveform.abs().max()

    max_length = int(target_sr * duration)
    if waveform.size(1) > max_length:
        waveform = waveform[:, :max_length]
    else:
        pad_length = max_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    logging.info(f"Preprocessing {file_path}: Took {time.time() - start_time:.4f}s")
    return waveform, target_sr

def extract_mel_spectrogram(waveform, sr=16000, n_mels=64, fmax=8000, hop_length=512, device="cuda"):
    start_time = time.time()
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        f_max=fmax,
        hop_length=hop_length,
        n_fft=1024
    ).to(device)
    mel_spectrogram = mel_spectrogram_transform(waveform)
    mel_spectrogram_db = T.AmplitudeToDB()(mel_spectrogram)
    logging.info(f"Mel spectrogram extraction: Took {time.time() - start_time:.4f}s")
    return mel_spectrogram_db

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
            correct += (preds == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)

            logging.info(f"Epoch {epoch + 1}, Batch {i + 1}: Took {time.time() - batch_start_time:.4f}s")

        train_accuracy = correct / total

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels.argmax(dim=1)).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        logging.info(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.4f}s")
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}")
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
            correct += (preds == labels.argmax(dim=1)).sum().item()
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

    # Load dataset
    cache_dir = os.path.expanduser("~/.cache/kagglehub/datasets/ejlok1")

    # Check if the dataset exists
    if not os.path.exists(cache_dir):
        print("Downloading TESS dataset...")
        tess_path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
    else:
        print("TESS dataset already exists.")
        tess_path = cache_dir

    emotions = []
    file_paths = []

    for root, dirs, files in os.walk(tess_path):
        for file in files:
            if file.endswith('.wav'):
                emotion = os.path.basename(root)
                emotions.append(emotion)
                file_paths.append(os.path.join(root, file))

    labels = [label.lower().split('_')[1] if '_' in label else label.lower() for label in emotions]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_onehot = onehot_encoder.fit_transform(encoded_labels.reshape(-1, 1)).astype(np.float32)

    # Train-test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        file_paths, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
    )

    # Create DataLoaders
    train_loader = create_pytorch_dataloader(X_train, y_train, preprocess_audio, batch_size=16, shuffle=True, device="cuda")
    val_loader = create_pytorch_dataloader(X_val, y_val, preprocess_audio, batch_size=16, shuffle=False, device="cuda")
    test_loader = create_pytorch_dataloader(X_test, y_test, preprocess_audio, batch_size=16, shuffle=False, device="cuda")

    # Initialize and train model
    model = ConvRNNWithAttention(num_classes=y_train.shape[1])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, device=device, epochs=1)

    # Evaluate the model
    evaluate_model(model, test_loader, criterion, device=device)