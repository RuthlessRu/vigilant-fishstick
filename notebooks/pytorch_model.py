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

from torch.utils.data import Dataset, DataLoader

def trim_silence(waveform, threshold=1e-4):
    """
    Trims silence from the waveform based on a threshold.
    """
    # Reduce multi-channel waveform to a single channel for silence detection
    if waveform.dim() > 1:
        reduced_waveform = waveform.mean(dim=0)  # Take the mean of channels
    else:
        reduced_waveform = waveform

    # Find non-silent indices
    non_silent_indices = torch.where(reduced_waveform.abs() > threshold)[0]
    
    # If no non-silent indices are found, return the original waveform
    if len(non_silent_indices) == 0:
        return waveform

    # Trim waveform to the range of non-silent indices
    start, end = non_silent_indices[0], non_silent_indices[-1] + 1
    return waveform[:, start:end] if waveform.dim() > 1 else waveform[start:end]

def preprocess_audio(file_path, target_sr=16000, duration=2.5, device="cuda"):
    """
    Preprocesses audio data for PyTorch.

    Args:
        file_path: Path to the audio file.
        target_sr: Target sampling rate.
        duration: Desired duration of the audio clip in seconds.
        device: Device to perform processing (e.g., "cuda" or "cpu").

    Returns:
        Preprocessed audio tensor.
    """
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.to(device)

    if sr != target_sr:
        # Move resampler to the same device as waveform
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
        waveform = resampler(waveform)

    # Trim silence
    waveform = trim_silence(waveform)

    # Normalize amplitude
    waveform = waveform / waveform.abs().max()

    # Pad or truncate to the target duration
    max_length = int(target_sr * duration)
    if waveform.size(1) > max_length:
        waveform = waveform[:, :max_length]  # Truncate
    else:
        pad_length = max_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))  # Pad

    return waveform, target_sr

def extract_mel_spectrogram(waveform, sr=16000, n_mels=128, fmax=8000, hop_length=512, device="cuda"):
    # Define the MelSpectrogram transform
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        f_max=fmax,
        hop_length=hop_length
    ).to(device)

    # Apply transform
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to decibels (similar to librosa.power_to_db)
    mel_spectrogram_db = T.AmplitudeToDB()(mel_spectrogram)

    return mel_spectrogram_db

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

        # Preprocess the audio file
        waveform, _ = self.preprocess_fn(file_path, device=self.device)

        # Extract mel spectrogram
        mel_spectrogram = extract_mel_spectrogram(waveform, device=self.device)

        return mel_spectrogram, torch.tensor(label, device=self.device)

def create_pytorch_dataloader(file_paths, labels, preprocess_fn, batch_size=32, shuffle=True, device="cuda"):
    """
    Creates a PyTorch DataLoader for audio data.

    Args:
        file_paths: List of file paths to audio files.
        labels: List of corresponding labels.
        preprocess_fn: Function for preprocessing audio.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
        device: Device to perform processing (e.g., "cuda" or "cpu").

    Returns:
        PyTorch DataLoader object.
    """
    dataset = AudioDataset(file_paths, labels, preprocess_fn, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
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
    def __init__(self, input_shape, num_classes):
        super(ConvRNNWithAttention, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Conv2D(32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Conv2D(64, (3, 3), padding='same')

        self.pool = nn.MaxPool2d(2)  # MaxPooling2D((2, 2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)

        # Reshape for RNN
        self.reshape_dim = (input_shape[0] // 4) * (input_shape[1] // 4) * 64

        # Bi-directional LSTMs
        self.rnn1 = nn.LSTM(self.reshape_dim, 128, bidirectional=True, batch_first=True)  # LSTM(128, return_sequences=True)
        self.rnn2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)  # LSTM(64, return_sequences=True)

        # Attention mechanism
        self.attention = AttentionMechanism(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)  # Dense(64)
        self.fc2 = nn.Linear(64, num_classes)  # Dense(num_classes)

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

        # Reshape for RNN input
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.reshape_dim)

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
        model.train()  # Set model to training mode
        train_loss, correct, total = 0.0, 0, 0
        
        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)
        
        train_accuracy = correct / total
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
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
        
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}")
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

def main():
    # Initialize model, loss function, and optimizer
    model = ConvRNNWithAttention(input_shape=(128, 79, 1), num_classes=y_train.shape[1])
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss as the criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoaders
    train_loader = create_pytorch_dataloader(X_train, y_train, preprocess_audio, batch_size=32, shuffle=True, device="cuda")
    val_loader = create_pytorch_dataloader(X_val, y_val, preprocess_audio, batch_size=32, shuffle=False, device="cuda")
    test_loader = create_pytorch_dataloader(X_test, y_test, preprocess_audio, batch_size=32, shuffle=False, device="cuda")

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device="cuda", epochs=1)

    # Evaluate the model
    evaluate_model(model, test_loader, criterion, device="cuda")

if __name__ == '__main__':
    lock = Lock()
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device("cuda")
    print("Current device:", device)


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
                emotion = os.path.basename(root) # take base name as emotion
                emotions.append(emotion)
                file_paths.append(os.path.join(root, file))

    labels = [label.lower().split('_')[1] if '_' in label else label.lower() for label in emotions]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_onehot = onehot_encoder.fit_transform(encoded_labels.reshape(-1, 1))
    y_onehot = y_onehot.astype(np.float32)
    y_onehot.shape


    X_train_val, X_test, y_train_val, y_test = train_test_split(
        file_paths, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
    )

    dataloader = create_pytorch_dataloader(file_paths, labels, preprocess_audio, batch_size=32, shuffle=True, device="cuda")

    train_loader = create_pytorch_dataloader(X_train, y_train, preprocess_audio, batch_size=32, shuffle=True, device="cuda")
    val_loader = create_pytorch_dataloader(X_val, y_val, preprocess_audio, batch_size=32, shuffle=False, device="cuda")
    test_loader = create_pytorch_dataloader(X_test, y_test, preprocess_audio, batch_size=32, shuffle=False, device="cuda")

    mp.set_start_method("spawn", force=True)

    main()