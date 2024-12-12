import os
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def trim_silence(waveform, threshold=1e-4):
    if waveform.dim() > 1:
        reduced_waveform = waveform.mean(dim=0)  # Take the mean of channels
    else:
        reduced_waveform = waveform

    non_silent_indices = torch.where(reduced_waveform.abs() > threshold)[0]

    if len(non_silent_indices) == 0:
        return waveform

    start, end = non_silent_indices[0], non_silent_indices[-1] + 1
    return waveform[:, start:end] if waveform.dim() > 1 else waveform[start:end]

def extract_mel_spectrogram(waveform, sr=16000, n_mels=64, fmax=8000, hop_length=512, device="cuda"):
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        f_max=fmax,
        hop_length=hop_length,
        n_fft=1024
    ).to(device)
    mel_spectrogram = mel_spectrogram_transform(waveform)
    mel_spectrogram_db = T.AmplitudeToDB()(mel_spectrogram)
    return mel_spectrogram_db

def preprocess_and_save_all(file_paths, labels, preprocess_fn, output_dir, device="cuda"):
    """
    Preprocess the entire dataset and save mel spectrograms and labels.

    Args:
        file_paths: List of file paths to audio files.
        labels: List of corresponding labels.
        preprocess_fn: Function to preprocess audio.
        output_dir: Directory to save preprocessed data.
        device: Device for preprocessing (e.g., "cuda").
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths))):
        waveform, _ = preprocess_fn(file_path, device="cpu")  # Preprocess waveform on CPU
        waveform = waveform.to(device)  # Move to GPU for spectrogram
        mel_spectrogram = extract_mel_spectrogram(waveform, device=device)

        # Save mel spectrogram and label
        torch.save(
            {"mel_spectrogram": mel_spectrogram.cpu(), "label": torch.tensor(label)},
            os.path.join(output_dir, f"data_{idx}.pt"),
        )

    print(f"Saved {len(file_paths)} preprocessed files to {output_dir}")


def preprocess_audio(file_path, target_sr=16000, duration=2.5, device="cuda"):
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.to(device)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
        waveform = resampler(waveform)
    
    waveform = trim_silence(waveform)
    waveform = waveform / waveform.abs().max()

    max_length = int(target_sr * duration)
    if waveform.size(1) > max_length:
        waveform = waveform[:, :max_length]
    else:
        pad_length = max_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    return waveform, target_sr


if __name__ == "__main__":
    # Define your dataset directory and output directories
    dataset_dir = os.path.expanduser("~/.cache/kagglehub/datasets/ejlok1")
    output_dir = "./preprocessed_data"
    train_output_dir = os.path.join(output_dir, "train")
    val_output_dir = os.path.join(output_dir, "val")
    test_output_dir = os.path.join(output_dir, "test")

    # Initialize variables
    file_paths = []
    labels = []

    # Traverse the dataset directory to collect file paths and labels
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion = os.path.basename(root)  # Take the folder name as the emotion label
                file_paths.append(os.path.join(root, file))
                labels.append(emotion)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # One-hot encode labels
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(encoded_labels.reshape(-1, 1)).astype("float32")

    # Split the dataset into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        file_paths, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
    )

    # Preprocess and save each dataset
    print("Preprocessing and saving training data...")
    preprocess_and_save_all(X_train, y_train, preprocess_audio, train_output_dir, device="cuda")

    print("Preprocessing and saving validation data...")
    preprocess_and_save_all(X_val, y_val, preprocess_audio, val_output_dir, device="cuda")

    print("Preprocessing and saving test data...")
    preprocess_and_save_all(X_test, y_test, preprocess_audio, test_output_dir, device="cuda")

    print("Preprocessing complete. Data saved to:", output_dir)
