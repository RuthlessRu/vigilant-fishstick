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

def preprocess_and_save(file_paths, labels, preprocess_fn, output_dir, device="cuda", subset_size=None):
    """
    Preprocess the dataset (or a subset) and save mel spectrograms and labels.

    Args:
        file_paths: List of file paths to audio files.
        labels: List of corresponding labels.
        preprocess_fn: Function to preprocess audio.
        output_dir: Directory to save preprocessed data.
        device: Device for preprocessing (e.g., "cuda").
        subset_size: If provided, only preprocess this many samples.
    """
    os.makedirs(output_dir, exist_ok=True)

    if subset_size is not None:
        file_paths = file_paths[:subset_size]
        labels = labels[:subset_size]
        print(f"Processing {subset_size} samples...")

    for idx, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths))):
        waveform, _ = preprocess_fn(file_path, device="cpu")  # Preprocess waveform on CPU
        waveform = waveform.to(device)  # Move to GPU for spectrogram
        mel_spectrogram = extract_mel_spectrogram(waveform, device=device)

        print(f"Saving file {file_path}, label: {label}")

        # Save mel spectrogram and class index label
        torch.save(
            {"mel_spectrogram": mel_spectrogram.cpu(), "label": torch.tensor(label)},
            os.path.join(output_dir, f"data_{idx}.pt"),
        )

    print(f"Saved {len(file_paths)} preprocessed samples to {output_dir}")


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
    emotions = []

    # Traverse the dataset directory to collect file paths and labels
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion = os.path.basename(root)  # Take the folder name as the emotion label
                file_paths.append(os.path.join(root, file))
                emotions.append(emotion)

    # Encode labels
    labels = [label.lower().split('_')[1] if '_' in label else label.lower() for label in emotions]
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

    from collections import defaultdict

    def group_by_class(file_paths, labels):
        class_dict = defaultdict(list)
        for file_path, label in zip(file_paths, labels):
            class_dict[label.argmax()].append(file_path)
        return class_dict
    def create_balanced_subset(file_paths, labels, num_per_class=1):
        class_dict = group_by_class(file_paths, labels)
        subset_file_paths = []
        subset_labels = []

        for class_label, files in class_dict.items():
            selected_files = files[:num_per_class]  # Select `num_per_class` files per class
            subset_file_paths.extend(selected_files)
            subset_labels.extend([class_label] * len(selected_files))

        return subset_file_paths, subset_labels

    small_train_dir = "./preprocessed_data/small_train"
    small_val_dir = "./preprocessed_data/small_val"
    small_test_dir = "./preprocessed_data/small_test"

    os.makedirs(small_train_dir, exist_ok=True)
    os.makedirs(small_val_dir, exist_ok=True)
    os.makedirs(small_test_dir, exist_ok=True)
    
    small_train_paths, small_train_labels = create_balanced_subset(X_train, y_train, num_per_class=10)
    small_val_paths, small_val_labels = create_balanced_subset(X_val, y_val, num_per_class=5)
    small_test_paths, small_test_labels = create_balanced_subset(X_test, y_test, num_per_class=5)

    # print(small_test_labels[:50])

    #     # Check for overlap between splits
    # assert set(X_train).isdisjoint(X_val), "Overlap found between train and validation splits!"
    # assert set(X_train).isdisjoint(X_test), "Overlap found between train and test splits!"
    # assert set(X_val).isdisjoint(X_test), "Overlap found between validation and test splits!"

    # # Check small subsets for overlaps
    # assert set(small_train_paths).isdisjoint(small_val_paths), "Overlap found between small train and validation!"
    # assert set(small_train_paths).isdisjoint(small_test_paths), "Overlap found between small train and test!"
    # assert set(small_val_paths).isdisjoint(small_test_paths), "Overlap found between small validation and test!"

    # from collections import Counter
    # print(Counter(small_train_labels))
    # print(Counter(small_val_labels))
    # print(Counter(small_test_labels))

    # import hashlib

    # def hash_file(filepath):
    #     """Generate a hash for a file to ensure unique contents."""
    #     hasher = hashlib.md5()
    #     with open(filepath, 'rb') as f:
    #         buf = f.read()
    #         hasher.update(buf)
    #     return hasher.hexdigest()

    # # Hash the contents of files in each split
    # train_hashes = {hash_file(fp) for fp in small_train_paths}
    # val_hashes = {hash_file(fp) for fp in small_val_paths}
    # test_hashes = {hash_file(fp) for fp in small_test_paths}

    # overlap_hashes = train_hashes.intersection(val_hashes)
    # print(f"Overlapping hashes: {len(overlap_hashes)}")
    # for hash_val in overlap_hashes:
    #     print(f"Files in train with hash {hash_val}: {[fp for fp in small_train_paths if hash_file(fp) == hash_val]}")
    #     print(f"Files in val with hash {hash_val}: {[fp for fp in small_val_paths if hash_file(fp) == hash_val]}")


    # # Check for content overlaps
    # assert train_hashes.isdisjoint(val_hashes), "Overlap in file contents between train and validation!"
    # assert train_hashes.isdisjoint(test_hashes), "Overlap in file contents between train and test!"
    # assert val_hashes.isdisjoint(test_hashes), "Overlap in file contents between validation and test!"


    print("Preprocessing and saving small training data...")
    preprocess_and_save(small_train_paths, small_train_labels, preprocess_audio, small_train_dir, device="cuda")

    print("Preprocessing and saving small validation data...")
    preprocess_and_save(small_val_paths, small_val_labels, preprocess_audio, small_val_dir, device="cuda")

    print("Preprocessing and saving small test data...")
    preprocess_and_save(small_test_paths, small_test_labels, preprocess_audio, small_test_dir, device="cuda")

    print("Small dataset preprocessing complete. Data saved to:", output_dir)

    # # Preprocess and save each dataset
    # print("Preprocessing and saving training data...")
    # preprocess_and_save_all(X_train, y_train, preprocess_audio, train_output_dir, device="cuda")

    # print("Preprocessing and saving validation data...")
    # preprocess_and_save_all(X_val, y_val, preprocess_audio, val_output_dir, device="cuda")

    # print("Preprocessing and saving test data...")
    # preprocess_and_save_all(X_test, y_test, preprocess_audio, test_output_dir, device="cuda")

    # print("Preprocessing complete. Data saved to:", output_dir)
