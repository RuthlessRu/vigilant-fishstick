import os
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import hashlib
from collections import Counter
import pandas as pd

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

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

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


def parse_tess(dataset_dir):
    """
    Parse the TESS dataset and return file paths and labels.

    Args:
        dataset_dir (str): Path to the TESS dataset.

    Returns:
        List of tuples: (file_path, emotion_label)
    """
    emotion_map = {
        "neutral": 0,   # Neutral
        "happy": 1,     # Happy
        "sad": 2,       # Sad
        "angry": 3,     # Angry
        "fear": 4,      # Fear
        "disgust": 5,   # Disgust
        "pleasant": 6,  # Surprised (to be removed)
    }

    file_paths = []
    labels = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                # Extract the emotion from the file path or filename
                emotion = os.path.basename(root).lower()  # Folder name
                if "_" in emotion:
                    emotion = emotion.split("_")[1]  # Extract emotion part
                
                if emotion == "pleasant":
                    continue
                
                if emotion in emotion_map:
                    file_paths.append(os.path.join(root, file))
                    labels.append(emotion_map[emotion])

    return file_paths, labels

def parse_ravdess(dataset_dir, gender="female", vocal_channel="speech"):
    """
    Parse the RAVDESS dataset and filter based on gender and modality.

    Args:
        dataset_dir (str): Path to the RAVDESS dataset.
        gender (str): "female" or "male" to filter actors.
        modality (str): "speech" or "song" to filter modality.

    Returns:
        List of tuples: (file_path, emotion_label)
    """
    emotion_map = {
    1: 0,  # Neutral
    3: 1,  # Happy
    4: 2,  # Sad
    5: 3,  # Angry
    6: 4,  # Fear
    7: 5,  # Disgust
    }
    gender_map = {"male": lambda x: int(x) % 2 == 1, "female": lambda x: int(x) % 2 == 0}
    vocal_channel_map = {"speech": "01", "song": "02"}

    file_paths = []
    labels = []

    # Traverse the directory structure
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            if dir_name.startswith("Actor_"):  # Check for Actor_* directories
                actor_id = int(dir_name.split("_")[1])  # Extract actor ID
                if not gender_map[gender](actor_id):  # Filter by gender
                    continue

                actor_path = os.path.join(root, dir_name)
                for file_name in os.listdir(actor_path):
                    if file_name.endswith(".wav"):
                        components = file_name.split("-")
                        file_vocal_channel, emotion = components[1], components[2]
                        if file_vocal_channel != vocal_channel_map[vocal_channel]:  # Filter by modality
                            continue

                        if int(emotion) == 2 or int(emotion) == 8: # skip calm and surprised
                            continue

                        file_path = os.path.join(actor_path, file_name)
                        file_paths.append(file_path)
                        labels.append(emotion_map[int(emotion)])

    return file_paths, labels

def parse_cremad(dataset_dir, demographics_file):
    """
    Parse the CREMA-D dataset and filter for female actors.

    Args:
        dataset_dir (str): Path to the CREMA-D audio files.
        demographics_file (str): Path to VideoDemographics.csv.

    Returns:
        List of tuples: (file_path, emotion_label)
    """
    # Unified emotion map
    emotion_map = {
        "NEU": 0,  # Neutral
        "HAP": 1,  # Happy/Joy
        "SAD": 2,  # Sad
        "ANG": 3,  # Angry
        "FEA": 4,  # Fear
        "DIS": 5,  # Disgust
    }

    df = pd.read_csv(demographics_file)
    female_actor_ids = df[df["Sex"] == "Female"]["ActorID"].astype(str).tolist()

    file_paths = []
    labels = []

    # Traverse dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                actor_id, emotion_code = file.split("_")[0], file.split("_")[2]
                if actor_id in female_actor_ids and emotion_code in emotion_map:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                    labels.append(emotion_map[emotion_code])

    return file_paths, labels    


if __name__ == "__main__":
    cremad_dir = os.path.expanduser("~/.cache/kagglehub/datasets/ejlok1/cremad")
    ravdess_dir = os.path.expanduser("~/.cache/kagglehub/datasets/uwrfkaggler")
    tess_dir = os.path.expanduser("~/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess")

    cremad_dem_file = os.path.expanduser("~/.cache/kagglehub/VideoDemographics.csv")

    output_dir = "./preprocessed_data"

    train_output_dir = os.path.join(output_dir, "train")
    val_output_dir = os.path.join(output_dir, "val")
    test_output_dir = os.path.join(output_dir, "test")

    tess_file_paths, tess_labels = parse_tess(tess_dir)
    ravdess_file_paths, ravdess_labels = parse_ravdess(ravdess_dir)
    cremad_file_paths, cremad_labels = parse_cremad(cremad_dir, cremad_dem_file)

    # use tess and cremad for training
    training_file_paths = tess_file_paths + cremad_file_paths
    training_labels = tess_labels + cremad_labels

    testing_file_paths = ravdess_file_paths
    testing_labels = ravdess_labels

    label_encoder = LabelEncoder()
    encoded_training_labels = label_encoder.fit_transform(training_labels)
    encoded_testing_labels = label_encoder.transform(testing_labels)

    X_train, X_val, y_train, y_val = train_test_split(
        training_file_paths, encoded_training_labels, test_size=0.2, stratify=encoded_training_labels, random_state=42
    )

    # Hashing function to ensure no duplicate content
    def hash_file(filepath):
        """Generate a hash for a file to ensure unique contents."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    # Hash the contents of files in each split
    train_hashes = {hash_file(fp) for fp in X_train}
    val_hashes = {hash_file(fp) for fp in X_val}

    # Check for content overlaps
    assert train_hashes.isdisjoint(val_hashes), "Overlap in file contents between train and validation!"
    # Preprocess and save each dataset
    print("Preprocessing and saving training data...")
    preprocess_and_save(X_train, y_train, preprocess_audio, train_output_dir, device="cuda")

    print("Preprocessing and saving validation data...")
    preprocess_and_save(X_val, y_val, preprocess_audio, val_output_dir, device="cuda")

    print("Preprocessing and saving test data...")
    preprocess_and_save(testing_file_paths, encoded_testing_labels, preprocess_audio, test_output_dir, device="cuda")

    print("Preprocessing complete. Data saved to:", output_dir)
