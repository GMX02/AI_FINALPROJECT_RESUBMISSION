import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIGURATION ===
DATASET_DIR = 'edge-collected-gunshot-audio'  # Path to folder with subfolders
LABELS_FILE = 'gunshot-audio-labels-only.csv'  # CSV file with labels
OUTPUT_X = 'X.npy'
OUTPUT_Y = 'y.npy'

SAMPLE_RATE = 22050
DURATION = 2.0
N_MELS = 128
EXPECTED_WIDTH = 87  # Roughly 2 seconds with default hop length

def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < sr * DURATION:
        y = np.pad(y, (0, int(sr * DURATION) - len(y)))  # Pad if too short
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to standard shape
    if mel_db.shape[1] < EXPECTED_WIDTH:
        mel_db = np.pad(mel_db, ((0, 0), (0, EXPECTED_WIDTH - mel_db.shape[1])))
    elif mel_db.shape[1] > EXPECTED_WIDTH:
        mel_db = mel_db[:, :EXPECTED_WIDTH]
    return mel_db

def build_dataset(dataset_dir, labels_csv):
    labels_df = pd.read_csv(labels_csv)

    # Clean column names
    labels_df.columns = labels_df.columns.str.strip().str.lower()
    print("CSV columns:", labels_df.columns.tolist())

    # Check for 'num_gunshots' column
    if 'num_gunshots' not in labels_df.columns or 'filename' not in labels_df.columns:
        raise KeyError("Expected 'num_gunshots' and 'filename' columns not found in CSV.")

    # Create label column
    labels_df['label'] = (labels_df['num_gunshots'] > 0).astype(int)
    label_map = dict(zip(labels_df['filename'], labels_df['label']))

    X = []
    y = []

    print("Processing audio files...")
    for root, _, files in os.walk(dataset_dir):
        for file in tqdm(files):
            if file.endswith('.wav'):
                base = os.path.splitext(file)[0]
                if base not in label_map:
                    continue  # Skip files with no label
                label = label_map[base]
                file_path = os.path.join(root, file)
                try:
                    mel = extract_mel_spectrogram(file_path)
                    X.append(mel)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == '__main__':
    print("Building dataset using CSV labels...")
    X, y = build_dataset(DATASET_DIR, LABELS_FILE)

    print(f"Saving {len(X)} samples...")
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    print("Done! Shapes:")
    print("X:", X.shape)
    print("y:", y.shape)
