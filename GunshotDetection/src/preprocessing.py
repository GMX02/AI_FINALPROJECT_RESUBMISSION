import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# === SETTINGS ===
# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if we're being called from the GUI
if os.path.basename(os.getcwd()) == 'GunshotDetection':
    # Direct call from GunshotDetection directory
    URBAN_DIR = 'data/UrbanSound8K/audio/'
    URBAN_CSV = 'data/UrbanSound8K/metadata/UrbanSound8K.csv'
    GUNSHOT_DIR = 'data/gunshot_data/audio/'
    FEATURES_DIR = 'features/'
else:
    # Called from GUI
    URBAN_DIR = os.path.join(CURRENT_DIR, '../data/UrbanSound8K/audio/')
    URBAN_CSV = os.path.join(CURRENT_DIR, '../data/UrbanSound8K/metadata/UrbanSound8K.csv')
    GUNSHOT_DIR = os.path.join(CURRENT_DIR, '../data/gunshot_data/audio/')
    FEATURES_DIR = os.path.join(CURRENT_DIR, '../features/')

SAMPLE_RATE = 22050  # Target sample rate
DURATION = 4.0       # Clip duration (seconds)
N_MELS = 64          # Number of mel bands

# === Ensure output directory exists ===
os.makedirs(FEATURES_DIR, exist_ok=True)

def extract_log_mel(audio_path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS):
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    if len(y) < int(sr * duration):
        y = np.pad(y, (0, int(sr * duration) - len(y)))  # pad if too short
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)
    return log_mel

# Only run the full processing if this file is run directly
if __name__ == '__main__':
    # === Prepare storage ===
    X = []
    y = []

    # === Process UrbanSound8K (urban = label 0) ===
    print("Processing UrbanSound8K...")
    urban_meta = pd.read_csv(URBAN_CSV)

    for idx, row in tqdm(urban_meta.iterrows(), total=len(urban_meta)):
        fold = f"fold{row['fold']}"
        file_path = os.path.join(URBAN_DIR, fold, row['slice_file_name'])
        try:
            feature = extract_log_mel(file_path)
            X.append(feature)
            y.append(0)  # Label 0 for urban sounds
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # === Process Gunshot Data (gunshot = label 1) ===
    print("Processing Gunshot Data...")
    for root, dirs, files in os.walk(GUNSHOT_DIR):
        for file_name in tqdm(files, desc=f"Scanning {root}", leave=False):
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                try:
                    feature = extract_log_mel(file_path)
                    X.append(feature)
                    y.append(1)  # Label 1 for gunshots
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # === Finalize ===
    X = np.array(X)
    y = np.array(y)
    print(f"Feature shape: {X.shape}, Labels shape: {y.shape}")

    np.save(os.path.join(FEATURES_DIR, 'X.npy'), X)
    np.save(os.path.join(FEATURES_DIR, 'y.npy'), y)

    print("Feature extraction complete! Files saved to /features/")