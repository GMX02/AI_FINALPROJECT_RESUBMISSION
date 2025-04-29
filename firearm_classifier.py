import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
SAMPLE_RATE = 44100
DURATION = 2.0  # seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
DATASET_PATH = 'edge-collected-gunshot-audio'
METADATA_PATH = 'Data files/gunshot-audio-all-metadata.csv'

def load_metadata():
    """Load and analyze metadata from CSV file."""
    print("Loading metadata...")
    df = pd.read_csv(METADATA_PATH)
    print(f"Metadata columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    return df

def extract_features(audio_path):
    """Extract multiple features from audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Pad or truncate to fixed duration
        if len(y) < SAMPLE_RATE * DURATION:
            y = np.pad(y, (0, int(SAMPLE_RATE * DURATION) - len(y)))
        else:
            y = y[:int(SAMPLE_RATE * DURATION)]
        
        # Extract features
        features = {}
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        features['mel_spec'] = librosa.power_to_db(mel_spec)
        
        # MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=20, n_mels=N_MELS, hop_length=HOP_LENGTH
        )
        features['mfcc'] = mfcc
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=HOP_LENGTH
        )
        features['chroma'] = chroma
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=HOP_LENGTH
        )
        features['contrast'] = contrast
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr'] = zcr
        
        # Root mean square energy
        rms = librosa.feature.rms(y=y)
        features['rms'] = rms
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def create_dataset(metadata_df):
    """Create dataset from metadata and audio files."""
    print("Creating dataset...")
    X = []
    y_firearm = []
    y_caliber = []
    
    # Print unique firearms and calibers in metadata
    print("\nUnique firearms in metadata:")
    print(metadata_df['firearm'].unique())
    print("\nUnique calibers in metadata:")
    print(metadata_df['caliber'].unique())
    
    # Create a mapping of firearm types to directory names
    firearm_dir_map = {
        'Glock 17': 'glock_17_9mm_caliber',
        'Smith & Wesson': '38s&ws_dot38_caliber',
        'Remington 870': 'remington_870_12_gauge',
        'Ruger AR-556': 'ruger_ar_556_dot223_caliber',
        'Ruger 556': 'ruger_ar_556_dot223_caliber'
    }
    
    processed_count = 0
    error_count = 0
    
    # First, verify the dataset directory exists
    if not os.path.exists(DATASET_PATH):
        raise ValueError(f"Dataset directory not found: {DATASET_PATH}")
    
    # Print available directories
    print("\nAvailable directories:")
    print(os.listdir(DATASET_PATH))
    
    for _, row in metadata_df.iterrows():
        filename = row['filename']
        firearm = row['firearm']
        caliber = row['caliber']
        
        # Get the correct directory name
        dir_name = firearm_dir_map.get(firearm)
        if not dir_name:
            print(f"Warning: No directory mapping found for firearm: {firearm}")
            continue
        
        # Add .wav extension if not present
        if not filename.endswith('.wav'):
            filename = f"{filename}.wav"
        
        # Construct audio path
        audio_path = os.path.join(DATASET_PATH, dir_name, filename)
        
        if os.path.exists(audio_path):
            try:
                features = extract_features(audio_path)
                if features is not None:
                    # Combine features into a single array
                    combined_features = np.concatenate([
                        features['mel_spec'].flatten(),
                        features['mfcc'].flatten(),
                        features['chroma'].flatten(),
                        features['contrast'].flatten(),
                        features['zcr'].flatten(),
                        features['rms'].flatten()
                    ])
                    
                    X.append(combined_features)
                    y_firearm.append(firearm)
                    y_caliber.append(caliber)
                    processed_count += 1
                    if processed_count % 10 == 0:  # Changed to show progress more frequently
                        print(f"Processed {processed_count} files...")
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                error_count += 1
        else:
            print(f"Warning: Audio file not found: {audio_path}")
            error_count += 1
    
    print(f"\nDataset creation complete:")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed to process: {error_count} files")
    
    if processed_count == 0:
        raise ValueError("No audio files were successfully processed. Please check the dataset paths and file structure.")
    
    return np.array(X), np.array(y_firearm), np.array(y_caliber)

def build_model(input_shape, num_classes):
    """Build a more sophisticated CNN model."""
    # Calculate the correct reshape dimensions
    # We want to reshape the input into a square-like shape for the CNN
    # Find the closest factors of the input size
    input_size = input_shape[0]
    sqrt_size = int(np.sqrt(input_size))
    while input_size % sqrt_size != 0:
        sqrt_size -= 1
    
    reshape_dim = (sqrt_size, input_size // sqrt_size, 1)
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Reshape for CNN
        layers.Reshape(reshape_dim),
        
        # First CNN block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second CNN block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third CNN block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history, title):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_training_history.png')
    plt.close()

def main():
    try:
        # Load metadata
        metadata_df = load_metadata()
        
        # Create dataset
        X, y_firearm, y_caliber = create_dataset(metadata_df)
        
        # Normalize features
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Encode labels
        firearm_encoder = LabelEncoder()
        caliber_encoder = LabelEncoder()
        
        y_firearm_encoded = firearm_encoder.fit_transform(y_firearm)
        y_caliber_encoded = caliber_encoder.fit_transform(y_caliber)
        
        # Split data
        X_train, X_test, y_firearm_train, y_firearm_test, y_caliber_train, y_caliber_test = train_test_split(
            X, y_firearm_encoded, y_caliber_encoded, test_size=0.2, random_state=42
        )
        
        # Reshape data for CNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        print(f"\nInput shape: {X_train.shape}")
        print(f"Number of firearm classes: {len(firearm_encoder.classes_)}")
        print(f"Number of caliber classes: {len(caliber_encoder.classes_)}")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Train firearm classification model
        print("\nTraining firearm classification model...")
        firearm_model = build_model(X_train.shape[1:], len(firearm_encoder.classes_))
        firearm_history = firearm_model.fit(
            X_train, y_firearm_train,
            validation_data=(X_test, y_firearm_test),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/firearm_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
        )
        
        # Plot firearm training history
        plot_training_history(firearm_history, 'Firearm Classification')
        
        # Train caliber classification model
        print("\nTraining caliber classification model...")
        caliber_model = build_model(X_train.shape[1:], len(caliber_encoder.classes_))
        caliber_history = caliber_model.fit(
            X_train, y_caliber_train,
            validation_data=(X_test, y_caliber_test),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/caliber_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
        )
        
        # Plot caliber training history
        plot_training_history(caliber_history, 'Caliber Classification')
        
        # Save encoders
        import pickle
        with open('models/firearm_encoder.pkl', 'wb') as f:
            pickle.dump(firearm_encoder, f)
        with open('models/caliber_encoder.pkl', 'wb') as f:
            pickle.dump(caliber_encoder, f)
        
        print("\nTraining complete! Models and encoders saved in 'models' directory.")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check the following:")
        print("1. The dataset directory structure")
        print("2. The metadata CSV file")
        print("3. The audio file paths")
        raise

if __name__ == "__main__":
    main() 