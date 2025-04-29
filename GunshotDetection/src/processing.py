# processing.py (dummy implementation)

import numpy as np
import librosa
import os
from basicTimeStamping import detect_gunshots as basic_detect_gunshots

# Retrieve basic audio info (duration and sample rate)
def get_audio_info(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Return metadata
        return {
            'length': duration,
            'sample_rate': sr
        }
    except Exception as e:
        # Handle any error that occurs during audio loading
        print(f"Error getting audio info: {e}")
        return {
            'length': 0,
            'sample_rate': 0
        }

# Detect if gunshots are present in an audio file
def detect_gunshot(file_path):
    try:
        # Use basicTimeStamping module to detect timestamps
        timestamps = basic_detect_gunshots(file_path)
        
        # Return detection result with confidence scaled by number of detections
        return {
            'presence': len(timestamps) > 0,
            'confidence': min(95.0, len(timestamps) * 10.0),  # Scale confidence with number of detections
            'timestamps': timestamps
        }
    except Exception as e:
        # Handle errors in detection process
        print(f"Error detecting gunshots: {e}")
        return {
            'presence': False,
            'confidence': 0.0,
            'timestamps': []
        }

# Locate and annotate all detected gunshots with metadata
def locate_gunshots(file_path):
    try:
        print(f"Starting gunshot detection on: {file_path}")
        
        # Use the actual detection from basicTimeStamping
        print("Running detection from basicTimeStamping...")
        timestamps = basic_detect_gunshots(file_path)
        print(f"Found {len(timestamps)} potential gunshots at times: {timestamps}")
        
        # Convert timestamps to the expected format with metadata
        gunshots = []
        for t in timestamps:
            # Create a temporary audio segment for this gunshot
            y, sr = librosa.load(file_path, sr=None, offset=t-0.1, duration=0.2)
            temp_file = f"temp_gunshot_{t}.wav"
            import soundfile as sf
            sf.write(temp_file, y, sr)
            
            # Categorize the firearm
            firearm_info = categorize_firearm(temp_file)
            
            # Clean up temporary file
            os.remove(temp_file)
            
            gunshots.append({
                'time': t,
                'confidence': 0.95,  # High confidence for detected spikes
                'type': firearm_info['firearm'],
                'caliber': firearm_info['caliber'],
                'energy': 'Unknown',  # Energy will be calculated in future
                'peak_pressure': 'Unknown',  # Pressure will be calculated in future
                'frequency': 'Unknown',  # Frequency will be calculated in future
                'match_confidence': firearm_info['match_confidence']
            })
        
        print(f"Returning {len(gunshots)} gunshot markers")
        return gunshots
    except Exception as e:
        print(f"Error locating gunshots: {e}")
        return []

# Categorize firearm based on audio characteristics
def categorize_firearm(file_path):
    try:
        print("\n=== FIREARM CLASSIFICATION DEBUG ===")
        print(f"Attempting to load models from: ../../models/")
        
        # Load the firearm classification model
        from tensorflow.keras.models import load_model
        import pickle
        
        # Load firearm model and encoder
        print("Loading firearm model...")
        firearm_model = load_model('../../models/firearm_model.h5')
        print("Loading firearm encoder...")
        with open('../../models/firearm_encoder.pkl', 'rb') as f:
            firearm_encoder = pickle.load(f)
        
        # Load caliber model and encoder
        print("Loading caliber model...")
        caliber_model = load_model('../../models/caliber_model.h5')
        print("Loading caliber encoder...")
        with open('../../models/caliber_encoder.pkl', 'rb') as f:
            caliber_encoder = pickle.load(f)
            
        print("All models and encoders loaded successfully!")
        
        # Extract features EXACTLY as in firearm_classifier.py
        print("\nExtracting features from audio file...")
        SAMPLE_RATE = 44100
        DURATION = 2.0
        N_MELS = 128
        HOP_LENGTH = 512
        N_FFT = 2048
        
        # Load and preprocess audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) < SAMPLE_RATE * DURATION:
            y = np.pad(y, (0, int(SAMPLE_RATE * DURATION) - len(y)))
        else:
            y = y[:int(SAMPLE_RATE * DURATION)]
        
        # Extract all features
        features = {}
        features['mel_spec'] = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
        ))
        features['mfcc'] = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=20, n_mels=N_MELS, hop_length=HOP_LENGTH
        )
        features['chroma'] = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=HOP_LENGTH
        )
        features['contrast'] = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=HOP_LENGTH
        )
        features['zcr'] = librosa.feature.zero_crossing_rate(y)
        features['rms'] = librosa.feature.rms(y=y)
        
        # Combine features exactly as in training
        X = np.concatenate([
            features['mel_spec'].flatten(),
            features['mfcc'].flatten(),
            features['chroma'].flatten(),
            features['contrast'].flatten(),
            features['zcr'].flatten(),
            features['rms'].flatten()
        ])
        
        # Normalize
        X = (X - X.mean()) / X.std()
        
        # Reshape for model input
        X = X.reshape(1, X.shape[0], 1)
        
        print(f"Input shape: {X.shape}")
        
        # Predict firearm type
        print("\nMaking firearm predictions...")
        firearm_pred = firearm_model.predict(X)
        firearm_idx = np.argmax(firearm_pred[0])
        firearm_type = firearm_encoder.inverse_transform([firearm_idx])[0]
        firearm_confidence = float(np.max(firearm_pred[0]) * 100)
        print(f"Firearm prediction probabilities: {firearm_pred[0]}")
        print(f"Selected firearm: {firearm_type} (index: {firearm_idx})")
        print(f"Firearm confidence: {firearm_confidence:.2f}%")
        
        # Predict caliber
        print("\nMaking caliber predictions...")
        caliber_pred = caliber_model.predict(X)
        caliber_idx = np.argmax(caliber_pred[0])
        caliber = caliber_encoder.inverse_transform([caliber_idx])[0]
        caliber_confidence = float(np.max(caliber_pred[0]) * 100)
        print(f"Caliber prediction probabilities: {caliber_pred[0]}")
        print(f"Selected caliber: {caliber} (index: {caliber_idx})")
        print(f"Caliber confidence: {caliber_confidence:.2f}%")
        
        print("=== END DEBUG ===\n")
        
        return {
            'firearm': firearm_type,
            'caliber': caliber,
            'match_confidence': firearm_confidence
        }
    except Exception as e:
        print(f"\n=== ERROR IN FIREARM CLASSIFICATION ===")
        print(f"Error details: {str(e)}")
        print("=== END ERROR ===\n")
        return {
            'firearm': 'Error',
            'caliber': 'N/A',
            'match_confidence': 0.0
        }
