import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
import soundfile as sf

# Configuration
SAMPLE_RATE = 22050
DURATION = 1.0  # 1 second clips
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

def load_models():
    """Load trained models and encoders"""
    firearm_model = tf.keras.models.load_model('models/firearm_classifier.h5')
    caliber_model = tf.keras.models.load_model('models/caliber_classifier.h5')
    
    with open('models/firearm_encoder.pkl', 'rb') as f:
        firearm_encoder = pickle.load(f)
    with open('models/caliber_encoder.pkl', 'rb') as f:
        caliber_encoder = pickle.load(f)
    
    return firearm_model, caliber_model, firearm_encoder, caliber_encoder

def extract_features(audio_path):
    """Extract mel spectrogram features from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or truncate to ensure consistent length
        if len(y) < int(SAMPLE_RATE * DURATION):
            y = np.pad(y, (0, int(SAMPLE_RATE * DURATION) - len(y)))
        else:
            y = y[:int(SAMPLE_RATE * DURATION)]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        return log_mel_spec
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def predict_firearm(audio_path, firearm_model, caliber_model, firearm_encoder, caliber_encoder):
    """Predict firearm type and caliber from audio file"""
    # Extract features
    features = extract_features(audio_path)
    if features is None:
        return None, None
    
    # Prepare features for prediction
    features = np.expand_dims(features, -1)  # Add channel dimension
    features = np.expand_dims(features, 0)   # Add batch dimension
    features = (features - np.min(features)) / (np.max(features) - np.min(features))  # Normalize
    
    # Predict firearm type
    firearm_pred = firearm_model.predict(features)
    firearm_idx = np.argmax(firearm_pred)
    firearm_type = firearm_encoder.inverse_transform([firearm_idx])[0]
    firearm_confidence = firearm_pred[0][firearm_idx]
    
    # Predict caliber
    caliber_pred = caliber_model.predict(features)
    caliber_idx = np.argmax(caliber_pred)
    caliber = caliber_encoder.inverse_transform([caliber_idx])[0]
    caliber_confidence = caliber_pred[0][caliber_idx]
    
    return {
        'firearm_type': firearm_type,
        'firearm_confidence': float(firearm_confidence),
        'caliber': caliber,
        'caliber_confidence': float(caliber_confidence)
    }

def main():
    # Load models and encoders
    print("Loading models...")
    firearm_model, caliber_model, firearm_encoder, caliber_encoder = load_models()
    
    # Get list of audio files to predict
    audio_files = [f for f in os.listdir('test_audio') if f.endswith('.wav')]
    
    # Predict for each audio file
    for audio_file in audio_files:
        audio_path = os.path.join('test_audio', audio_file)
        print(f"\nPredicting for {audio_file}...")
        
        result = predict_firearm(
            audio_path,
            firearm_model,
            caliber_model,
            firearm_encoder,
            caliber_encoder
        )
        
        if result:
            print(f"Predicted Firearm: {result['firearm_type']} ({result['firearm_confidence']:.2%})")
            print(f"Predicted Caliber: {result['caliber']} ({result['caliber_confidence']:.2%})")

if __name__ == '__main__':
    main() 