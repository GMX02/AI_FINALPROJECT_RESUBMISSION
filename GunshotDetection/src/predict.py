import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

# === SETTINGS ===
MODEL_PATH = '../models/gunshot_classifier.h5'
SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 64

# === Load model ===
print("Loading model...")
model = load_model(MODEL_PATH)

# === Prediction function ===
def extract_log_mel(audio_path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS):
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    if len(y) < int(sr * duration):
        y = np.pad(y, (0, int(sr * duration) - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)
    return log_mel

def predict_audio(audio_path):
    # Extract feature
    feature = extract_log_mel(audio_path)
    feature = np.expand_dims(feature, axis=-1)  # Add channel
    feature = np.expand_dims(feature, axis=0)   # Add batch dimension

    # Normalize (IMPORTANT: same as training!)
    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

    # Predict
    pred = model.predict(feature)

    if pred[0][0] > 0.5:
        return "Gunshot", float(pred[0][0])
    else:
        return "Urban Sound", float(1 - pred[0][0])

# === Example usage ===
if __name__ == "__main__":
    test_audio = '../some_test_file.wav'  # Replace with your test file

    if os.path.exists(test_audio):
        label, confidence = predict_audio(test_audio)
        print(f"Prediction: {label} ({confidence*100:.2f}% confidence)")
    else:
        print(f"Test file {test_audio} not found.")
