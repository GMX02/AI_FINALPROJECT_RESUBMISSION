import numpy as np
import librosa
import os
import soundfile as sf

# Detects gunshots in an audio file based on energy spikes
def detect_gunshots(audio_path, frame_duration=0.05, energy_threshold=0.6, min_time_between=0.3):
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(frame_duration * sr)
    hop_length = frame_length // 2  # 50% overlap between frames
    
    # Calculate energy for each frame
    energy = np.array([
        np.sum(np.abs(y[i:i+frame_length])**2)
        for i in range(0, len(y), hop_length)
    ])
    energy /= np.max(energy)
    
    # Find spikes above threshold
    spikes = np.where(energy > energy_threshold)[0]
    
    # Convert frame indices to timestamps
    timestamps = [round((i * hop_length) / sr, 3) for i in spikes]
    
    # Filter out multiple detections of the same gunshot
    filtered = []
    
    for t in timestamps:
        if not filtered or (t - filtered[-1]) > min_time_between:
            filtered.append(t)
    
    return filtered
