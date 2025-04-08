import numpy as np
import librosa
import librosa.display
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from moviepy import VideoFileClip
import matplotlib.pyplot as plt

def show_plot(y, sr, timestamps):
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(10, 4))
    plt.plot(times, y, label='Audio Signal')
    for t in timestamps:
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.7)
    plt.title("Detected Gunshots")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

def extract_audio_from_video(video_path, output_path="temp_audio.wav"):
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_path, logger=None)
        return output_path
    except Exception as e:
        messagebox.showerror("Error", f"Audio extraction failed: {e}")
        return None

def detect_gunshots_onset(audio_path, min_separation=0.5):
    y, sr = librosa.load(audio_path, sr=None)
    y = librosa.effects.preemphasis(y)

    hop_length = 512
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=False)

    times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Filter close detections
    filtered = []
    last_time = -np.inf
    for t in times:
        if t - last_time > min_separation:
            filtered.append(round(t, 3))
            last_time = t

    return filtered, y, sr

def show_spectrogram(y, sr, timestamps, hop_length=512):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    for t in timestamps:
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.7)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram with Detected Gunshots')
    plt.tight_layout()
    plt.show()

def choose_file():
    path = filedialog.askopenfilename(
        title="Select Audio or Video File",
        filetypes=[("Media files", "*.wav *.mp3 *.mp4 *.mov *.mkv")]
    )
    if not path:
        return
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.mp4', '.mov', '.mkv']:
        audio_path = extract_audio_from_video(path)
    else:
        audio_path = path

    if not audio_path:
        return

    timestamps, y, sr = detect_gunshots_onset(audio_path)
    show_spectrogram(y, sr, timestamps)

    if timestamps:
        result = "\n".join([f"{t:.3f} sec" for t in timestamps])
        messagebox.showinfo("Gunshot Timestamps", result)
    else:
        messagebox.showinfo("No Gunshots Detected", "No onsets found.")

# === GUI Setup ===
root = tk.Tk()
root.title("Gunshot Timestamp Detector")
root.geometry("300x150")
root.resizable(False, False)

btn = tk.Button(root, text="Choose Audio/Video File", command=choose_file, font=("Arial", 12))
btn.pack(expand=True)

root.mainloop()
