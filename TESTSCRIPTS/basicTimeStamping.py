import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import soundfile as sf
from moviepy import VideoFileClip

def extract_audio_from_video(video_path, output_path="temp_audio.wav"):
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_path, logger=None)
        return output_path
    except Exception as e:
        messagebox.showerror("Error", f"Audio extraction failed: {e}")
        return None

def detect_gunshots(audio_path, frame_duration=0.01, energy_threshold=0.6):
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(frame_duration * sr)
    hop_length = frame_length // 2
    energy = np.array([
        np.sum(np.abs(y[i:i+frame_length])**2)
        for i in range(0, len(y), hop_length)
    ])
    energy /= np.max(energy)
    spikes = np.where(energy > energy_threshold)[0]
    timestamps = [round((i * hop_length) / sr, 3) for i in spikes]
    filtered = []
    for t in timestamps:
        if not filtered or (t - filtered[-1]) > 0.2:
            filtered.append(t)
    return filtered

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

    timestamps = detect_gunshots(audio_path)
    if timestamps:
        result = "\n".join([f"{t:.3f} sec" for t in timestamps])
        messagebox.showinfo("Gunshot Timestamps", result)
    else:
        messagebox.showinfo("No Gunshots Detected", "No spikes found.")

# === GUI Setup ===
root = tk.Tk()
root.title("Gunshot Timestamp Detector")
root.geometry("300x150")
root.resizable(False, False)

btn = tk.Button(root, text="Choose Audio/Video File", command=choose_file, font=("Arial", 12))
btn.pack(expand=True)

root.mainloop()
