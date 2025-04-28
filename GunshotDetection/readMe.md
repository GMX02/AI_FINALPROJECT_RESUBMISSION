# Gunshot Detection and Classification System

This project is a full Python-based system to:
- Detect gunshot sounds in audio files.
- Differentiate gunshots from urban environmental sounds using Machine Learning (CNN).
- Provide a graphical interface (PyQt5) for interaction.
- Generate professional LaTeX reports.

Built with TensorFlow, librosa, PyQt5, and trained on UrbanSound8K and real-world gunshot datasets.

---

## Project Structure

```
GunshotDetection/
├── src/
│   ├── main_gui.py           # Full PyQt5 GUI
│   ├── predict.py            # Model inference code
│   ├── preprocessing.py      # Feature extraction script
│   ├── processing.py         # Basic detection/wrappers
│   ├── train.py              # CNN training script
│   ├── basicTimeStamping.py  # Basic energy spike detection
│   ├── database.py           # Query/save detection results
│   ├── urbanSoundsDownload.py# Dataset downloader
├── models/                   # Trained models 
├── features/                 # Extracted features
├── reports/                  # Generated LaTeX reports
├── README.md
```

---

##  How to Set Up

### 1. Clone the Repository
```bash
https://github.com/your-username/GunshotDetection.git
cd GunshotDetection
```

### 2. Install Python Requirements
```bash
pip install -r requirements.txt
```

### 3. Download the Required Datasets
Run this script to automatically download and extract datasets:
```bash
cd src/
python urbanSoundsDownload.py
```
This will download:
- UrbanSound8K dataset (urban noises)
- Edge-collected gunshot dataset (real firearm recordings)

Datasets will be placed inside `/data/`.

---

## How to Train Your Model

1. **Extract features**:
```bash
cd src/
python preprocessing.py
```
This generates `/features/X.npy` and `/features/y.npy`.

2. **Train the CNN model**:
```bash
python train.py
```
The trained model will be saved into `/models/gunshot_classifier.h5`.

---

## How to Run the GUI

1. Launch the GUI:
```bash
cd src/
python main_gui.py
```

2. In the GUI you can:
- Load an audio file (`.wav`)
- Play, pause, seek
- Detect gunshots
- Classify each detection as "Gunshot" or "Urban Sound"
- Generate a professional LaTeX report summarizing results

---


## Requirements
- Python 3.9-3.11
- TensorFlow 2.11+
- librosa
- PyQt5
- pandas
- scikit-learn
- numpy
- soundfile
- tqdm
- gdown
- requests

*(maybe add a requirements.txt)*

---

## Needed Improvements
- Improve detection
- Firearm classification
- Improved GUI for database browsing
- LaTeX to PDF report generation

---

## Authors
- Developed by [Our names]
