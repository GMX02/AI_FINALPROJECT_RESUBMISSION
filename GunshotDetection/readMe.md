# Gunshot Detection System

An advanced audio analysis platform for detecting and classifying gunshots using machine learning.

## Project Structure

```
GunshotDetection/
├── src/                    # Source code directory
│   ├── main_gui.py        # Main GUI application
│   ├── startup.py         # System initialization and model training
│   ├── firearm_classifier.py  # Firearm classification model
│   ├── caliber_classifier.py  # Caliber classification model
│   ├── basicTimeStamping.py   # Gunshot detection algorithm
│   └── processing.py      # Audio processing utilities
├── models/                 # Trained model files
│   ├── firearm_model.h5   # Firearm classification model
│   ├── caliber_model.h5   # Caliber classification model
│   ├── firearm_encoder.pkl # Firearm label encoder
│   └── caliber_encoder.pkl # Caliber label encoder
├── GUI_Files/             # Image assets for GUI
│   ├── firearm_images/    # Firearm type images
│   └── bullet_images/     # Bullet/caliber images
├── reports/               # Generated analysis reports
└── data/                  # Data directories
    ├── raw/              # Raw audio samples
    ├── processed/        # Processed audio features
    └── splits/           # Train/test splits
```

## System Architecture

The system consists of three main components:

1. **Gunshot Detection**
   - Uses energy-based detection algorithm
   - Identifies potential gunshot events in audio
   - Provides timestamps and confidence scores

2. **Firearm Classification**
   - CNN-based model for firearm type identification
   - Classifies between Glock, Ruger, Remington, and Smith & Wesson
   - Uses mel spectrogram features

3. **Caliber Classification**
   - CNN-based model for caliber identification
   - Classifies between 9mm, 5.56mm, 12 Gauge, and .38 cal
   - Uses mel spectrogram features

## Model Training Process

### Data Preparation
1. Audio samples are collected for each firearm type
2. Samples are converted to mel spectrograms
3. Features are normalized and preprocessed
4. Data is split into training (80%) and validation (20%) sets

### Firearm Model Architecture
```python
Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 firearm types
])
```

### Caliber Model Architecture
```python
Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 caliber types
])
```

### Training Parameters
- Optimizer: Adam (learning rate = 0.001)
- Loss: Categorical Crossentropy
- Batch Size: 32
- Epochs: 50
- Early Stopping: Patience = 5
- Validation Split: 20%

## Startup Process

The `startup.py` script handles system initialization:

1. **Directory Structure Check**
   - Creates required directories if missing
   - Verifies data and model locations

2. **Model Verification**
   - Checks for existing trained models
   - If missing, initiates training process
   - Validates model performance

3. **Data Verification**
   - Checks for required audio samples
   - If missing, downloads required data
   - Prepares data for training

4. **System Initialization**
   - Loads trained models
   - Initializes audio processing
   - Launches GUI application

## Running the System

1. **Initial Setup**
   ```bash
   cd GunshotDetection/src
   python startup.py
   ```

2. **System Checks**
   - Verifies model files
   - Checks data availability
   - Initializes required components

3. **Automatic Training (if needed)**
   - Downloads missing data
   - Trains new models
   - Saves trained models

4. **GUI Launch**
   - Shows splash screen
   - Initializes main interface
   - Ready for analysis

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- PyQt5
- librosa
- soundfile
- matplotlib
- scikit-learn
- numpy

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run startup script:
   ```bash
   python src/startup.py
   ```

## Usage

1. Launch the application using `startup.py`
2. Load an audio file through the GUI
3. Run detection and classification
4. View results and generate reports

## Performance Metrics

### Firearm Classification
- Accuracy: 98.75%
- Precision: 0.99
- Recall: 0.99
- F1-Score: 0.99

### Caliber Classification
- Accuracy: 97.50%
- Precision: 0.98
- Recall: 0.98
- F1-Score: 0.98

## Future Improvements

1. **Model Enhancements**
   - Add more firearm types
   - Implement data augmentation
   - Optimize for real-time processing

2. **System Features**
   - Add batch processing
   - Implement cloud storage
   - Add user authentication

3. **UI Improvements**
   - Add real-time visualization
   - Implement report customization
   - Add user preferences
