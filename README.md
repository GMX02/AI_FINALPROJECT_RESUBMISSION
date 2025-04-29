# Gunshot Detection and Classification System

This project is a full Python-based system to:
- Detect gunshot sounds in audio files
- Differentiate gunshots from urban environmental sounds using Machine Learning (CNN)
- Provide a graphical interface (PyQt5) for interaction
- Generate professional PDF reports

Built with TensorFlow, librosa, PyQt5, and trained on UrbanSound8K and real-world gunshot datasets.

## System Components

### 1. Gunshot Detection
- CNN-based binary classification model
- Distinguishes between gunshots and urban sounds
- Uses mel spectrogram features
- Binary output (gunshot vs urban sound)

### 2. Firearm Classification
- CNN-based model
- Classifies between 4 firearm types:
  - Glock
  - Ruger
  - Remington
  - Smith & Wesson
- 98.75% accuracy

### 3. Caliber Classification
- CNN-based model
- Classifies between 4 caliber types:
  - 9mm
  - 5.56mm
  - 12 Gauge
  - .38 cal
- 97.50% accuracy

## Project Structure

```
.
├── GunshotDetection/          # Main project directory
│   ├── src/                  # Source code
│   │   ├── main_gui.py      # GUI application
│   │   ├── startup.py       # System initialization
│   │   ├── firearm_classifier.py  # Firearm model
│   │   ├── caliber_classifier.py  # Caliber model
│   │   ├── basicTimeStamping.py   # Detection algorithm
│   │   └── processing.py    # Audio processing
│   ├── models/              # Trained models
│   │   ├── gunshot_classifier.h5  # Gunshot detection model
│   │   ├── firearm_model.h5
│   │   ├── caliber_model.h5
│   │   ├── firearm_encoder.pkl
│   │   └── caliber_encoder.pkl
│   ├── GUI_Files/          # GUI assets
│   │   ├── firearm_images/
│   │   └── bullet_images/
│   ├── reports/            # Generated reports
│   └── data/              # Data directories
│       ├── raw/          # Raw audio
│       ├── processed/    # Processed features
│       └── splits/       # Train/test splits
└── README.md             # This file
```

## Getting Started

### 1. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Install Python Requirements
```bash
pip install -r GunshotDetection/requirements.txt
```

### 3. Run the Startup Script
```bash
cd GunshotDetection/src
python startup.py
```

The startup script (`startup.py`) automatically handles:
1. **Directory Structure Setup**
   - Creates required directories if missing
   - Sets up data and model paths

2. **Model Verification and Training**
   - Checks for existing trained models
   - If models are missing:
     - Downloads required datasets
     - Extracts audio features
     - Trains all three CNN models:
       - Gunshot detection model
       - Firearm classification model
       - Caliber classification model
   - Saves trained models to `/models/`

3. **Data Verification**
   - Checks for required audio samples
   - Downloads missing data from:
     - UrbanSound8K dataset (urban noises)
     - Edge-collected gunshot dataset (real firearm recordings)
   - Processes and prepares data for training

4. **System Initialization**
   - Loads trained models
   - Initializes audio processing
   - Launches the GUI application

## Model Architectures

### 1. Gunshot Detection Model
```python
Sequential([
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### 2. Firearm Classification Model
```python
Sequential([
    Input(shape=input_shape),
    Reshape(reshape_dim),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### 3. Caliber Classification Model
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
- Optimizer: Adam (LR: 0.001)
- Loss: 
  - Gunshot Detection: Binary Crossentropy
  - Firearm/Caliber: Categorical Crossentropy
- Batch Size: 32
- Epochs: 50
- Early Stopping: Patience = 5
- Validation Split: 20%

## Using the System

1. **Launch the Application**
   - Run `startup.py` to initialize the system
   - The GUI will launch automatically

2. **Load Audio**
   - Click "Load Audio" to select a file
   - Supported formats: WAV, MP3

3. **Run Analysis**
   - Click "Detect Gunshots" to start analysis
   - View real-time results in the GUI
   - Generate detailed reports

4. **View Results**
   - Spectrogram visualization
   - Detection timestamps
   - Firearm and caliber classification
   - Confidence scores

## Performance Metrics

### Gunshot Detection
- Binary classification accuracy
- Precision and recall for gunshot detection
- False positive rate for urban sounds

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

## Needed Improvements
- Improve detection
- Firearm classification
- Improved GUI for database browsing
- PDF report generation

## Future Enhancements

1. **Model Enhancements**
   - Add more firearm types
   - Implement data augmentation
   - Optimize for real-time processing

2. **System Features**
   - Batch processing
   - Cloud storage integration
   - User authentication

3. **UI Improvements**
   - Real-time visualization
   - Report customization
   - User preferences
