import sys
import time
from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from main_gui import GunshotDetectionApp
import os
import subprocess
import shutil
from pathlib import Path
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

# Widget that displays a rotating loading spinner
class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 50)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_angle)
        self.timer.start(50)  # Update every 50ms

    def update_angle(self):
        # Increase angle by 10 degrees each update
        self.angle = (self.angle + 10) % 360
        self.update()

    # Custom painting of the spinner
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw spinner
        pen = QPen(QColor(100, 180, 255), 3)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        rect = QRect(5, 5, 40, 40)
        painter.drawArc(rect, self.angle * 16, 270 * 16)  # 270 degrees arc

# Splash screen displayed at application startup
class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(400, 300)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # Add title
        title = QLabel("Gunshot Detection Platform")
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
                font-family: 'Arial';
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Add subtitle
        subtitle = QLabel("Advanced Audio Analysis")
        subtitle.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 16px;
                font-family: 'Arial';
            }
        """)
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # Add loading spinner
        self.spinner = LoadingSpinner()
        layout.addWidget(self.spinner, 0, Qt.AlignCenter)
        
        # Add loading text
        self.loading_text = QLabel("Initializing...")
        self.loading_text.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 14px;
                font-family: 'Arial';
            }
        """)
        self.loading_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_text)
        
        self.setLayout(layout)
        
        # Add fade-in animation
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(500)  # 500ms
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # Add fade-out animation
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(500)  # 500ms
        self.fade_out.setStartValue(1.0)
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.InOutQuad)
        self.fade_out.finished.connect(self.close)

    # When splash screen is shown, start animations
    def showEvent(self, event):
        super().showEvent(event)
        self.animation.start()
        
        # Start loading sequence
        self.loading_sequence = [
            ("Loading audio engine...", 1),
            ("Initializing detection algorithms...", 2),
            ("Preparing user interface...", 2),
            ("Starting application...", 1)
        ]
        self.current_step = 0
        self.start_loading_sequence()

    # Display each loading step sequentially with a delay
    def start_loading_sequence(self):
        if self.current_step < len(self.loading_sequence):
            text, duration = self.loading_sequence[self.current_step]
            self.loading_text.setText(text)
            QTimer.singleShot(duration * 1000, self.next_loading_step)
        else:
            self.fade_out.start()

    # Proceed to next step in the loading sequence
    def next_loading_step(self):
        self.current_step += 1
        self.start_loading_sequence()

    # Paint semi-transparent background with rounded corners
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background with rounded corners
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(43, 43, 43, 230))  # Semi-transparent dark background
        painter.drawRoundedRect(self.rect(), 10, 10)

def check_directory_structure():
    """Check if the required directory structure exists, create if not"""
    # Get the root directory (two levels up from this script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    
    required_dirs = [
        os.path.join(root_dir, 'models'),
        os.path.join(root_dir, 'data', 'raw'),
        os.path.join(root_dir, 'data', 'processed'),
        os.path.join(root_dir, 'data', 'splits'),
        os.path.join(root_dir, 'reports'),
        os.path.join(root_dir, 'Data files'),  # Add this directory as it's needed for metadata
        os.path.join(root_dir, 'GUI_Files')    # Add this directory as it's needed for GUI assets
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Checked/created directory: {dir_path}")

def check_models():
    """Check if all required models exist"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    
    required_models = [
        os.path.join(root_dir, 'models', 'firearm_model.h5'),
        os.path.join(root_dir, 'models', 'caliber_model.h5'),
        os.path.join(root_dir, 'models', 'firearm_encoder.pkl'),
        os.path.join(root_dir, 'models', 'caliber_encoder.pkl')
    ]
    
    missing_models = [model for model in required_models if not os.path.exists(model)]
    return missing_models

def check_data():
    """Check if required data exists"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    
    required_data = [
        os.path.join(root_dir, 'GUI_Files', 'Glock 17Semi-automatic pistol9mm caliber.gif'),
        os.path.join(root_dir, 'GUI_Files', 'REMINGTON.gif'),
        os.path.join(root_dir, 'GUI_Files', '38 Smith & Wesson Special Revolver.38 caliber.gif'),
        os.path.join(root_dir, 'GUI_Files', 'Everest-9mm-Ammo-8.jpg'),
        os.path.join(root_dir, 'GUI_Files', 'elite-556-65-sbt-Edit__49747.jpg'),
        os.path.join(root_dir, 'GUI_Files', 'red-shells_2d5206ed-0595-45ca-831a-0460fc82e62d.webp'),
        os.path.join(root_dir, 'GUI_Files', '38_158g_ammo_1200x.webp')
    ]
    
    missing_data = [data for data in required_data if not os.path.exists(data)]
    return missing_data

def download_data():
    """Run the data download script"""
    print("Downloading required data...")
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        download_script = os.path.join(current_dir, 'urbanSoundsDownload.py')
        
        # Run the download script
        subprocess.run(['python', download_script], check=True)
        print("Data download completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)

def extract_features(audio_path, n_mels=128, n_fft=2048, hop_length=512):
    """Extract mel spectrogram features from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, 
                                                n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def create_firearm_model(input_shape, num_classes):
    """Create and compile the firearm classification model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def create_caliber_model(input_shape, num_classes):
    """Create and compile the caliber classification model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_models():
    """Train both firearm and caliber models"""
    print("Preparing data for training...")
    
    # Load and preprocess data
    X = []
    y_firearm = []
    y_caliber = []
    
    firearm_types = ['glock', 'ruger', 'remington', 'smith']
    caliber_types = ['9mm', '5.56mm', '12 Gauge', '.38 cal']
    
    for firearm_type in firearm_types:
        data_dir = f'../data/raw/{firearm_type}'
        if not os.path.exists(data_dir):
            continue
            
        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                features = extract_features(os.path.join(data_dir, file))
                if features is not None:
                    X.append(features)
                    y_firearm.append(firearm_type)
                    # Map firearm to caliber
                    caliber_map = {
                        'glock': '9mm',
                        'ruger': '5.56mm',
                        'remington': '12 Gauge',
                        'smith': '.38 cal'
                    }
                    y_caliber.append(caliber_map[firearm_type])
    
    if not X:
        print("No data available for training")
        return False
    
    # Convert to numpy arrays
    X = np.array(X)
    X = X.reshape(X.shape + (1,))  # Add channel dimension
    
    # Create encoders
    from sklearn.preprocessing import LabelEncoder
    
    # Firearm encoder
    firearm_encoder = LabelEncoder()
    y_firearm_encoded = firearm_encoder.fit_transform(y_firearm)
    y_firearm_onehot = to_categorical(y_firearm_encoded)
    
    # Caliber encoder
    caliber_encoder = LabelEncoder()
    y_caliber_encoded = caliber_encoder.fit_transform(y_caliber)
    y_caliber_onehot = to_categorical(y_caliber_encoded)
    
    # Split data
    X_train, X_test, y_firearm_train, y_firearm_test = train_test_split(
        X, y_firearm_onehot, test_size=0.2, random_state=42
    )
    
    _, _, y_caliber_train, y_caliber_test = train_test_split(
        X, y_caliber_onehot, test_size=0.2, random_state=42
    )
    
    # Train firearm model
    print("Training firearm model...")
    firearm_model = create_firearm_model(X_train.shape[1:], len(firearm_types))
    firearm_checkpoint = ModelCheckpoint(
        '../models/firearm_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    firearm_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    firearm_model.fit(
        X_train, y_firearm_train,
        validation_data=(X_test, y_firearm_test),
        epochs=50,
        batch_size=32,
        callbacks=[firearm_checkpoint, firearm_early_stopping]
    )
    
    # Train caliber model
    print("Training caliber model...")
    caliber_model = create_caliber_model(X_train.shape[1:], len(caliber_types))
    caliber_checkpoint = ModelCheckpoint(
        '../models/caliber_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    caliber_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    caliber_model.fit(
        X_train, y_caliber_train,
        validation_data=(X_test, y_caliber_test),
        epochs=50,
        batch_size=32,
        callbacks=[caliber_checkpoint, caliber_early_stopping]
    )
    
    # Save encoders
    with open('../models/firearm_encoder.pkl', 'wb') as f:
        pickle.dump(firearm_encoder, f)
    
    with open('../models/caliber_encoder.pkl', 'wb') as f:
        pickle.dump(caliber_encoder, f)
    
    print("Model training completed successfully")
    return True

def main():
    """Main startup function"""
    print("Starting system initialization...")
    
    # Get directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    
    # Save original directory
    original_dir = os.getcwd()
    
    # Change to root directory for all operations
    os.chdir(root_dir)
    
    # Check for required data directories and files
    required_data = [
        'edge-collected-gunshot-audio',
        'UrbanSound8K',
        'Data files/gunshot-audio-all-metadata.csv',
        'Data files/gunshot-audio-labels-only.csv'
    ]
    
    # Check for required model files
    required_models = [
        'models/firearm_model.h5',
        'models/caliber_model.h5',
        'models/firearm_encoder.pkl',
        'models/caliber_encoder.pkl'
    ]
    
    # Check for required GUI files
    required_gui = [
        'GUI_Files/Glock 17Semi-automatic pistol9mm caliber.gif',
        'GUI_Files/REMINGTON.gif',
        'GUI_Files/38 Smith & Wesson Special Revolver.38 caliber.gif',
        'GUI_Files/Everest-9mm-Ammo-8.jpg',
        'GUI_Files/elite-556-65-sbt-Edit__49747.jpg',
        'GUI_Files/red-shells_2d5206ed-0595-45ca-831a-0460fc82e62d.webp',
        'GUI_Files/38_158g_ammo_1200x.webp'
    ]
    
    # Check if we need to download data
    need_download = False
    for data in required_data:
        if not os.path.exists(data):
            print(f"Missing required data: {data}")
            need_download = True
    
    # Check if we need to train models
    need_train = False
    for model in required_models:
        if not os.path.exists(model):
            print(f"Missing required model: {model}")
            need_train = True
    
    # Check if we have all GUI files
    missing_gui = []
    for gui_file in required_gui:
        if not os.path.exists(gui_file):
            missing_gui.append(gui_file)
    
    if missing_gui:
        print("Error: Missing required GUI files:")
        for gui_file in missing_gui:
            print(f"- {gui_file}")
        os.chdir(original_dir)
        sys.exit(1)
    
    # Run download script if needed
    if need_download:
        print("Running download script...")
        try:
            if os.path.exists('urbanSoundsDownload.py'):
                subprocess.run(['python', 'urbanSoundsDownload.py'], check=True)
            else:
                raise FileNotFoundError("Could not find urbanSoundsDownload.py")
            print("Download script completed")
        except Exception as e:
            print(f"Error during download: {e}")
            os.chdir(original_dir)
            sys.exit(1)
    
    # Run training if needed
    if need_train:
        print("Running firearm classifier training...")
        try:
            if os.path.exists('firearm_classifier.py'):
                subprocess.run(['python', 'firearm_classifier.py'], check=True)
            else:
                raise FileNotFoundError("Could not find firearm_classifier.py")
            print("Training completed")
        except Exception as e:
            print(f"Error during training: {e}")
            os.chdir(original_dir)
            sys.exit(1)
    
    # Return to original directory
    os.chdir(original_dir)
    
    print("System initialization completed successfully")
    print("All required models and data are present")
    
    # Initialize Qt application
    app = QApplication(sys.argv)
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    
    # Create main window but don't show it yet
    main_window = GunshotDetectionApp()
    
    # Connect splash screen's fade-out animation to show main window
    def show_main_window():
        main_window.show()
        splash.close()  # Ensure splash screen is closed
    
    splash.fade_out.finished.connect(show_main_window)
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 