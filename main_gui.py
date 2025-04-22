# main_gui.py (GUI with audio timeline-style visualization and playback)

import sys
import os
import numpy as np
import librosa
import librosa.display
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QSlider
)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt

from processing import get_audio_info, detect_gunshot, locate_gunshots, categorize_firearm
from database import init_db, query_past_files

class GunshotDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gunshot Audio Detection Platform")
        self.setMinimumSize(1200, 800)

        self.current_file = None
        self.duration = 0
        self.sample_rate = 0
        self.audio_data = None
        self.is_playing = False
        self.playback_timer = QTimer()

        init_db()
        self.init_ui()

    def init_ui(self):
        self.init_menu()

        main_layout = QHBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # LEFT PANEL — file info and controls
        self.left_panel = QVBoxLayout()
        left_widget = QGroupBox("Audio File Information")
        left_widget.setLayout(self.left_panel)
        main_layout.addWidget(left_widget, 1)

        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        self.left_panel.addWidget(self.file_info_label)

        self.detect_btn = QPushButton("Detect Gunshot")
        self.detect_btn.setToolTip("Run detection algorithm to find gunshot presence")
        self.detect_btn.clicked.connect(self.detect_gunshot_handler)
        self.detect_btn.setEnabled(False)
        self.left_panel.addWidget(self.detect_btn)

        self.detection_result_label = QLabel("")
        self.left_panel.addWidget(self.detection_result_label)

        self.locate_btn = QPushButton("Locate Gunshots")
        self.locate_btn.setVisible(False)
        self.locate_btn.clicked.connect(self.locate_gunshots_handler)
        self.left_panel.addWidget(self.locate_btn)

        self.run_all_btn = QPushButton("Run Full Analysis")
        self.run_all_btn.setToolTip("Run all analysis steps on current audio file")
        self.run_all_btn.clicked.connect(self.run_all)
        self.run_all_btn.setEnabled(False)
        self.left_panel.addWidget(self.run_all_btn)

        self.report_btn = QPushButton("Generate Report")
        self.report_btn.setToolTip("Export analysis results into LaTeX format")
        self.report_btn.clicked.connect(self.generate_report)
        self.left_panel.addWidget(self.report_btn)

        # RIGHT PANEL — waveform and controls
        self.right_panel = QVBoxLayout()
        right_widget = QGroupBox("Audio Timeline Viewer")
        right_widget.setLayout(self.right_panel)
        main_layout.addWidget(right_widget, 3)

        self.waveform_label = QLabel()
        self.waveform_label.setAlignment(Qt.AlignCenter)
        self.right_panel.addWidget(self.waveform_label)

        self.scrub_slider = QSlider(Qt.Horizontal)
        self.scrub_slider.setRange(0, 100)
        self.scrub_slider.setEnabled(False)
        self.scrub_slider.sliderReleased.connect(self.scrub_audio)
        self.right_panel.addWidget(self.scrub_slider)

        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        self.right_panel.addWidget(self.play_btn)

        self.placeholder_text = QLabel("Load an audio file to begin.")
        self.placeholder_text.setAlignment(Qt.AlignCenter)
        self.right_panel.addWidget(self.placeholder_text)

    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Audio File", self)
        open_action.setToolTip("Open an audio file for analysis")
        open_action.triggered.connect(self.load_file)
        file_menu.addAction(open_action)

        db_action = QAction("Query Past Files", self)
        db_action.setToolTip("Access previously analyzed files from database")
        db_action.triggered.connect(self.query_database)
        file_menu.addAction(db_action)

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', os.getcwd(), "Audio Files (*.wav *.mp3)")
        if fname:
            self.current_file = fname
            y, sr = librosa.load(fname, sr=None)
            self.audio_data = y
            self.duration = librosa.get_duration(y=y, sr=sr)
            self.sample_rate = sr

            info = get_audio_info(fname)
            self.file_info_label.setText(
                f"<b>File:</b> {os.path.basename(fname)}<br>"
                f"<b>Length:</b> {self.duration:.2f} sec<br>"
                f"<b>Sample Rate:</b> {sr} Hz"
            )
            self.generate_waveform(y, sr)
            self.scrub_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.placeholder_text.setText("")
            self.detect_btn.setEnabled(True)
            self.run_all_btn.setEnabled(True)

    def generate_waveform(self, y, sr):
        plt.figure(figsize=(12, 2))
        librosa.display.waveshow(y, sr=sr, x_axis='time', color='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("waveform_temp.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        image = QImage("waveform_temp.png")
        self.waveform_label.setPixmap(QPixmap.fromImage(image))

    def toggle_playback(self):
        if self.is_playing:
            sd.stop()
            self.playback_timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            if self.audio_data is not None:
                start_sample = int((self.scrub_slider.value() / 100.0) * len(self.audio_data))
                data_to_play = self.audio_data[start_sample:]
                sd.play(data_to_play, self.sample_rate)

                self.playback_timer.timeout.connect(self.update_slider_during_playback)
                self.playback_timer.start(200)

                self.play_btn.setText("Pause")
                self.is_playing = True

    def update_slider_during_playback(self):
        if self.audio_data is not None:
            pos = sd.get_stream().time
            new_value = int((pos / self.duration) * 100)
            if new_value <= 100:
                self.scrub_slider.setValue(new_value)
            else:
                self.playback_timer.stop()
                self.scrub_slider.setValue(100)
                self.play_btn.setText("Play")
                self.is_playing = False

    def scrub_audio(self):
        if self.audio_data is not None:
            sd.stop()
            self.is_playing = False
            self.play_btn.setText("Play")

    def detect_gunshot_handler(self):
        if not self.current_file:
            return
        result = detect_gunshot(self.current_file)
        if result['presence']:
            self.detection_result_label.setText(
                f"<span style='color: green; font-weight: bold;'>Gunshots Detected ({result['confidence']}%)</span>"
            )
            self.locate_btn.setVisible(True)
        else:
            self.detection_result_label.setText(
                f"<span style='color: red; font-weight: bold;'>No Gunshots Detected ({result['confidence']}%)</span>"
            )
            self.locate_btn.setVisible(False)

    def locate_gunshots_handler(self):
        if not self.current_file:
            return
        locations = locate_gunshots(self.current_file)
        print("Located gunshots at:", locations)

    def run_all(self):
        self.detect_gunshot_handler()
        self.locate_gunshots_handler()

    def generate_report(self):
        print("Generating LaTeX report... (dummy)")

    def query_database(self):
        results = query_past_files()
        print("Queried database:", results)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GunshotDetectionApp()
    gui.show()
    sys.exit(app.exec_())
