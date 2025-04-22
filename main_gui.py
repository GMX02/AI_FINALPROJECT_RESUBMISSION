# main_gui.py (GUI with audio timeline-style visualization and playback)

import sys
import os
import numpy as np
import librosa
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QSlider, QFrame,
    QProgressDialog, QProgressBar, QDialog, QVBoxLayout
)
from PyQt5.QtGui import QIcon, QPixmap, QImage, QColor, QPainter, QPen, QLinearGradient
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, QSize, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from processing import get_audio_info, detect_gunshot, locate_gunshots, categorize_firearm
from database import init_db, query_past_files

class TimelineWorker(QThread):
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(int)

    def __init__(self, audio_data, sample_rate):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def run(self):
        # Simulate some processing time
        self.progress.emit(0)
        # Process audio data
        self.progress.emit(50)
        # Emit the results
        self.finished.emit(self.audio_data, self.sample_rate)
        self.progress.emit(100)

class GunshotMetadataDialog(QDialog):
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gunshot Details")
        self.setFixedSize(300, 200)
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QLabel {
                color: #cccccc;
                font-size: 12px;
            }
            QPushButton {
                background-color: #3b3b3b;
                color: #cccccc;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4b4b4b;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Add metadata fields
        layout.addWidget(QLabel(f"<b>Time:</b> {metadata['time']:.2f} seconds"))
        layout.addWidget(QLabel(f"<b>Confidence:</b> {metadata['confidence']*100:.1f}%"))
        layout.addWidget(QLabel(f"<b>Type:</b> {metadata['type']}"))
        layout.addWidget(QLabel(f"<b>Caliber:</b> {metadata['caliber']}"))
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)

class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.audio_data = None
        self.sample_rate = None
        self.cursor_position = 0
        self.gunshot_markers = []
        self.hovered_marker = None
        self.setMouseTracking(True)
        self.dragging = False
        self.setCursor(Qt.PointingHandCursor)
        self.last_click_time = 0

    def set_audio_data(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.update()

    def set_cursor_position(self, position):
        self.cursor_position = position
        self.update()

    def set_gunshot_markers(self, markers):
        self.gunshot_markers = markers
        self.update()

    def mousePressEvent(self, event):
        current_time = event.timestamp()
        if event.button() == Qt.LeftButton:
            if self.hovered_marker is not None:
                # Check for double click (within 500ms)
                if current_time - self.last_click_time < 500:
                    # Show metadata dialog
                    dialog = GunshotMetadataDialog(self.hovered_marker, self)
                    dialog.exec_()
                self.last_click_time = current_time
                return
            self.dragging = True
            x = event.x()
            self.cursor_position = (x / self.width()) * self.duration
            self.parent().parent().parent().scrub_to_position(self.cursor_position)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def mouseMoveEvent(self, event):
        if self.dragging:
            x = event.x()
            self.cursor_position = (x / self.width()) * self.duration
            self.parent().parent().parent().scrub_to_position(self.cursor_position)
        
        # Check for marker hover
        self.hovered_marker = None
        if self.audio_data is not None:
            x = event.x()
            width = self.width()
            for marker in self.gunshot_markers:
                marker_x = int((marker['time'] / self.duration) * width)
                if abs(x - marker_x) < 10:  # 10 pixel hover radius
                    self.hovered_marker = marker
                    break
        self.update()

    def paintEvent(self, event):
        if self.audio_data is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        width = self.width()
        height = self.height()
        center_y = height // 2

        # Draw time grid
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        grid_spacing = width // 10  # 10 segments
        for x in range(0, width, grid_spacing):
            painter.drawLine(x, 0, x, height)
            time = (x / width) * self.duration
            painter.drawText(x + 2, height - 5, f"{time:.1f}s")

        # Draw waveform with gradient
        if self.audio_data is not None:
            gradient = QLinearGradient(0, 0, 0, height)
            gradient.setColorAt(0, QColor(100, 180, 255))
            gradient.setColorAt(1, QColor(60, 120, 200))
            painter.setPen(QPen(gradient, 1))

            samples_per_pixel = len(self.audio_data) / width
            for x in range(width):
                start_idx = int(x * samples_per_pixel)
                end_idx = int((x + 1) * samples_per_pixel)
                if start_idx < len(self.audio_data):
                    chunk = self.audio_data[start_idx:end_idx]
                    if len(chunk) > 0:
                        max_val = np.max(np.abs(chunk))
                        y_height = int(max_val * height * 0.8)
                        painter.drawLine(x, center_y - y_height//2, x, center_y + y_height//2)

        # Draw gunshot markers with glow effect and labels
        if self.gunshot_markers:
            for i, marker in enumerate(self.gunshot_markers):
                x = int((marker['time'] / self.duration) * width)
                
                # Draw glow
                glow_color = QColor(255, 100, 100, 50)
                for j in range(5, 0, -1):
                    painter.setPen(QPen(glow_color, j*2))
                    painter.drawLine(x, 0, x, height)
                
                # Draw marker line
                if marker == self.hovered_marker:
                    painter.setPen(QPen(QColor(255, 50, 50), 3))
                else:
                    painter.setPen(QPen(QColor(255, 50, 50), 2))
                painter.drawLine(x, 0, x, height)
                
                # Only draw label if hovering
                if marker == self.hovered_marker:
                    label = f"Gunshot {i+1}"
                    font = painter.font()
                    font.setPointSize(10)
                    painter.setFont(font)
                    painter.setPen(QPen(QColor(255, 255, 255)))
                    
                    # Draw label background
                    text_rect = painter.fontMetrics().boundingRect(label)
                    label_rect = QRect(x - text_rect.width()//2 - 5, 5, 
                                     text_rect.width() + 10, text_rect.height() + 5)
                    painter.fillRect(label_rect, QColor(0, 0, 0, 180))
                    
                    # Draw label text
                    painter.drawText(label_rect, Qt.AlignCenter, label)

        # Draw playhead cursor with glow
        cursor_x = int((self.cursor_position / self.duration) * width)
        # Draw cursor glow
        glow_color = QColor(255, 255, 255, 30)
        for i in range(6, 0, -1):
            painter.setPen(QPen(glow_color, i*2))
            painter.drawLine(cursor_x, 0, cursor_x, height)
        # Draw cursor line
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(cursor_x, 0, cursor_x, height)
        # Draw cursor handle
        handle_height = 10
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPolygon([
            QPoint(cursor_x - 5, 0),
            QPoint(cursor_x + 5, 0),
            QPoint(cursor_x, handle_height)
        ])
        painter.drawPolygon([
            QPoint(cursor_x - 5, height),
            QPoint(cursor_x + 5, height),
            QPoint(cursor_x, height - handle_height)
        ])

    @property
    def duration(self):
        if self.audio_data is not None and self.sample_rate is not None:
            return len(self.audio_data) / self.sample_rate
        return 0

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
        self.playback_position = 0
        self.gunshot_locations = []
        self.current_stream = None

        init_db()
        self.init_ui()
        
        # Update timer for smooth playback visualization
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(16)  # ~60fps

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
        self.detection_result_label.setStyleSheet("color: #cccccc;")
        self.left_panel.addWidget(self.detection_result_label)

        self.locate_btn = QPushButton("Locate Gunshots")
        self.locate_btn.setToolTip("Find exact positions of gunshots in the audio")
        self.locate_btn.clicked.connect(self.locate_gunshots_handler)
        self.locate_btn.setEnabled(False)
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

        # RIGHT PANEL — timeline and controls
        self.right_panel = QVBoxLayout()
        right_widget = QGroupBox("Audio Timeline Viewer")
        right_widget.setLayout(self.right_panel)
        main_layout.addWidget(right_widget, 3)

        # Create timeline widget
        self.timeline = TimelineWidget()
        self.right_panel.addWidget(self.timeline)

        # Controls layout with modern styling
        controls_layout = QHBoxLayout()
        controls_widget = QWidget()
        controls_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #3b3b3b;
                border: none;
                border-radius: 3px;
                padding: 5px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #4b4b4b;
            }
            QPushButton:pressed {
                background-color: #5b5b5b;
            }
            QSlider::groove:horizontal {
                border: 1px solid #4b4b4b;
                height: 4px;
                background: #2b2b2b;
                margin: 0px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6b6b6b;
                border: none;
                width: 18px;
                margin: -8px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #7b7b7b;
            }
            QLabel {
                color: #cccccc;
                font-size: 12px;
            }
        """)
        
        controls_layout.setContentsMargins(10, 5, 10, 5)
        controls_widget.setLayout(controls_layout)
        
        # Play button with icon
        self.play_btn = QPushButton()
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        
        # Create play/pause icons using QPixmap
        play_pixmap = QPixmap(20, 20)
        play_pixmap.fill(Qt.transparent)
        painter = QPainter(play_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(Qt.white)
        # Draw play triangle
        points = [QPoint(5, 5), QPoint(15, 10), QPoint(5, 15)]
        painter.drawPolygon(points)
        painter.end()
        
        pause_pixmap = QPixmap(20, 20)
        pause_pixmap.fill(Qt.transparent)
        painter = QPainter(pause_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.white, 2))
        # Draw pause bars
        painter.drawLine(5, 5, 5, 15)
        painter.drawLine(15, 5, 15, 15)
        painter.end()
        
        self.play_icon = QIcon(play_pixmap)
        self.pause_icon = QIcon(pause_pixmap)
        self.play_btn.setIcon(self.play_icon)
        self.play_btn.setIconSize(QSize(20, 20))
        controls_layout.addWidget(self.play_btn)
        
        self.scrub_slider = QSlider(Qt.Horizontal)
        self.scrub_slider.setRange(0, 1000)
        self.scrub_slider.setEnabled(False)
        self.scrub_slider.sliderMoved.connect(self.scrub_audio)
        controls_layout.addWidget(self.scrub_slider)

        self.right_panel.addWidget(controls_widget)

        # Time display
        time_display = QHBoxLayout()
        time_display.setContentsMargins(10, 0, 10, 5)
        
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setAlignment(Qt.AlignLeft)
        time_display.addWidget(self.current_time_label)
        
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setAlignment(Qt.AlignRight)
        time_display.addWidget(self.total_time_label)
        
        self.right_panel.addLayout(time_display)

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
            # Show loading dialog
            progress = QProgressDialog("Generating timeline visualization...", None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Loading")
            progress.setCancelButton(None)  # Remove cancel button
            progress.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
            progress.setStyleSheet("""
                QProgressDialog {
                    background-color: #2b2b2b;
                    color: #cccccc;
                }
                QProgressBar {
                    border: 1px solid #4b4b4b;
                    border-radius: 3px;
                    background-color: #2b2b2b;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #6b6b6b;
                    border-radius: 2px;
                }
            """)
            progress.setMinimumDuration(0)  # Show immediately
            progress.setValue(0)
            
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
            
            # Create and start worker thread
            self.worker = TimelineWorker(y, sr)
            self.worker.progress.connect(progress.setValue)
            self.worker.finished.connect(lambda y, sr: self.timeline_loaded(y, sr, progress))
            self.worker.start()

    def timeline_loaded(self, y, sr, progress):
        self.timeline.set_audio_data(y, sr)
        self.scrub_slider.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.placeholder_text.setText("")
        self.detect_btn.setEnabled(True)
        self.locate_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        
        # Update time display
        self.update_time_display(0)
        
        # Close progress dialog
        progress.close()

    def update_visualization(self):
        if self.is_playing and self.current_stream:
            try:
                pos = self.current_stream.time
                self.timeline.set_cursor_position(pos)
                new_value = int((pos / self.duration) * 1000)
                if new_value <= 1000:
                    self.scrub_slider.setValue(new_value)
                else:
                    self.stop_playback()
            except:
                self.stop_playback()

    def stop_playback(self):
        if self.current_stream:
            try:
                self.current_stream.stop()
                self.current_stream.close()
            except:
                pass
            self.current_stream = None
            
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
            
        self.play_btn.setIcon(self.play_icon)
        self.is_playing = False
        
        # Update final position
        if self.audio_data is not None:
            current_time = self.playback_position / self.sample_rate
            self.timeline.set_cursor_position(current_time)
            self.scrub_slider.setValue(int((current_time / self.duration) * 1000))

    def scrub_to_position(self, position):
        if self.audio_data is not None:
            value = int((position / self.duration) * 1000)
            self.scrub_slider.setValue(value)
            self.timeline.set_cursor_position(position)
            if self.is_playing:
                self.stop_playback()
                self.toggle_playback()  # Restart playback from new position

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            if self.audio_data is not None:
                start_sample = int((self.scrub_slider.value() / 1000.0) * len(self.audio_data))
                self.playback_position = start_sample
                
                try:
                    # Create a non-blocking stream
                    self.current_stream = sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        blocksize=1024,
                        dtype='float32'
                    )
                    self.current_stream.start()
                    
                    # Start the update timer
                    self.update_timer = QTimer()
                    self.update_timer.timeout.connect(self.update_playback)
                    self.update_timer.start(16)  # ~60fps
                    
                    self.play_btn.setIcon(self.pause_icon)
                    self.is_playing = True
                except Exception as e:
                    print(f"Audio playback error: {e}")
                    self.stop_playback()

    def update_playback(self):
        if not self.is_playing or self.current_stream is None:
            return
            
        try:
            # Calculate how many samples to write
            remaining = len(self.audio_data) - self.playback_position
            if remaining <= 0:
                self.stop_playback()
                return
                
            # Write a chunk of audio
            chunk_size = min(1024, remaining)
            chunk = self.audio_data[self.playback_position:self.playback_position + chunk_size]
            self.current_stream.write(chunk)
            self.playback_position += chunk_size
            
            # Update visualization
            current_time = self.playback_position / self.sample_rate
            self.timeline.set_cursor_position(current_time)
            slider_value = int((current_time / self.duration) * 1000)
            if slider_value <= 1000:
                self.scrub_slider.setValue(slider_value)
            
            # Update time display
            self.update_time_display(current_time)
                
        except Exception as e:
            print(f"Playback update error: {e}")
            self.stop_playback()

    def update_time_display(self, current_time):
        # Format current time
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        self.current_time_label.setText(f"{minutes:02d}:{seconds:02d}")
        
        # Format total time
        total_minutes = int(self.duration // 60)
        total_seconds = int(self.duration % 60)
        self.total_time_label.setText(f"{total_minutes:02d}:{total_seconds:02d}")

    def scrub_audio(self):
        if self.audio_data is not None:
            pos = (self.scrub_slider.value() / 1000.0) * self.duration
            self.timeline.set_cursor_position(pos)
            if self.is_playing:
                self.stop_playback()
                self.toggle_playback()  # Restart playback from new position

    def locate_gunshots_handler(self):
        if not self.current_file:
            return
        locations = locate_gunshots(self.current_file)
        self.timeline.set_gunshot_markers(locations)

    def detect_gunshot_handler(self):
        if not self.current_file:
            return
        result = detect_gunshot(self.current_file)
        if result['presence']:
            self.detection_result_label.setText(
                f"<span style='color: green; font-weight: bold;'>Gunshots Detected ({result['confidence']}%)</span>"
            )
            self.locate_btn.setEnabled(True)
        else:
            self.detection_result_label.setText(
                f"<span style='color: red; font-weight: bold;'>No Gunshots Detected ({result['confidence']}%)</span>"
            )
            self.locate_btn.setEnabled(False)

    def run_all(self):
        self.detect_gunshot_handler()
        self.locate_gunshots_handler()

    def generate_report(self):
        print("Generating LaTeX report... (dummy)")

    def query_database(self):
        results = query_past_files()
        print("Queried database:", results)

def locate_gunshots(audio_file):
    # Dummy function that returns 3 evenly spaced gunshots
    y, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Create 3 evenly spaced gunshots with metadata
    gunshots = []
    gunshot_types = ["Pistol", "Rifle", "Shotgun"]
    calibers = ["9mm", "5.56mm", "12 Gauge"]
    
    for i in range(3):
        time = (i + 1) * duration / 4  # Space them evenly
        gunshots.append({
            'time': time,
            'confidence': 0.95 - (i * 0.05),  # Slightly decreasing confidence
            'type': gunshot_types[i],
            'caliber': calibers[i],
            'energy': f"{1000 - (i * 100)} J",  # Example energy values
            'peak_pressure': f"{150 + (i * 10)} dB",  # Example pressure values
            'frequency': f"{1000 + (i * 500)} Hz"  # Example frequency values
        })
    
    return gunshots

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GunshotDetectionApp()
    gui.show()
    sys.exit(app.exec_())
