# main_gui.py (GUI with audio timeline-style visualization and playback)

import sys
import os
import numpy as np
import datetime
import librosa
from predict import predict_audio
import soundfile as sf 
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QSlider, QFrame,
    QProgressDialog, QProgressBar, QDialog, QVBoxLayout, QLineEdit,
    QComboBox, QGridLayout, QScrollArea
)
from PyQt5.QtGui import QIcon, QPixmap, QImage, QColor, QPainter, QPen, QLinearGradient
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, QSize, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random

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
        if event.button() == Qt.LeftButton:
            if self.hovered_marker is not None:
                # Check for double click (within 500ms)
                if event.timestamp() - self.last_click_time < 500:
                    # Show metadata dialog
                    dialog = GunshotMetadataDialog(self.hovered_marker, self)
                    dialog.exec_()
                self.last_click_time = event.timestamp()
                return
            self.dragging = True
            x = event.x()
            self.cursor_position = (x / self.width()) * self.duration
            # Get the main window through the widget hierarchy
            main_window = self.window()
            if hasattr(main_window, 'scrub_to_position'):
                main_window.scrub_to_position(self.cursor_position)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def mouseMoveEvent(self, event):
        if self.dragging:
            x = event.x()
            self.cursor_position = (x / self.width()) * self.duration
            # Get the main window through the widget hierarchy
            main_window = self.window()
            if hasattr(main_window, 'scrub_to_position'):
                main_window.scrub_to_position(self.cursor_position)
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
                    label = marker.get('label', 'Unknown')
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

class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 50)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_angle)
        self.timer.start(50)  # Update every 50ms

    def update_angle(self):
        self.angle = (self.angle + 10) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw spinner
        pen = QPen(QColor(100, 180, 255), 3)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        rect = QRect(5, 5, 40, 40)
        painter.drawArc(rect, self.angle * 16, 270 * 16)  # 270 degrees arc

class PopupWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setFixedSize(300, 150)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setModal(True)  # Make it modal to block parent window
        
        # Set dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QLabel {
                color: #cccccc;
                font-size: 14px;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # Add spinner
        self.spinner = LoadingSpinner()
        layout.addWidget(self.spinner, 0, Qt.AlignCenter)
        
        # Add message
        self.message_label = QLabel("Generating spectrogram...")
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)
        
        self.setLayout(layout)
        
        # Add fade-in animation
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(200)  # 200ms
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

    def showEvent(self, event):
        super().showEvent(event)
        self.animation.start()

    def closeEvent(self, event):
        self.animation.setDirection(QPropertyAnimation.Backward)
        self.animation.start()
        event.accept()

class ProcessingThread(QThread):
    finished = pyqtSignal(object)  # Changed to emit result
    error = pyqtSignal(str)
    
    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            result = self.task_func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class GunshotDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gunshot Audio Detection Platform")
        self.setMinimumSize(1200, 1000)

        # Initialize image storage
        self.firearm_images = {}
        self.bullet_images = {}
        self.load_images()

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

        # Create main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(scroll)

        # Create container widget for all content
        container = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)  # Increased spacing between elements
        main_layout.setContentsMargins(20, 20, 20, 20)  # Increased margins
        container.setLayout(main_layout)
        scroll.setWidget(container)

        # Top section with left and right panels
        top_section = QHBoxLayout()
        top_section.setSpacing(10)
        main_layout.addLayout(top_section)

        # LEFT PANEL — file info and controls
        self.left_panel = QVBoxLayout()
        left_widget = QGroupBox("Audio File Information")
        left_widget.setLayout(self.left_panel)
        top_section.addWidget(left_widget, 1)

        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        self.left_panel.addWidget(self.file_info_label)

        self.detect_btn = QPushButton("Detect Gunshot")
        self.detect_btn.setToolTip("Run detection algorithm to find gunshot presence")
        self.detect_btn.clicked.connect(self.detect_gunshot_handler)
        self.detect_btn.setEnabled(False)
        self.left_panel.addWidget(self.detect_btn)

        # Add detection parameter controls
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()
        
        # Frame duration control
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame Duration (s):"))
        self.frame_duration = QLineEdit("0.05")
        self.frame_duration.setFixedWidth(60)
        frame_layout.addWidget(self.frame_duration)
        param_layout.addLayout(frame_layout)
        
        # Energy threshold control
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Energy Threshold:"))
        self.energy_threshold = QLineEdit("0.6")
        self.energy_threshold.setFixedWidth(60)
        threshold_layout.addWidget(self.energy_threshold)
        param_layout.addLayout(threshold_layout)
        
        # Min time between shots control
        min_time_layout = QHBoxLayout()
        min_time_layout.addWidget(QLabel("Min Time Between Shots (s):"))
        self.min_time_between = QLineEdit("0.3")
        self.min_time_between.setFixedWidth(60)
        min_time_layout.addWidget(self.min_time_between)
        param_layout.addLayout(min_time_layout)
        
        param_group.setLayout(param_layout)
        self.left_panel.addWidget(param_group)

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

        # Add comprehensive report button
        self.comprehensive_report_btn = QPushButton("Generate Comprehensive Report")
        self.comprehensive_report_btn.setToolTip("Generate a detailed report with all analyses and visualizations")
        self.comprehensive_report_btn.clicked.connect(self.generate_comprehensive_report)
        self.comprehensive_report_btn.setEnabled(False)
        self.left_panel.addWidget(self.comprehensive_report_btn)

        # RIGHT PANEL — timeline and controls
        self.right_panel = QVBoxLayout()
        right_widget = QGroupBox("Audio Timeline Viewer")
        right_widget.setLayout(self.right_panel)
        top_section.addWidget(right_widget, 3)

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

        # BOTTOM PANEL — categorization
        bottom_panel = QHBoxLayout()
        bottom_panel.setSpacing(20)  # Increased spacing
        bottom_widget = QGroupBox("Firearm Categorization")
        bottom_widget.setLayout(bottom_panel)
        bottom_widget.setMinimumHeight(400)  # Increased minimum height
        main_layout.addWidget(bottom_widget)

        # Input section
        input_section = QVBoxLayout()
        input_widget = QWidget()
        input_widget.setLayout(input_section)
        input_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-radius: 5px;
                padding: 10px;
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
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4b4b4b;
            }
        """)
        bottom_panel.addWidget(input_widget, 1)

        # Add analyze button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze_firearm)
        input_section.addWidget(self.analyze_btn)

        # Results section
        results_section = QHBoxLayout()
        results_widget = QWidget()
        results_widget.setLayout(results_section)
        results_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-radius: 5px;
                padding: 10px;
            }
            QLabel {
                color: #cccccc;
                font-size: 12px;
            }
        """)
        bottom_panel.addWidget(results_widget, 2)

        # Image containers
        self.firearm_image = QLabel()
        self.firearm_image.setFixedSize(200, 200)
        self.firearm_image.setStyleSheet("border: 1px solid #4b4b4b;")
        self.firearm_image.setAlignment(Qt.AlignCenter)
        results_section.addWidget(self.firearm_image)

        self.bullet_image = QLabel()
        self.bullet_image.setFixedSize(200, 200)
        self.bullet_image.setStyleSheet("border: 1px solid #4b4b4b;")
        self.bullet_image.setAlignment(Qt.AlignCenter)
        results_section.addWidget(self.bullet_image)

        # Info labels
        info_layout = QVBoxLayout()
        self.firearm_info = QLabel("Firearm: AR-15\nCaliber: 5.56mm\nType: Rifle")
        self.firearm_info.setStyleSheet("font-size: 14px;")
        info_layout.addWidget(self.firearm_info)
        results_section.addLayout(info_layout)

        # Add more spacing at the bottom
        main_layout.addStretch(2)  # Increased stretch factor

        # Set minimum size for the container to ensure all content is visible
        container.setMinimumSize(1200, 1200)  # Increased minimum height

        # Add a spacer widget at the bottom
        bottom_spacer = QWidget()
        bottom_spacer.setMinimumHeight(100)  # Add 100 pixels of space
        main_layout.addWidget(bottom_spacer)

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

    def show_processing_popup(self, message="Processing..."):
        self.popup = PopupWindow(self)
        self.popup.message_label.setText(message)
        self.popup.show()

    def process_with_popup(self, task_func, message="Processing...", *args, **kwargs):
        # Show popup
        self.show_processing_popup(message)
        
        # Create and start processing thread
        self.processing_thread = ProcessingThread(task_func, *args, **kwargs)
        self.processing_thread.finished.connect(self._handle_processing_result)
        self.processing_thread.error.connect(self.handle_processing_error)
        self.processing_thread.start()

    def _handle_processing_result(self, result):
        self.hide_processing_popup()
        
        # Handle report generation result
        if isinstance(result, dict) and 'report_path' in result:
            self._handle_report_result(result)
            return
            
        # Handle other results
        if isinstance(result, dict) and 'filename' in result:
            self.current_file = result['filename']
            self.audio_data = result['audio_data']
            self.sample_rate = result['sample_rate']
            self.duration = result['duration']
            self.timeline_loaded(self.audio_data, self.sample_rate)
        elif isinstance(result, dict) and 'presence' in result:
            self._handle_detection_result(result)
        elif isinstance(result, list):
            self._handle_locate_result(result)
        elif isinstance(result, dict) and 'firearm_data' in result:
            # This is a firearm analysis result
            firearm_data = result['firearm_data']
            confidence = result['confidence']
            
            print("\n=== IMAGE LOADING DEBUG ===")
            print(f"Firearm image path: {firearm_data['image']}")
            print(f"Bullet image path: {firearm_data['bullet']}")
            
            # Load and display firearm image
            if os.path.exists(firearm_data['image']):
                print("Firearm image exists, loading...")
                pixmap = QPixmap(firearm_data['image'])
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.firearm_image.setPixmap(scaled_pixmap)
                    print("Firearm image loaded successfully")
                else:
                    print("Error: Firearm image is null after loading")
                    self.firearm_image.setText("Error loading firearm image")
            else:
                print(f"Error: Firearm image not found at {firearm_data['image']}")
                self.firearm_image.setText("Firearm image not found")
                
            # Load and display bullet image
            if os.path.exists(firearm_data['bullet']):
                print("Bullet image exists, loading...")
                pixmap = QPixmap(firearm_data['bullet'])
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.bullet_image.setPixmap(scaled_pixmap)
                    print("Bullet image loaded successfully")
                else:
                    print("Error: Bullet image is null after loading")
                    self.bullet_image.setText("Error loading bullet image")
            else:
                print(f"Error: Bullet image not found at {firearm_data['bullet']}")
                self.bullet_image.setText("Bullet image not found")
            
            # Update info with confidence level
            info_text = (
                f"<b>Firearm:</b> {firearm_data['name']}<br>"
                f"<b>Caliber:</b> {firearm_data['caliber']}<br>"
                f"<b>Type:</b> {firearm_data['type']}<br>"
                f"<b>Confidence:</b> {confidence*100:.1f}%"
            )
            print(f"\nUpdating info text: {info_text}")
            self.firearm_info.setText(info_text)
            
            # Store the analysis result
            self.firearm_analysis = {
                'match_percentage': confidence * 100,
                'firearm_type': firearm_data['type'],
                'ammunition': firearm_data['caliber']
            }
            print("=== END IMAGE LOADING DEBUG ===\n")

    def hide_processing_popup(self):
        if hasattr(self, 'popup'):
            self.popup.close()

    def handle_processing_error(self, error_message):
        self.hide_processing_popup()
        # You can add error handling here, like showing a message box
        print(f"Processing error: {error_message}")

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', os.getcwd(), "Audio Files (*.wav *.mp3)")
        if fname:
            # Show processing popup
            self.process_with_popup(
                self._load_file_task,
                "Generating timeline visualization...",
                fname
            )

    def _load_file_task(self, fname):
        # This function runs in the background thread
        # Only do non-GUI operations here
        y, sr = librosa.load(fname, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Return the data to be processed in the main thread
        return {
            'filename': fname,
            'audio_data': y,
            'sample_rate': sr,
            'duration': duration
        }

    def timeline_loaded(self, y, sr):
        self.timeline.set_audio_data(y, sr)
        self.scrub_slider.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.placeholder_text.setText("")
        self.detect_btn.setEnabled(True)
        self.locate_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        self.comprehensive_report_btn.setEnabled(True)
        
        # Update time display
        self.update_time_display(0)

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

    def detect_gunshot_handler(self):
        if not self.current_file:
            return
        
        # Show processing popup
        self.process_with_popup(
            self._detect_gunshot_task,
            "Detecting gunshots...",
            self.current_file
        )

    def _detect_gunshot_task(self, file_path):
        try:
            # Get parameters from GUI
            frame_duration = float(self.frame_duration.text())
            energy_threshold = float(self.energy_threshold.text())
            min_time_between = float(self.min_time_between.text())
            
            # Call detection with parameters
            from basicTimeStamping import detect_gunshots
            timestamps = detect_gunshots(
                file_path,
                frame_duration=frame_duration,
                energy_threshold=energy_threshold,
                min_time_between=min_time_between
            )
            
            return {
                'presence': len(timestamps) > 0,
                'confidence': min(95.0, len(timestamps) * 10.0),
                'timestamps': timestamps
            }
        except ValueError as e:
            print(f"Invalid parameter value: {e}")
            return {
                'presence': False,
                'confidence': 0.0,
                'timestamps': []
            }
        except Exception as e:
            print(f"Error detecting gunshots: {e}")
            return {
                'presence': False,
                'confidence': 0.0,
                'timestamps': []
            }

    def _handle_detection_result(self, result):
        if result['presence']:
            self.detection_result_label.setText(
                f"<span style='color: green; font-weight: bold;'>Gunshots Detected ({result['confidence']:.1f}%)</span>"
            )
            self.locate_btn.setEnabled(True)

            self.gunshot_locations = []

            for timestamp in result['timestamps']:
                try:
                    # === 1. Slice 1-second clip around timestamp ===
                    start_time = max(0, timestamp - 0.5)
                    end_time = min(self.duration, timestamp + 0.5)
                    start_sample = int(start_time * self.sample_rate)
                    end_sample = int(end_time * self.sample_rate)
                    clip = self.audio_data[start_sample:end_sample]

                    # === 2. Save temporary clip ===
                    temp_filename = "temp_clip.wav"
                    sf.write(temp_filename, clip, self.sample_rate)

                    # === 3. Predict with the model ===
                    label, confidence = predict_audio(temp_filename)

                    # === 4. Store marker with classification ===
                    self.gunshot_locations.append({
                        'time': timestamp,
                        'label': label,
                        'confidence': confidence
                    })

                except Exception as e:
                    print(f"Error processing timestamp {timestamp}: {e}")

            # Update timeline
            self.timeline.set_gunshot_markers(self.gunshot_locations)
            self.timeline.update()

        else:
            self.detection_result_label.setText(
                f"<span style='color: red; font-weight: bold;'>No Gunshots Detected ({result['confidence']:.1f}%)</span>"
            )
            self.locate_btn.setEnabled(False)
            self.gunshot_locations = []
            self.timeline.set_gunshot_markers([])
            self.timeline.update()

    def locate_gunshots_handler(self):
        if not self.current_file:
            return
        
        # Show processing popup
        self.process_with_popup(
            self._locate_gunshots_task,
            "Locating gunshots...",
            self.current_file
        )

    def _locate_gunshots_task(self, file_path):
        # Use real detection by default
        return locate_gunshots(file_path, use_dummy=False)

    def _handle_locate_result(self, timestamps):
        if timestamps:
            self.gunshot_locations = timestamps
            self.timeline.set_gunshot_markers(timestamps)
            self.timeline.update()  # Force update of the timeline
        else:
            self.gunshot_locations = []
            self.timeline.set_gunshot_markers([])
            self.timeline.update()  # Force update of the timeline

    def run_all(self):
        self.detect_gunshot_handler()
        self.locate_gunshots_handler()

    def generate_report(self):
        if self.audio_data is None:
            print("No audio file loaded.")
            return

        if not self.gunshot_locations:
            print("No gunshot detections to report.")
            return

        os.makedirs("../reports", exist_ok=True)

        report_filename = f"../reports/gunshot_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"

        with open(report_filename, 'w') as f:
            f.write(r"\documentclass{article}" + "\n")
            f.write(r"\usepackage{geometry}" + "\n")
            f.write(r"\geometry{margin=1in}" + "\n")
            f.write(r"\title{Gunshot Detection Report}" + "\n")
            f.write(r"\author{Gunshot Detection System}" + "\n")
            f.write(r"\date{\today}" + "\n")
            f.write(r"\begin{document}" + "\n")
            f.write(r"\maketitle" + "\n")
            f.write("\n")

            # File Info
            f.write("\\section*{Audio File Information}\n")
            f.write(f"\\textbf{{Filename}}: {os.path.basename(self.current_file)}\\\\\n")
            f.write(f"\\textbf{{Duration}}: {self.duration:.2f} seconds\\\\\n")
            f.write(f"\\textbf{{Sample Rate}}: {self.sample_rate} Hz\n")

            f.write("\n")

            # Detected Gunshots
            f.write("\\section*{Detected Gunshots}\n")
            f.write("\\begin{tabular}{|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Timestamp (s) & Label & Confidence (\\%)\\\\\n")
            f.write("\\hline\n")

            for marker in self.gunshot_locations:
                timestamp = marker['time']
                label = marker.get('label', 'Unknown')
                confidence = marker.get('confidence', 0) * 100
                f.write(f"{timestamp:.2f} & {label} & {confidence:.1f}\\\\\n")
                f.write("\\hline\n")

            f.write("\\end{tabular}\n")

            # Firearm Analysis
            if hasattr(self, 'firearm_analysis') and self.firearm_analysis:
                f.write("\n")
                f.write("\\section*{Firearm Analysis}\n")
                f.write(f"\\textbf{{Match Percentage}}: {self.firearm_analysis.get('match_percentage', '--')}\\%\\\\\n")
                f.write(f"\\textbf{{Firearm Type}}: {self.firearm_analysis.get('firearm_type', 'Unknown')}\\\\\n")
                f.write(f"\\textbf{{Ammunition}}: {self.firearm_analysis.get('ammunition', 'Unknown')}\\\\\n")

            f.write("\n")
            f.write("\\end{document}\n")

        print(f"Report generated: {report_filename}")

    def query_database(self):
        results = query_past_files()
        print("Queried database:", results)

    def load_images(self):
        # Create image mappings with correct paths
        self.firearm_images = {
            'glock': {
                'image': '../../GUI_Files/Glock 17Semi-automatic pistol9mm caliber.gif',
                'bullet': '../../GUI_Files/Everest-9mm-Ammo-8.jpg',
                'caliber': '9mm',
                'type': 'Pistol',
                'name': 'Glock 17'
            },
            'ruger': {
                'image': '../../GUI_Files/ruger.gif',  # Changed to use ruger.gif
                'bullet': '../../GUI_Files/elite-556-65-sbt-Edit__49747.jpg',
                'caliber': '5.56mm',
                'type': 'Rifle',
                'name': 'Ruger 556'
            },
            'remington': {
                'image': '../../GUI_Files/REMINGTON.gif',
                'bullet': '../../GUI_Files/red-shells_2d5206ed-0595-45ca-831a-0460fc82e62d.webp',
                'caliber': '12 Gauge',
                'type': 'Shotgun',
                'name': 'Remington 870'
            },
            'smith': {
                'image': '../../GUI_Files/38 Smith & Wesson Special Revolver.38 caliber.gif',
                'bullet': '../../GUI_Files/38_158g_ammo_1200x.webp',
                'caliber': '.38 cal',
                'type': 'Revolver',
                'name': 'Smith & Wesson'
            }
        }
        
        # Print debug info about loaded images
        print("\n=== LOADED FIREARM IMAGES ===")
        for key, data in self.firearm_images.items():
            print(f"\n{key.upper()}:")
            print(f"Image path: {data['image']}")
            print(f"Bullet path: {data['bullet']}")
            print(f"Exists: {os.path.exists(data['image'])}, {os.path.exists(data['bullet'])}")
        print("=== END LOADED IMAGES ===\n")

    def analyze_firearm(self):
        if not self.current_file:
            return
            
        # Show processing popup
        self.process_with_popup(
            self._analyze_firearm_task,
            "Analyzing firearm...",
            self.current_file
        )

    def _analyze_firearm_task(self, file_path):
        try:
            print("\n=== FIREARM ANALYSIS DEBUG ===")
            # Use the processing module to get actual firearm analysis
            from processing import categorize_firearm
            result = categorize_firearm(file_path)
            
            if result is None:
                print("Error: No result returned from categorize_firearm")
                return None
                
            print(f"Analysis result: {result}")
            
            # Get the firearm type from the result
            firearm_type = result['firearm'].lower()
            print(f"Looking for firearm type: {firearm_type}")
            
            # Find matching firearm data
            firearm_data = None
            for key, data in self.firearm_images.items():
                # Check if the firearm type contains any of our keys or vice versa
                if key in firearm_type or firearm_type in key or \
                   data['name'].lower() in firearm_type or firearm_type in data['name'].lower():
                    firearm_data = data
                    print(f"Found matching firearm data: {data}")
                    break
            
            if firearm_data is None:
                print("Warning: No matching firearm data found, using default")
                firearm_data = self.firearm_images['glock']  # Default to Glock if no match
            
            # Return the data to be processed in the main thread
            return {
                'label': result['firearm'],
                'confidence': result['match_confidence'] / 100.0,  # Convert percentage to decimal
                'firearm_data': firearm_data
            }
        except Exception as e:
            print(f"\n=== ERROR IN FIREARM ANALYSIS ===")
            print(f"Error details: {str(e)}")
            print("=== END ERROR ===\n")
            return None

    def generate_comprehensive_report(self):
        if self.audio_data is None:
            print("No audio file loaded.")
            return

        # Show processing popup
        self.process_with_popup(
            self._generate_report_task,
            "Generating comprehensive report...",
            self.current_file
        )

    def _generate_report_task(self, file_path):
        try:
            # Create reports directory if it doesn't exist
            os.makedirs("../reports", exist_ok=True)
            
            # Generate timestamp for report filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"../reports/gunshot_report_{timestamp}.pdf"
            
            # Run all analyses
            print("Running gunshot detection...")
            detection_result = detect_gunshot(file_path)
            
            print("Locating gunshots...")
            gunshot_locations = locate_gunshots(file_path)
            
            print("Analyzing firearm...")
            firearm_analysis = categorize_firearm(file_path)
            
            # Generate spectrogram visualization
            print("Generating spectrogram...")
            y, sr = librosa.load(file_path, sr=None)
            hop_length = 512
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
            
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
            for marker in gunshot_locations:
                plt.axvline(x=marker['time'], color='r', linestyle='--', alpha=0.7)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram with Detected Gunshots')
            plt.tight_layout()
            spectrogram_path = f"../reports/spectrogram_{timestamp}.png"
            plt.savefig(spectrogram_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate firearm analysis visualization
            print("Generating firearm analysis visualization...")
            plt.figure(figsize=(8, 4))
            plt.bar(['Firearm Type', 'Caliber', 'Match Confidence'],
                    [100, 100, firearm_analysis.get('match_confidence', 0)],
                    color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            plt.title('Firearm Analysis Results')
            plt.ylim(0, 100)
            plt.ylabel('Confidence (%)')
            firearm_analysis_path = f"../reports/firearm_analysis_{timestamp}.png"
            plt.savefig(firearm_analysis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate PDF using ReportLab
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            # Create PDF document with larger left margin
            doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                                  leftMargin=72, rightMargin=72,
                                  topMargin=72, bottomMargin=72)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=0  # Left align
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue,
                alignment=0  # Left align
            )
            
            # Content
            content = []
            
            # Title with line
            content.append(Paragraph("GUNSHOT DETECTION REPORT", title_style))
            content.append(HRFlowable(width="100%", thickness=2, color=colors.darkblue))
            content.append(Spacer(1, 30))  # Increased from 20 to 30
            
            # File Information
            content.append(Paragraph("AUDIO FILE INFORMATION", heading_style))
            content.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
            content.append(Spacer(1, 15))  # Added spacer after line
            file_info = [
                ["PARAMETER", "VALUE"],
                ["Filename", os.path.basename(file_path)],
                ["Duration", f"{self.duration:.2f} seconds"],
                ["Sample Rate", f"{self.sample_rate} Hz"]
            ]
            file_table = Table(file_info, colWidths=[2*inch, 4*inch])
            file_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.darkblue),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            content.append(file_table)
            content.append(Spacer(1, 30))  # Increased from 20 to 30
            
            # Spectrogram
            content.append(Paragraph("SPECTROGRAM ANALYSIS", heading_style))
            content.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
            content.append(Spacer(1, 15))  # Added spacer after line
            spectrogram_img = Image(spectrogram_path, width=6*inch, height=2.4*inch)
            content.append(spectrogram_img)
            content.append(Spacer(1, 30))  # Increased from 20 to 30
            
            # Detected Gunshots
            content.append(Paragraph("DETECTED GUNSHOTS", heading_style))
            content.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
            content.append(Spacer(1, 15))  # Added spacer after line
            gunshot_data = [["TIMESTAMP (s)", "TYPE", "CALIBER", "CONFIDENCE (%)"]]
            for marker in gunshot_locations:
                gunshot_data.append([
                    f"{marker['time']:.2f}",
                    marker.get('type', 'Unknown'),
                    marker.get('caliber', 'Unknown'),
                    f"{marker.get('confidence', 0) * 100:.1f}"
                ])
            gunshot_table = Table(gunshot_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            gunshot_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.darkblue),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            content.append(gunshot_table)
            content.append(Spacer(1, 30))  # Increased from 20 to 30
            
            # Firearm Analysis
            content.append(Paragraph("FIREARM ANALYSIS", heading_style))
            content.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
            content.append(Spacer(1, 15))  # Added spacer after line
            
            # Get firearm and bullet images
            firearm_type = firearm_analysis.get('firearm', '').lower()
            firearm_data = None
            for key, data in self.firearm_images.items():
                if key in firearm_type or firearm_type in key or \
                   data['name'].lower() in firearm_type or firearm_type in data['name'].lower():
                    firearm_data = data
                    break
            
            if firearm_data:
                # Create a table for the images
                img_table_data = [
                    ["FIREARM IMAGE", "BULLET IMAGE"],
                    [Image(firearm_data['image'], width=2*inch, height=2*inch),
                     Image(firearm_data['bullet'], width=2*inch, height=2*inch)]
                ]
                img_table = Table(img_table_data, colWidths=[3*inch, 3*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.darkblue),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6)
                ]))
                content.append(img_table)
                content.append(Spacer(1, 30))  # Increased from 20 to 30
            
            # Firearm analysis chart
            firearm_img = Image(firearm_analysis_path, width=4.8*inch, height=2.4*inch)
            content.append(firearm_img)
            
            firearm_data = [
                ["PARAMETER", "VALUE"],
                ["Firearm Type", firearm_analysis.get('firearm', 'Unknown')],
                ["Caliber", firearm_analysis.get('caliber', 'Unknown')],
                ["Match Confidence", f"{firearm_analysis.get('match_confidence', 0):.1f}%"]
            ]
            firearm_table = Table(firearm_data, colWidths=[2*inch, 4*inch])
            firearm_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.darkblue),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            content.append(firearm_table)
            content.append(Spacer(1, 30))  # Increased from 20 to 30
            
            # Summary
            content.append(Paragraph("SUMMARY", heading_style))
            content.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
            content.append(Spacer(1, 15))  # Added spacer after line
            summary_text = """
            This report contains the results of the comprehensive gunshot detection and analysis performed on the audio file. 
            The analysis includes detection of gunshot events, their timestamps, and classification of the firearm type. 
            The confidence levels are provided for each detection and classification.
            """
            content.append(Paragraph(summary_text, styles['Normal']))
            
            # Build PDF
            doc.build(content)
            
            print(f"Report generated: {pdf_filename}")
            return {
                'success': True,
                'pdf_path': pdf_filename,
                'spectrogram_path': spectrogram_path,
                'firearm_analysis_path': firearm_analysis_path
            }
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _handle_report_result(self, result):
        if result['success']:
            print("Report generation completed successfully!")
            # Open the PDF file if it exists
            if 'pdf_path' in result and os.path.exists(result['pdf_path']):
                os.startfile(result['pdf_path'])
            else:
                # Fall back to opening the directory
                os.startfile(os.path.dirname(result['pdf_path']))
        else:
            print(f"Error generating report: {result['error']}")

def dummy_locate_gunshots(audio_file):
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

def locate_gunshots(audio_file, use_dummy=False):
    if use_dummy:
        return dummy_locate_gunshots(audio_file)
    else:
        # Use the real detection from processing.py
        from processing import locate_gunshots as real_locate_gunshots
        return real_locate_gunshots(audio_file)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GunshotDetectionApp()
    gui.show()
    sys.exit(app.exec_())
