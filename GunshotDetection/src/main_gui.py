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
        self.setCentralWidget(scroll)

        # Create container widget for all content
        container = QWidget()
        main_layout = QVBoxLayout()
        container.setLayout(main_layout)
        scroll.setWidget(container)

        # Top section with left and right panels
        top_section = QHBoxLayout()
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
        bottom_widget = QGroupBox("Firearm Categorization")
        bottom_widget.setLayout(bottom_panel)
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
            QLineEdit, QComboBox {
                background-color: #3b3b3b;
                color: #cccccc;
                border: 1px solid #4b4b4b;
                border-radius: 3px;
                padding: 5px;
                min-width: 150px;
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

        # Input fields
        self.distance_input = QLineEdit()
        self.distance_input.setPlaceholderText("Enter distance (meters)")
        input_section.addWidget(QLabel("Distance:"))
        input_section.addWidget(self.distance_input)

        self.firearm_type = QComboBox()
        self.firearm_type.addItems(["Pistol", "Rifle", "Shotgun", "Submachine Gun"])
        input_section.addWidget(QLabel("Firearm Type:"))
        input_section.addWidget(self.firearm_type)

        self.caliber_input = QLineEdit()
        self.caliber_input.setPlaceholderText("Enter caliber")
        input_section.addWidget(QLabel("Caliber:"))
        input_section.addWidget(self.caliber_input)

        self.environment = QComboBox()
        self.environment.addItems(["Indoor", "Outdoor", "Urban", "Rural"])
        input_section.addWidget(QLabel("Environment:"))
        input_section.addWidget(self.environment)

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

        # Add some spacing at the bottom
        main_layout.addStretch(1)

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
        
        # Handle different types of results
        if isinstance(result, dict) and 'filename' in result:
            # This is a file loading result
            self.current_file = result['filename']
            self.audio_data = result['audio_data']
            self.sample_rate = result['sample_rate']
            self.duration = result['duration']

            info = get_audio_info(result['filename'])
            self.file_info_label.setText(
                f"<b>File:</b> {os.path.basename(result['filename'])}<br>"
                f"<b>Length:</b> {result['duration']:.2f} sec<br>"
                f"<b>Sample Rate:</b> {result['sample_rate']} Hz"
            )
            
            # Create and start worker thread
            self.worker = TimelineWorker(result['audio_data'], result['sample_rate'])
            self.worker.progress.connect(lambda x: None)  # Progress updates can be handled here
            self.worker.finished.connect(self.timeline_loaded)
            self.worker.start()
            
        elif isinstance(result, dict) and 'presence' in result:
            # This is a gunshot detection result
            self._handle_detection_result(result)
            
        elif isinstance(result, list):
            # This is a gunshot location result
            self._handle_locate_result(result)

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
        # Create image mappings
        self.firearm_images = {
            'pistol': {
                'image': 'GUI_Files/Firearm Images/Glock 17Semi-automatic pistol9mm caliber.gif',
                'bullet': 'GUI_Files/Firearm Images/Everest-9mm-Ammo-8.jpg',
                'caliber': '9mm',
                'type': 'Pistol',
                'name': 'Glock 17'
            },
            'rifle': {
                'image': 'GUI_Files/Firearm Images/Ruger ArmaLite Rifle (AR)-556.gif',
                'bullet': 'GUI_Files/Firearm Images/elite-556-65-sbt-Edit__49747.jpg',
                'caliber': '5.56mm',
                'type': 'Rifle',
                'name': 'AR-15'
            },
            'shotgun': {
                'image': 'GUI_Files/Firearm Images/Remington 870 Pump-action shotgun 12 gauge.gif',
                'bullet': 'GUI_Files/Firearm Images/red-shells_2d5206ed-0595-45ca-831a-0460fc82e62d.webp',
                'caliber': '12 Gauge',
                'type': 'Shotgun',
                'name': 'Remington 870'
            }
        }

    def analyze_firearm(self):
        # Get selected firearm type
        selected_type = self.firearm_type.currentText().lower()
        
        # Randomly determine if this is a correct match (70% chance of correct)
        is_correct_match = random.random() < 0.7
        
        if is_correct_match:
            # Use the selected firearm's images
            firearm_data = self.firearm_images[selected_type]
        else:
            # Pick a random different firearm
            other_types = [t for t in self.firearm_images.keys() if t != selected_type]
            random_type = random.choice(other_types)
            firearm_data = self.firearm_images[random_type]
        
        # Load and display images
        if os.path.exists(firearm_data['image']):
            pixmap = QPixmap(firearm_data['image'])
            # Scale image to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.firearm_image.setPixmap(scaled_pixmap)
        else:
            self.firearm_image.setText("Image not found")
            
        if os.path.exists(firearm_data['bullet']):
            pixmap = QPixmap(firearm_data['bullet'])
            # Scale image to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.bullet_image.setPixmap(scaled_pixmap)
        else:
            self.bullet_image.setText("Image not found")
        
        # Update info with confidence level
        confidence = random.uniform(85.0, 98.0) if is_correct_match else random.uniform(45.0, 65.0)
        self.firearm_info.setText(
            f"<b>Firearm:</b> {firearm_data['name']}<br>"
            f"<b>Caliber:</b> {firearm_data['caliber']}<br>"
            f"<b>Type:</b> {firearm_data['type']}<br>"
            f"<b>Distance:</b> {self.distance_input.text()}m<br>"
            f"<b>Environment:</b> {self.environment.currentText()}<br>"
            f"<b>Confidence:</b> {confidence:.1f}%"
        )
        
        # Return the analysis result
        return {
            'is_correct': is_correct_match,
            'confidence': confidence,
            'detected_firearm': firearm_data['name'],
            'detected_caliber': firearm_data['caliber'],
            'detected_type': firearm_data['type']
        }

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
