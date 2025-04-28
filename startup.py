import sys
import time
from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from main_gui import GunshotDetectionApp

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

    def start_loading_sequence(self):
        if self.current_step < len(self.loading_sequence):
            text, duration = self.loading_sequence[self.current_step]
            self.loading_text.setText(text)
            QTimer.singleShot(duration * 1000, self.next_loading_step)
        else:
            self.fade_out.start()

    def next_loading_step(self):
        self.current_step += 1
        self.start_loading_sequence()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background with rounded corners
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(43, 43, 43, 230))  # Semi-transparent dark background
        painter.drawRoundedRect(self.rect(), 10, 10)

def main():
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

if __name__ == '__main__':
    main() 