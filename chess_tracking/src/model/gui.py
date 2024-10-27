from typing import Optional
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QGridLayout, QPushButton, QHBoxLayout, QTextEdit, QGraphicsView, QGraphicsScene, QGraphicsTextItem
from PyQt5.QtGui import QPixmap, QFont
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


class GUI(QtGui.QMainWindow):
    __canvas: pg.GraphicsLayoutWidget = None
    __window_size = (640, 640)
    __console_texts: list[str] = []
    __max_buffer_size = 10
    __view: pg.ViewBox = None
    __image_item: pg.ImageItem = None

    def __init__(self, title: str = 'ChessCapture - by Rookie Technologies'):
        super(GUI, self).__init__(parent=None)
        self.setWindowTitle(title)
        self.setStyleSheet("background-color: #2C2C2C;")  # Dark grey background
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()
        availableWidth = rect.width()
        self.display_width = int(availableWidth / 3)
        self.display_height = int(self.display_width * 3/4)
        self.text_items = []

        # Create the main layout
        self.display_screen = QVBoxLayout()

        # Graphics view setup for chess projection
        self.projection_display = QVBoxLayout()
        self.chess_view = QGraphicsView(self)
        self.chess_scene = QGraphicsScene()
        self.chess_view.setScene(self.chess_scene)

         # Load chess background image and add to the scene
        pixmap = QPixmap('src/model/GUI_Images/chessBackground.png').scaled(self.display_width, self.display_width, Qt.KeepAspectRatio)
        background_item = self.chess_scene.addPixmap(pixmap)

        self.game_state_box = QTextEdit(self)  # Using QTextEdit instead of QLabel
        self.game_state_box.setStyleSheet("background-color: #D3D3D3; color: black; padding: 10px; border-radius: 5px;")  # White background and black text
        self.game_state_box.setReadOnly(True)  # Make it read-only
        self.game_state_box.resize(self.display_width,self.display_width)

        # Set the initial board state
        self.chess_board_matrix = [
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.']
        ]
        
        # Add chess pieces as text items
        self.set_chess_pieces(self.chess_board_matrix)
        
        self.projection_display.addWidget(self.chess_view)
        self.projection_display.addWidget(self.game_state_box)

        self.text_display = QVBoxLayout()
        self.text_display_box = QTextEdit(self)  # Using QTextEdit instead of QLabel
        self.text_display_box.setReadOnly(True)  # Make it read-only
        self.text_display_box.setStyleSheet("background-color: #D3D3D3; color: black; padding: 10px; border-radius: 5px;")  # White background and black text
        self.text_display.addWidget(self.text_display_box)

        # Create left column for the image display
        self.__canvas = pg.GraphicsLayoutWidget()
        self.__canvas.setStyleSheet("background-color: #2C2C2C;")
        self.__view = pg.ViewBox(enableMouse=False,lockAspect=1.0)
        self.__view.suggestPadding = lambda *_: 0.0
        self.__view.invertY()
        self.__canvas.addItem(self.__view)

        self.__image_item = pg.ImageItem(axisOrder='row-major')
        self.__view.addItem(self.__image_item)

        # Create right column layout for the console log
        self.__createConsole()

        self.live_display = QVBoxLayout()
        self.live_display.addWidget(self.__canvas)  # Add image display
        self.live_display.addWidget(self.label)      # Add console label

        # Add both columns to the main layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.live_display)
        self.main_layout.addLayout(self.projection_display)
        self.main_layout.addLayout(self.text_display)

        # Create the title label
        title_label = QLabel("ChessCapture")
        font = QFont()
        font.setPointSize(24)  # Set font size
        font.setBold(True)  # Set font weight to bold
        title_label.setFont(font)
        title_label.setStyleSheet("color: #388E3C;")  # Darker green
        title_label.setAlignment(Qt.AlignCenter)  # Center the title

        filler_label = QLabel(self)

        # Create a label for the logo
        logo_label = QLabel(self)
        logo_pixmap = QPixmap('src/model/GUI_Images/logoImage.png')  # Replace with your logo image path
        logo_label.setPixmap(logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # Set logo size

        # Create layout for title and logo
        self.header_layout = QHBoxLayout()
        self.header_layout.addWidget(logo_label, alignment=Qt.AlignRight)
        self.header_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        self.header_layout.addWidget(filler_label, alignment=Qt.AlignCenter)

        self.display_screen.addLayout(self.header_layout)
        self.display_screen.addLayout(self.main_layout)

        # Set the main layout to a central widget
        central_widget = QWidget()
        central_widget.setLayout(self.display_screen)
        self.setCentralWidget(central_widget)

        # Define tool tip settings
        QtGui.QToolTip.setFont(QtGui.QFont('Helvetica', 18))

    def set_chess_pieces(self, board_matrix):
        """Sets chess pieces as text items on the chessboard background."""
        piece_map = {
            'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
            'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', '.': None
        }
        
        cell_size = self.display_width // 10  # Set cell size based on board dimensions
        for item in self.text_items:
          self.chess_scene.removeItem(item)
        self.text_items.clear()
        for row in range(10):
            for col in range(10):
                piece = board_matrix[row][col]
                if piece_map[piece]:  # Only add items for actual pieces
                    text_item = QGraphicsTextItem(piece_map[piece])
                    text_item.setDefaultTextColor(Qt.black)
                    text_item.setFont(QFont("Arial", 30))
                    text_item.setPos(col * cell_size, row * cell_size)
                    self.chess_scene.addItem(text_item)
                    self.text_items.append(text_item)  # Store the text item

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        if len(cv_img.shape) == 2:  # grayscale or edge images
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def update_chess_board(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def set_normal(self):
        self.thread.set_filter("normal")

    def set_grayscale(self):
        self.thread.set_filter("grayscale")

    def set_canny(self):
        self.thread.set_filter("canny")

    def set_sobel(self):
        self.thread.set_filter("sobel")

    def update_text(self, text):
        """Updates the text display with the provided text"""
        self.text_display_box.append(text)

    def __createConsole(self):
        self.label = QtGui.QLabel()
        self.label.setStyleSheet('QLabel { color: black; margin: 10px; font-weight: bold; }')

    def __showConsoleText(self):
        self.text_display_box.append('\n'.join(self.__console_texts))

    def setImage(self, img):
        self.__image_item.setImage(img)

    def print(self, text: str = '', index: Optional[int] = None):
        if index is None:
            self.__console_texts.append(text)
        else:
            if len(self.__console_texts) > 0:
                self.__console_texts.pop(index)
            self.__console_texts.insert(index, text)

        if len(self.__console_texts) > self.__max_buffer_size:
            self.__console_texts.pop(1)

        self.__showConsoleText()

    def show(self):
        """Show application window"""
        super(GUI, self).show()
