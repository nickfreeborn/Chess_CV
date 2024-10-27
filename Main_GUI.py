from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QGridLayout, QPushButton, QHBoxLayout, QTextEdit, QGraphicsView, QGraphicsScene, QGraphicsTextItem
from PyQt5.QtGui import QPixmap, QFont
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.filter_type = "normal"  # Default filter
        

    def run(self):
        # capture from webcam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                if self.filter_type == "grayscale":
                    cv_img = self.apply_grayscale(cv_img)
                elif self.filter_type == "canny":
                    cv_img = self.apply_canny(cv_img)
                elif self.filter_type == "sobel":
                    cv_img = self.apply_sobel(cv_img)
                # emit the frame with the selected filter
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def apply_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_canny(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 100)
        return edges

    def apply_sobel(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # Normalize to bring to range [0, 255] and convert to uint8
        sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
        sobel_mag = sobel_mag.astype(np.uint8)  # Convert to 8-bit
        return sobel_mag

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def set_filter(self, filter_type):
        self.filter_type = filter_type


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Filter GUI")
        self.setStyleSheet("background-color: #2C2C2C;")  # Dark grey background
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()
        availableWidth = rect.width()
        self.display_width = int(availableWidth / 3)
        self.display_height = int(self.display_width * 3/4)

        # create the label that holds the image
        display_screen = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        # create buttons for filters
        self.btn_normal = QPushButton("Normal")
        self.btn_gray = QPushButton("Grayscale")
        self.btn_canny = QPushButton("Canny Edges")
        self.btn_sobel = QPushButton("Sobel Magnitude")

        # Set button text color to white
        self.btn_normal.setStyleSheet("color: black; background-color: #D3D3D3;")
        self.btn_gray.setStyleSheet("color: black; background-color: #D3D3D3;")
        self.btn_canny.setStyleSheet("color: black; background-color: #D3D3D3;")
        self.btn_sobel.setStyleSheet("color: black; background-color: #D3D3D3;")

        # create a layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_normal)
        button_layout.addWidget(self.btn_gray)
        button_layout.addWidget(self.btn_canny)
        button_layout.addWidget(self.btn_sobel)

        # create a vertical layout and add the image label and button layout
        live_display = QVBoxLayout()
        live_display.addWidget(self.image_label)
        live_display.addLayout(button_layout)

        # Graphics view setup for chess projection
        projection_display = QVBoxLayout()
        self.chess_view = QGraphicsView(self)
        self.chess_scene = QGraphicsScene()
        self.chess_view.setScene(self.chess_scene)
        
        # Load chess background image and add to the scene
        pixmap = QPixmap('GUI_Images/chessBackground.png').scaled(self.display_width, self.display_width, Qt.KeepAspectRatio)
        background_item = self.chess_scene.addPixmap(pixmap)

        self.game_state_box = QTextEdit(self)  # Using QTextEdit instead of QLabel
        self.game_state_box.setStyleSheet("background-color: #D3D3D3; color: black; padding: 10px; border-radius: 5px;")  # White background and black text
        self.game_state_box.setReadOnly(True)  # Make it read-only
        self.game_state_box.resize(self.display_width,self.display_width)

        # Set the initial board state
        self.chess_board_matrix = [
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R','.'],
            ['.','P', 'P', 'P', 'P', 'P', 'P', 'P', 'P','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.'],
            ['.','p', 'p', 'p', 'p', 'p', 'p', 'p', 'p','.'],
            ['.','r', 'n', 'b', 'q', 'k', 'b', 'n', 'r','.'],
            ['.','.', '.', '.', '.', '.', '.', '.', '.','.']
        ]
        
        # Add chess pieces as text items
        self.set_chess_pieces(self.chess_board_matrix)
        
        projection_display.addWidget(self.chess_view)
        projection_display.addWidget(self.game_state_box)

        text_display = QVBoxLayout()
        self.text_display_box = QTextEdit(self)  # Using QTextEdit instead of QLabel
        self.text_display_box.setReadOnly(True)  # Make it read-only
        self.text_display_box.setStyleSheet("background-color: #D3D3D3; color: black; padding: 10px; border-radius: 5px;")  # White background and black text
        text_display.addWidget(self.text_display_box)

        divided_screen = QHBoxLayout()
        divided_screen.addLayout(live_display)
        divided_screen.addLayout(projection_display)
        divided_screen.addLayout(text_display)

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
        logo_pixmap = QPixmap('GUI_Images/logoImage.png')  # Replace with your logo image path
        logo_label.setPixmap(logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # Set logo size

        # Create layout for title and logo
        header_layout = QHBoxLayout()
        header_layout.addWidget(logo_label, alignment=Qt.AlignRight)
        header_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        header_layout.addWidget(filler_label, alignment=Qt.AlignCenter)

        display_screen.addLayout(header_layout)
        display_screen.addLayout(divided_screen)

        self.setLayout(display_screen)

        # create the video capture thread
        self.thread = VideoThread()

        # connect button signals to filter selection slots
        self.btn_normal.clicked.connect(self.set_normal)
        self.btn_gray.clicked.connect(self.set_grayscale)
        self.btn_canny.clicked.connect(self.set_canny)
        self.btn_sobel.clicked.connect(self.set_sobel)

        # connect the thread's signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # Update text display with button clicks (optional)
        self.btn_normal.clicked.connect(lambda: self.update_text("Normal Filter Applied"))
        self.btn_gray.clicked.connect(lambda: self.update_text("Grayscale Filter Applied"))
        self.btn_canny.clicked.connect(lambda: self.update_text("Canny Filter Applied"))
        self.btn_sobel.clicked.connect(lambda: self.update_text("Sobel Filter Applied"))

    
    def set_chess_pieces(self, board_matrix):
        """Sets chess pieces as text items on the chessboard background."""
        piece_map = {
            'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
            'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', '.': None
        }
        
        cell_size = self.display_width // 10  # Set cell size based on board dimensions
        
        for row in range(10):
            for col in range(10):
                piece = board_matrix[row][col]
                if piece_map[piece]:  # Only add items for actual pieces
                    text_item = QGraphicsTextItem(piece_map[piece])
                    text_item.setDefaultTextColor(Qt.black)
                    text_item.setFont(QFont("Arial", 30))
                    text_item.setPos(col * cell_size, row * cell_size)
                    self.chess_scene.addItem(text_item)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
