from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import sys
import qtawesome as qta
from PyQt6.QtCore import QSize
import cv2 as cv
import mediapipe as mp
import pickle
from model import KeyPointClassifier
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks as mp_draw_landmarks
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

# Initialize KeyPointClassifier with the correct model path
keypoint_classifier = KeyPointClassifier('model/keypoint_classifier/keypoint_classifier.tflite')  # Update path

# Load labels
def load_labels(label_path):
    with open(label_path, 'r', encoding='utf-8-sig') as f:
        labels = [line.strip() for line in f]
    return labels

labels = load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')  # Update path

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands

# Function to calculate landmark list
def calc_landmark_list(image, landmarks):

    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []

    # Convert absolute coordinates to pixel values
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append((landmark_x, landmark_y))

    # Normalize coordinates relative to the first landmark
    base_x, base_y = landmark_list[0]
    relative_landmarks = [(x - base_x, y - base_y) for x, y in landmark_list]

    # Flatten and normalize to fit in the range [-1, 1]
    max_value = max(max(abs(x), abs(y)) for x, y in relative_landmarks) or 1  # Avoid division by zero
    normalized_landmarks = [(x / max_value, y / max_value) for x, y in relative_landmarks]
    return [val for pair in normalized_landmarks for val in pair]  # Flatten to 1D list



# Function to process the frame
def process_frame(frame, hands, keypoint_classifier, labels):
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    recognized_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            frame = draw_landmarks(frame, hand_landmarks)
            landmark_list = calc_landmark_list(frame, hand_landmarks)  # Preprocess landmarks
            class_id = keypoint_classifier(landmark_list)  # Predict class ID
            recognized_text = labels[class_id]  # Map class ID to label

    return recognized_text, frame

# Edit landmarks
def draw_landmarks(image, hand_landmarks):
    mp_draw_landmarks(
        image,
        hand_landmarks,
        HAND_CONNECTIONS,
        landmark_drawing_spec=DrawingSpec(color=(255, 165, 0), thickness=3, circle_radius=3),
        connection_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=2),
    )
    return image

# Change camera and its size here
def initialize_camera(device=0, width=960, height=960):
    cap = cv.VideoCapture(device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class KeyPointClassifier:
    def __init__(self, model_path):
        """
        Initialize the TensorFlow Lite interpreter.

        Args:
            model_path: Path to the TFLite model file.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        """
        Predict the class ID based on the input landmarks.

        Args:
            landmark_list: A 1D list of normalized, flattened landmarks.

        Returns:
            class_id: The predicted class ID.
        """
        # Ensure input matches expected shape (e.g., [1, N])
        input_data = np.array([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get the prediction results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argmax(output_data)  # Return the class ID

# Function to release the camera
def release_camera(cap):
    cap.release()
    cv.destroyAllWindows()

# Thread for live camera feed
class LiveThread(QThread):
    frame_signal = pyqtSignal(object)
    text_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True

    def run(self):
        cap = initialize_camera()
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            recognized_text, processed_frame = process_frame(frame, hands, keypoint_classifier, labels)
            self.frame_signal.emit(processed_frame)
            self.text_signal.emit(recognized_text)

        release_camera(cap)

    def stop(self):
        self.running = False

# PyQt app
class ASLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.live_thread = None

    def start_live_recognition(self):
        if not self.live_thread:
            self.live_thread = LiveThread()
            self.live_thread.frame_signal.connect(self.update_live_feed)
            self.live_thread.text_signal.connect(self.update_output)
            self.live_thread.start()

    def update_live_feed(self, frame):
        
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.input_window.setPixmap(QPixmap.fromImage(qt_image))

    def update_output(self, text):
        self.output_text.setText(text)

    def closeEvent(self, event):
        if self.live_thread:
            self.live_thread.stop()
            self.live_thread.wait()
        event.accept()
    
    def concat(self):
        current_text = self.output_text1.text()
        recognized_text = self.output_text.text()  
        if recognized_text != "Text/letter output here":  
            new_text = current_text + recognized_text
            self.output_text1.setText(new_text)
        else:
            self.output_text1.setText("")

    def clear(self):
        self.output_text1.setText("")

    def initUI(self):
        self.main_layout = QHBoxLayout()  

        # Input 
        self.input_card = QVBoxLayout()
        self.input_label = QLabel("ASL Input")
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_label.setStyleSheet("font-size: 25px; font-weight: bold; color: white;")
        self.input_card.addWidget(self.input_label)

        self.input_window = QLabel("Input here")
        self.input_window.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_window.setStyleSheet(
            "background-color: #1E1E1E; color: white; border: 2px solid #3333FF; border-radius: 10px; font-size: 18px;")
        self.input_window.setMinimumHeight(400)  # Set larger height for the camera feed
        self.input_window.setMinimumWidth(375)
        self.input_card.addWidget(self.input_window)

        self.live_button = QPushButton("Live")
        self.live_button.setStyleSheet("background-color: #3333FF; color: white; padding: 15px;")
        self.live_button.clicked.connect(self.start_live_recognition)
        self.input_card.addWidget(self.live_button)

        self.live_button.pressed.connect(lambda: self.live_button.setStyleSheet("background-color: #5555FF; color: white;"))
        self.live_button.released.connect(lambda: self.live_button.setStyleSheet("background-color: #3333FF; color: white;"))
        self.input_card.addWidget(self.live_button, alignment=Qt.AlignmentFlag.AlignCenter)

        input_frame = QFrame()
        input_frame.setLayout(self.input_card)
        input_frame.setStyleSheet("background-color: #242424; border-radius: 20px; padding: 20px;")
        input_frame.setMinimumWidth(400)
        self.main_layout.addWidget(input_frame)
        
    
        self.main_layout.setSpacing(20)

        # Output 
        self.output_card = QVBoxLayout()
        self.output_label = QLabel("ASL Output")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setStyleSheet("font-size: 25px; font-weight: bold; color: white;")
        self.output_card.addWidget(self.output_label)

        self.output_text = QLabel("Text/letter output here")
        self.output_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_text.setStyleSheet(
            "background-color: #1E1E1E; color: white; border: 2px solid #3333FF; border-radius: 10px; font-size: 18px;")
        self.output_text.setMinimumHeight(200)  # Set larger height for the camera feed
        self.output_text.setMinimumWidth(300)
        self.output_card.addWidget(self.output_text)

        self.output_text1 = QLabel("Concat Output: ")
        self.output_text1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_text1.setStyleSheet(
            "background-color: #1E1E1E; color: white; border: 2px solid #3333FF; border-radius: 10px; font-size: 18px;")
        self.output_text1.setMinimumHeight(200)
        self.output_text1.setMinimumWidth(300)
        self.output_card.addWidget(self.output_text1)


        button_layout = QHBoxLayout()

        self.out_button = QPushButton("Concatenate")
        self.out_button.setStyleSheet("background-color: #3333FF; color: white; padding: 15px;")
        self.out_button.clicked.connect(self.concat)
        self.out_button.pressed.connect(lambda: self.out_button.setStyleSheet("background-color: #5555FF; color: white;"))
        self.out_button.released.connect(lambda: self.out_button.setStyleSheet("background-color: #3333FF; color: white;"))
        button_layout.addWidget(self.out_button)

        
        self.clearbutton = QPushButton("Clear")
        self.clearbutton.setStyleSheet("background-color: #3333FF; color: white; padding: 15px;")
        self.clearbutton.clicked.connect(self.clear)
        self.clearbutton.pressed.connect(lambda: self.clearbutton.setStyleSheet("background-color: #5555FF; color: white;"))
        self.clearbutton.released.connect(lambda: self.clearbutton.setStyleSheet("background-color: #3333FF; color: white;"))
        button_layout.addWidget(self.clearbutton)

        
        self.output_card.addLayout(button_layout)

        output_frame = QFrame()
        output_frame.setLayout(self.output_card)
        output_frame.setStyleSheet("background-color: #242424; border-radius: 20px; padding: 20px;")
        output_frame.setMinimumWidth(400)
        output_frame.setMinimumHeight(400)
        self.main_layout.addWidget(output_frame)

        self.setLayout(self.main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLApp()
    window.show()
    sys.exit(app.exec())
