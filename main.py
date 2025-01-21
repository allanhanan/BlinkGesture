import sys
import os
os.environ["QT_QPA_PLATFORM"] = "minimal"
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from threading import Thread, Lock
import time
import queue
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image


SCRIPT_DIR = __file__.rsplit('/', 1)[0]
FRAME_WIDTH = 420
FRAME_HEIGHT = 340
EYE_AR_CONSEC_FRAMES = 1
COUNTER = 0
TOTAL = 0


model_file_path = './BDmodel.keras'
model = tf.keras.models.load_model(model_file_path)

cap = None
cap_lock = Lock()
frame_queue = queue.Queue()
running = False
fps = 0.0


def preprocess_image(img):
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def capture_frame(camera_index):
    global cap
    with cap_lock:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while running:
        with cap_lock:
            if cap is None or not cap.isOpened():
                break
            ret, frame = cap.read()
            if ret:
                if frame_queue.qsize() < 1:
                    frame_queue.put(frame)
            else:
                break
        time.sleep(0.01)

def process_frame(frame):
    global COUNTER, TOTAL

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    preprocessed_image = preprocess_image(img_pil)
    predictions = model.predict(preprocessed_image)
    closed_eye_score = predictions[0][0]

    if closed_eye_score > 0.5:
        COUNTER += 1
    else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            TOTAL += 1
            print("Blink detected!")
        COUNTER = 0

    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

class BlinkApp(QtWidgets.QMainWindow):
    update_frame_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.capture_thread = None
        self.update_frame_signal.connect(self.update_frame)

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Blink Gesture Control')

        self.start_button = QtWidgets.QPushButton('Start', self)
        self.start_button.setGeometry(50, 50, 100, 50)
        self.start_button.clicked.connect(self.start_processing)

        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.setGeometry(200, 50, 100, 50)
        self.stop_button.clicked.connect(self.stop_processing)

        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setGeometry(50, 150, FRAME_WIDTH, FRAME_HEIGHT)

        self.show()

    def start_processing(self):
        global running
        running = True
        self.capture_thread = Thread(target=capture_frame, args=(0,))
        self.capture_thread.start()
        self.process_and_send_frame()

    def stop_processing(self):
        global running
        running = False
        if self.capture_thread is not None:
            self.capture_thread.join()

    def update_frame(self, frame):
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        qt_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap)

    def process_and_send_frame(self):
        if running:
            if not frame_queue.empty():
                frame = frame_queue.get()
                processed_frame = process_frame(frame)
                self.update_frame_signal.emit(processed_frame)

            QtCore.QTimer.singleShot(10, self.process_and_send_frame)


def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = BlinkApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
