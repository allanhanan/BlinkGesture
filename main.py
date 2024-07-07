import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread
import time
import queue
import tkinter as tk
from tkinter import ttk, Scale, Entry
import keyboard
from PIL import Image, ImageDraw
import pystray
import sys
import json

#init face detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

#constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 1
DOUBLE_BLINK_INTERVAL = 0.6
COUNTER = 0
TOTAL = 0
DOUBLE_BLINK_COUNT = 0
blink_times = []

#set frame res
FRAME_WIDTH = 420
FRAME_HEIGHT = 340

#start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

#queues for threading shit
frame_queue = queue.Queue()
gray_queue = queue.Queue()

#init counters
ear = 0.0
fps = 0.0


command = "alt+tab"  # Default command for double blink

#plot shit(idk what else to do to fix)
ani = None
fig = None
ax = None
ln = None
threshold_line = None
xdata, ydata = [], []
frame_window_open = False

#capture frame
def capture_frame():
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_queue.qsize() < 1:
                frame_queue.put(frame)
                gray_queue.put(gray)
        time.sleep(0.01)

#start frame capture thread
capture_thread = Thread(target=capture_frame)
capture_thread.daemon = True
capture_thread.start()

#function name is pretty self-explanatory i belive
def init_plot():
    global fig, ax, ln, threshold_line
    fig, ax = plt.subplots()
    ln, = plt.plot([], [], 'r-', label='EAR')
    threshold_line = ax.axhline(y=EYE_AR_THRESH, color='b', linestyle='--', label='Threshold')
    plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('EAR')

#this for the same plot above
def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(0.1, 0.4)
    return ln, threshold_line

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#again idk why im commenting this
def process_frame(frame, gray):
    global COUNTER, TOTAL, DOUBLE_BLINK_COUNT, ear, blink_times, command
    
    #if shit dont work (had issues with grayscale)
    if gray is None or gray.dtype != np.uint8:
        print(f"Gray image is not valid or not in correct format. Type: {gray.dtype}, Shape: {gray.shape if gray is not None else 'None'}")
        return 0.0, frame

    rects = detector(gray, 0) #detect from grayscale
    ear = 0.0 #default ear

    #finally the juicy stuff
    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        ear = (left_ear + right_ear) / 2.0 #average left and right ear for more accuracy
        
        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        if ear < EYE_AR_THRESH: #eye closed
            COUNTER += 1   #eye closed duration
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES: #threashold check
                TOTAL += 1  #update blinks
                blink_times.append(time.time())
                if len(blink_times) >= 2 and (blink_times[-1] - blink_times[-2]) <= DOUBLE_BLINK_INTERVAL: #double blink condition
                    DOUBLE_BLINK_COUNT += 1
                    keyboard.press_and_release(command) #this literally sends keypress as keyboard
            COUNTER = 0 #reset 
        
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Double Blinks: {}".format(DOUBLE_BLINK_COUNT), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) #display them details
    
    return ear, frame

#i aint commenting this
def update(frame_num): 
    global xdata, ydata, fps
    
    start_time = time.time()
    
    if not frame_queue.empty():
        frame = frame_queue.get()
        gray = gray_queue.get()
        ear, processed_frame = process_frame(frame, gray)
        
        xdata.append(len(xdata))
        ydata.append(ear)
        
        if len(xdata) > 100:
            ax.set_xlim(len(xdata)-100, len(xdata))
        
        ln.set_data(xdata, ydata)
        threshold_line.set_ydata([EYE_AR_THRESH, EYE_AR_THRESH])  #update threshold line based on gui changes
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(processed_frame, "FPS: {:.2f}".format(fps), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if frame_window_open:
            cv2.imshow("Frame", processed_frame)
    
    return ln, threshold_line


def start_processing(show_window=False):
    global ani, frame_window_open
    if ani is not None:
        plt.close(fig)
    init_plot()
    ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50, save_count=100)
    if show_window:
        plt.show(block=False)
        frame_window_open = True
    root.after(100, show_frame_window)  #show frame after 100ms

def stop_processing():
    global ani, frame_window_open
    if ani is not None:
        plt.close(fig)
    frame_window_open = False
    cv2.destroyAllWindows()

def show_frame_window():
    global frame_window_open
    if frame_window_open:
        cv2.namedWindow("Frame")
        cv2.moveWindow("Frame", 100, 100)

def update_threshold(val):
    global EYE_AR_THRESH
    EYE_AR_THRESH = float(val)
    if threshold_line:
        threshold_line.set_ydata([EYE_AR_THRESH, EYE_AR_THRESH])
    plt.draw()

def update_consec_frames(val):
    global EYE_AR_CONSEC_FRAMES
    EYE_AR_CONSEC_FRAMES = int(val)

def update_command():
    global command
    command = command_entry.get()

def save_settings():
    settings = {
        "EYE_AR_THRESH": EYE_AR_THRESH,
        "EYE_AR_CONSEC_FRAMES": EYE_AR_CONSEC_FRAMES,
        "command": command_entry.get()
    }
    with open("settings.json", "w") as f:
        json.dump(settings, f)

def load_settings():
    global EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, command
    try:
        with open("settings.json", "r") as f:
            settings = json.load(f)
            EYE_AR_THRESH = settings.get("EYE_AR_THRESH", EYE_AR_THRESH)
            EYE_AR_CONSEC_FRAMES = settings.get("EYE_AR_CONSEC_FRAMES", EYE_AR_CONSEC_FRAMES)
            command = settings.get("command", command)
    except FileNotFoundError:
        pass

def create_menu():
    menu = pystray.Menu(
        pystray.MenuItem("Show", show_window),
        pystray.MenuItem("Exit", quit_app)
    )
    return menu

def minimize_to_tray():
    image = Image.open("./icon.ico")
    menu = create_menu()
    icon = pystray.Icon("Blink Gesture", image, "Blink Gesture", menu)
    root.withdraw()
    Thread(target=icon.run, daemon=True).start()

def show_window(icon, item):
    root.deiconify()
    icon.stop()

def quit_app(icon=None, item=None):
    icon.stop() if icon else None
    save_settings()  #save settings on quit
    root.quit()
    root.destroy()

#load settings on start
load_settings()

#gui (painful)
root = tk.Tk()
root.title("Blink Gesture")

#set icon
root.iconbitmap("./icon.ico")

start_button = ttk.Button(root, text="Start", command=lambda: start_processing(False))
start_button.grid(row=0, column=0, padx=10, pady=10)

stop_button = ttk.Button(root, text="Stop", command=stop_processing)
stop_button.grid(row=0, column=1, padx=10, pady=10)

show_button = ttk.Button(root, text="Show Plot and Frame", command=lambda: start_processing(True))
show_button.grid(row=0, column=2, padx=10, pady=10)

threshold_label = ttk.Label(root, text="EAR Threshold:")
threshold_label.grid(row=1, column=0, padx=10, pady=10)

threshold_scale = Scale(root, from_=0.1, to=0.4, resolution=0.01, orient='horizontal', command=update_threshold)
threshold_scale.set(EYE_AR_THRESH)
threshold_scale.grid(row=1, column=1, padx=10, pady=10)

consec_frames_label = ttk.Label(root, text="Consec Frames:")
consec_frames_label.grid(row=2, column=0, padx=10, pady=10)

consec_frames_scale = Scale(root, from_=1, to=10, resolution=1, orient='horizontal', command=update_consec_frames)
consec_frames_scale.set(EYE_AR_CONSEC_FRAMES)
consec_frames_scale.grid(row=2, column=1, padx=10, pady=10)

command_label = ttk.Label(root, text="Command:")
command_label.grid(row=3, column=0, padx=10, pady=10)

command_entry = Entry(root)
command_entry.grid(row=3, column=1, padx=10, pady=10)
command_entry.insert(0, command)

command_button = ttk.Button(root, text="Set Command", command=update_command)
command_button.grid(row=3, column=2, padx=10, pady=10)

root.protocol("WM_DELETE_WINDOW", minimize_to_tray)

root.bind("<Unmap>", lambda event: minimize_to_tray() if root.state() == 'iconic' else None)

root.mainloop()

#finally, inner peace.