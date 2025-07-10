import os
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = 64
CONF_THRESH = 0.7
MODEL_PATH = 'eye_cnn_int8.onnx'  # INT8 model
DATASET_PATH = 'dataset/test'
CLASSES = ['close eyes', 'open eyes']

# Hardcoded quantization parameters from quantize_to_int8.py
SCALE = 1.0 / 255.0
ZERO_POINT = 0

def apply_clahe_grayscale(img):
    """Apply CLAHE on a grayscale OpenCV image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    # Apply CLAHE
    img = apply_clahe_grayscale(img)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize and quantize
    img = img.astype(np.float32) / 255.0
    img_q = (img * 255.0).round().clip(-128, 127).astype(np.int8)
    img_q = img_q.reshape(1, 1, IMG_SIZE, IMG_SIZE)
    return img_q

def classify_image(img_q, session):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: img_q})
    logits = outputs[0][0]

    probs = softmax(logits)
    pred = int(probs[1] > probs[0])
    conf = max(probs)
    return pred, conf

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def evaluate_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    y_true = []
    y_pred = []

    for label, class_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, class_name)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                img_q = preprocess_image(fpath)
                pred, conf = classify_image(img_q, session)
                if conf >= CONF_THRESH:
                    y_true.append(label)
                    y_pred.append(pred)
            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e}")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
