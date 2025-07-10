# BlinkGesture

Send keyboard commands handsfree by blinking twice / Double Blink gesture

now uses a CNN to detect blinks more efficiently


---

## Overview

1. **Capture & Detection**  
   - OpenCV `VideoCapture` grabs frames.  
   - Haar cascade locates eyes at intervals.  
   - KCF trackers follow detected eye regions between detections.

2. **Eye‑Strip Extraction**  
   - Compute midpoint, angle, distance between eyes.  
   - Affine warp to align eyes horizontally.  
   - Output a fixed‑size strip image.

3. **Classification**  
   - CLAHE to enhance contrast.  
   - Sliding 64×64 window scans the strip at two scales.  
   - ONNX INT8 model classifies patches as open/closed.  
   - Voting and thresholds yield a frame‑level state.

4. **Blink Logic & Dispatch**  
   - Buffer of recent states smooths flicker.  
   - Detect single and double blinks by timing closed→open transitions.  
   - On double‑blink, issue a user‑defined command.

---

## Prerequisites

- Webcam
- Linux (aint doin windows for now, i mean it should work on windows but xdotool doesnt)
- Qt5  
- OpenCV (can choose to build)
- ONNX Runtime(cmake will take care of it) 
- xdotool (for command dispatch)  
- CMake ≥ 3.17  

---

## Build

```bash
git clone <repo-url>
cd blinkGesture
mkdir build && cd build
cmake .. \
  -DUSE_SYSTEM_OPENCV=ON \
  -DUSE_SYSTEM_ONNX=ON
make -j
```

* To build bundled OpenCV/ONNX Runtime, omit or set `USE_SYSTEM_*` to `OFF`.  
* Ensure `haarcascade_eye.xml` is in the executable's directory or adjust the path at runtime.

---

## Run

```bash
./blinkGesture
```

1. Select your camera.
2. Adjust thresholds, buffer size, blink frames, and command string.  
3. Click **Start**.  
4. Double‑blink to trigger the command.

---

## Pipeline

1. **Training (Python)**

   * Dataset in `dataset/train/{open eyes, close eyes}`.  
   * Grayscale → Resize 64×64 → CLAHE → ToTensor.  
   * Class balance via downsampling closed‑eyes.  
   * Train `EyeCNN` (Conv‑BatchNorm‑ReLU, pooling, FC layers) with weighted cross‑entropy and label smoothing.  
   * Save best model as `eye_cnn.pth`.

2. **Export & Quantize**

   * `export_cnn_to_onnx.py` → `eye_cnn.onnx`.  
   * `quantize_to_int8.py` → `eye_cnn_int8.onnx` (static quantization, QInt8, input patched to INT8 + scale/zero‑point).

3. **Evaluation (Python)**

   * `confusionmatrix.py`: preprocess test images (CLAHE, resize, quantize), run ONNX INT8 model, print classification report and confusion matrix.

4. **Runtime (C++)**

   * Load `eye_cnn_int8.onnx` via ONNX Runtime.  
   * Capture frames, detect & track eyes, extract eye‑strip, apply CLAHE.  
   * Scan patches, classify via INT8 model, vote & threshold.  
   * Buffer temporal states, detect single/double blinks, dispatch command.

---

## Configuration

* **Haar Cascade**: path to cascade XML  
* **Thresholds**: closed/open confidence  
* **Timing**: buffer size, blink‑frames, double‑blink gap  
* **Command**: any `xdotool`‑compatible key or click string

```bash
xdotool key ctrl+alt+t       # example command
```

---
