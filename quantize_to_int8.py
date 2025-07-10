import os
import torch
import numpy as np
import cv2
from PIL import Image
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from onnx import helper, TensorProto, numpy_helper
from collections import defaultdict

# CONFIG
MODEL_PATH = "eye_cnn.onnx"
TEMP_QUANT_MODEL = "eye_cnn_quant_tmp.onnx"
FINAL_INT8_MODEL = "eye_cnn_int8.onnx"
CALIBRATION_DATA_PATH = "dataset/train"
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_CALIBRATION_IMAGES = 256

# CLAHE Transform Helper
def apply_clahe(img: Image.Image) -> Image.Image:
    img_np = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    return Image.fromarray(enhanced)

# Transform and Dataset with CLAHE
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(apply_clahe),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(CALIBRATION_DATA_PATH, transform=transform)

# Build Balanced Subset
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(dataset):
    class_indices[label].append(idx)

num_classes = len(class_indices)
num_per_class = min(len(class_indices[0]), len(class_indices[1]), NUM_CALIBRATION_IMAGES // 2)

balanced_indices = (
    class_indices[0][:num_per_class] + class_indices[1][:num_per_class]
)
np.random.shuffle(balanced_indices)

subset = torch.utils.data.Subset(dataset, balanced_indices)
calib_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

# Calibration Reader
class EyeCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            input_name = onnx.load(MODEL_PATH).graph.input[0].name
            self.enum_data = iter([{input_name: batch.numpy()} for batch, _ in self.dataloader])
        return next(self.enum_data, None)

# Step 1: Quantize using QOperator format
print("-- Quantizing weights/activations (QOperator format) --")
quantize_static(
    model_input=MODEL_PATH,
    model_output=TEMP_QUANT_MODEL,
    calibration_data_reader=EyeCalibrationDataReader(calib_loader),
    quant_format=QuantFormat.QOperator,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

# Step 2: Patch to accept INT8 input
print("-- Patching model to accept int8 input --")
model = onnx.load(TEMP_QUANT_MODEL)

# Remove inputs that are also initializers
initializer_names = {init.name for init in model.graph.initializer}
inputs_to_keep = [i for i in model.graph.input if i.name not in initializer_names]
model.graph.ClearField("input")
model.graph.input.extend(inputs_to_keep)

# Get original input
original_input = model.graph.input[0]
input_name = original_input.name

# Handle dynamic or missing shape dims
input_shape = []
for dim in original_input.type.tensor_type.shape.dim:
    if dim.dim_value > 0:
        input_shape.append(dim.dim_value)
    else:
        input_shape.append(1)  # fallback

# Replace input with INT8
model.graph.input.remove(original_input)
int8_input = helper.make_tensor_value_info(input_name, TensorProto.INT8, input_shape)
model.graph.input.insert(0, int8_input)

# Add scale and zero point
scale_name = input_name + "_scale"
zp_name = input_name + "_zero_point"

scale = numpy_helper.from_array(np.array([1.0 / 255.0], dtype=np.float32), name=scale_name)
zero_point = numpy_helper.from_array(np.array([0], dtype=np.int8), name=zp_name)
model.graph.initializer.extend([scale, zero_point])

# Clean duplicate initializers
seen = set()
clean_initializers = []
for init in model.graph.initializer:
    if init.name not in seen:
        clean_initializers.append(init)
        seen.add(init.name)
model.graph.ClearField("initializer")
model.graph.initializer.extend(clean_initializers)

# Remove QuantizeLinear node directly after input
def remove_input_quantize_linear(model, input_name):
    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[0] == input_name:
            print(f"Removing QuantizeLinear node: {node.name or '[unnamed]'}")
            quant_output = node.output[0]
            for n in model.graph.node:
                for i, inp in enumerate(n.input):
                    if inp == quant_output:
                        n.input[i] = input_name
            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        model.graph.node.remove(node)

remove_input_quantize_linear(model, input_name)

# Save final model
onnx.save(model, FINAL_INT8_MODEL)
print(f"INT8 input model saved to: {FINAL_INT8_MODEL}")
