import torch
import torch.nn as nn

# CNN (same architecture used in training)
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# Settings
ONNX_FILE = "eye_cnn.onnx"
MODEL_FILE = "eye_cnn.pth"
IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EyeCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

# Dummy input (batch size 1, grayscale, 64x64)
dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_FILE,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11,
    export_params=True
)

print(f"Model exported to {ONNX_FILE}")
