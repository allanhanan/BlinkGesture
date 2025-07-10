import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Set seeds for reproducibility
random.seed(420)
torch.manual_seed(420)
np.random.seed(420)

# CLAHE preprocessing
def apply_clahe(img: Image.Image) -> Image.Image:
    img_np = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    return Image.fromarray(enhanced)

# Custom Dataset Wrapper
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

# CNN Model Definition
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

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
common_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(apply_clahe),
    transforms.ToTensor()
])

# Load dataset
train_dir = "dataset/train"
full_dataset = datasets.ImageFolder(train_dir, transform=None)

# Downsample class 0 (closed eyes)
indices_by_class = {0: [], 1: []}
for idx, (_, label) in enumerate(full_dataset):
    indices_by_class[label].append(idx)

closed_keep_ratio = 0.8
closed_sampled = random.sample(indices_by_class[0], int(len(indices_by_class[0]) * closed_keep_ratio))
open_samples = indices_by_class[1]

final_indices = closed_sampled + open_samples
random.shuffle(final_indices)

# Train/Val split
val_size = int(0.1 * len(final_indices))
train_indices = final_indices[:-val_size]
val_indices = final_indices[-val_size:]

train_dataset = CustomDataset(full_dataset, train_indices, transform=common_transform)
val_dataset = CustomDataset(full_dataset, val_indices, transform=common_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model setup
model = EyeCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 1.8]).to(DEVICE),
    label_smoothing=0.15
)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training loop
best_val_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    model.eval()
    total_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val

    print(f"Epoch {epoch:03d} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "eye_cnn.pth")
        print("Model improved. Saved checkpoint.")

print("Training complete.")
print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
