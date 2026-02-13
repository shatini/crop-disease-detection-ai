import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# ============================
# Mount Google Drive
# ============================
from google.colab import drive
drive.mount('/content/drive')

# ============================
# Extract dataset from ZIP
# ============================
ZIP_PATH    = '/content/drive/MyDrive/PlantVillage.zip'
EXTRACT_DIR = '/content/PlantVillage'

if not os.path.exists(EXTRACT_DIR):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    print("Done!")
else:
    print("Dataset already extracted.")

# ============================
# Data paths
# ============================
TRAIN_DIR = "/content/PlantVillage/train"
VAL_DIR   = "/content/PlantVillage/val"

# ============================
# Hyperparameters
# ============================
BATCH_SIZE = 32
LR         = 0.001
NUM_EPOCHS = 15

# ============================
# Image transforms
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================
# Datasets and loaders
# ============================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================
# Device setup
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Classes: {len(train_dataset.classes)}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ============================
# Model (ResNet18)
# ============================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)

# ============================
# Loss function and optimizer
# ============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================
# Training function
# ============================
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

# ============================
# Validation function
# ============================
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc  = correct / total
    return val_loss, val_acc

# ============================
# Training loop — 15 epochs
# ============================
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ============================
# Save model to Google Drive
# ============================
SAVE_PATH = '/content/drive/MyDrive/agro_ai_resnet18.pth'
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved: {SAVE_PATH}")
