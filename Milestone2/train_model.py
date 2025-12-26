import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB safe
])

# ---- SAFE ImageFolder loading ----
train_data = ImageFolder(TRAIN_DIR, transform=transform, allow_empty=True)
test_data = ImageFolder(TEST_DIR, transform=transform, allow_empty=True)

# Remove empty classes (important)
valid_classes = []
for i, c in enumerate(train_data.classes):
    if len(os.listdir(os.path.join(TRAIN_DIR, c))) > 0:
        valid_classes.append(c)

train_data.classes = valid_classes
test_data.classes = valid_classes

print("Classes:", train_data.classes)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

num_classes = len(train_data.classes)

# Load EfficientNet
model = torchvision.models.efficientnet_b0(weights="DEFAULT")

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# ---------------- TRAINING ----------------
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Loss: {running_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "pcb_defect_model.pth"))
print("\n🎯 Model saved to models/pcb_defect_model.pth")
