import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "models/pcb_defect_model.pth"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

test_data = ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
class_names = test_data.classes

# Load model
model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

# Metrics
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

print("\nClassification Report:\n", report)

# Save report
with open(os.path.join(RESULT_DIR, "report.txt"), "w") as f:
    f.write(report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

print("🎯 Confusion matrix & report saved in results/")
