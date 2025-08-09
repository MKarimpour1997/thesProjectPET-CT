import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torch.nn as nn

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms

# --- Settings ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 3
class_names = ["normal", "NSCLC", "SCC"]

# --- Load Model ---
model = CustomConvNet(num_classes=num_classes)
model.load_state_dict(torch.load('best_best_model.pth', map_location=device))
model.to(device)
model.eval()

# --- Gather Predictions and Labels ---
all_labels = []
all_preds = []
all_probs = []
all_filename = []
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        labels = batch['label'].cpu().numpy()
        filename = batch['filename']
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_filename.extend(filename)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Pretty confusion matrix
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --- Per-Class Accuracy ---
class_acc = cm.diagonal() / cm.sum(axis=1)
print("\nPer-Class Accuracy:")
for i, acc in enumerate(class_acc):
    print(f"{class_names[i]}: {acc:.2%}")

# --- Classification Report ---
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\nClassification Report:")
print(report)

# --- ROC Curves and AUC ---
labels_binarized = label_binarize(all_labels, classes=np.arange(num_classes))

plt.figure(figsize=(8, 7))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc="lower right")
plt.show()
