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


# Hyperparameters
hyper_param_epoch = 200
hyper_param_batch = 64
hyper_param_learning_rate = 0.001
weight_decay = 1e-4
patience = 100


# Data augmentation

# Enhanced Data Augmentation
transforms_train = transforms.Compose([
    transforms.Resize((245, 457)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(0, shear=15, scale=(0.9, 1.1)),
    transforms.ToTensor()
])

transforms_test = transforms.Compose([
    transforms.Resize((245, 457)),
    transforms.ToTensor()
])


# Datasets and DataLoaders

train_data_set = CustomImageDataset(data_set_path="E://Project//PTE//trainingSet", transforms=transforms_train)
test_data_set = CustomImageDataset(data_set_path="E://Project//PTE//testSet", transforms=transforms_test)

# Define the validation split ratio
validation_ratio = 0.2

# Calculate the number of samples for training and validation
train_size = int((1 - validation_ratio) * len(train_data_set))
val_size = len(train_data_set) - train_size

# Split the dataset into training and validation
train_subset, val_subset = random_split(train_data_set, [train_size, val_size])

# Create DataLoader for validation set
val_loader = DataLoader(val_subset, batch_size=hyper_param_batch, shuffle=False)
train_loader = DataLoader(train_subset, batch_size=hyper_param_batch, shuffle=True)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=False)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = train_data_set.num_classes
custom_model = CustomConvNet(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Modified training loop
best_accuracy = 0.0
no_improve = 0
train_losses = []
valid_losses = []

for epoch in range(hyper_param_epoch):
    custom_model.train()
    running_loss = 0.0

    # Training phase
    for i_batch, item in enumerate(train_loader):
        images = item['image'].to(device)
        labels = item['label'].to(device)

        outputs = custom_model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(custom_model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    

    # Validation phase
    custom_model.eval()
    test_loss, correct, total = 0.0, 0, 0


    with torch.no_grad():
        for item in val_loader:
            images = item['image'].to(device)
            labels = item['label'].to(device)
            outputs = custom_model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= total
    valid_losses.append(test_loss)
    test_accuracy = 100 * correct / total

    # Update learning rate based on loss
    scheduler.step(test_loss)

    # Save model ONLY if accuracy improves
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        no_improve = 0
        torch.save(custom_model.state_dict(), 'best_model.pth')
        print(f"▲ New best model saved with accuracy: {test_accuracy:.2f}%")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"▼ Early stopping at epoch {epoch+1} (No improvement for {patience} epochs)")
            break

    # Print epoch stats
    print(f'Epoch [{epoch+1}/{hyper_param_epoch}] | '
          f'Train Loss: {train_loss:.4f} | '
          f'Test Loss: {test_loss:.4f} | '
          f'Test Accuracy: {test_accuracy:.2f}% | '
          f'Best Accuracy: {best_accuracy:.2f}% | '
          f'LR: {optimizer.param_groups[0]["lr"]:.1e}')

# Load best model for final evaluation
custom_model.load_state_dict(torch.load('best_model.pth'))
print(f"\nTraining complete. Best accuracy: {best_accuracy:.2f}%")


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
