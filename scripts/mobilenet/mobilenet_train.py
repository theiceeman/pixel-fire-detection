# python3 ./scripts/mobilenet_train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

DATA_DIR = "dataset"
EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001
SAVE_DIR = "runs/classify/mobilenetv4"

os.makedirs(os.path.join(SAVE_DIR, "weights"), exist_ok=True)

# image preprocessing, resizing, converting to tensor
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# load datasets
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# load model
model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True, num_classes=2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# train model
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {running_loss/len(train_loader):.4f} - val_acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model": model.state_dict(), "class_names": train_dataset.classes}, os.path.join(SAVE_DIR, "weights", "best.pt"))

torch.save({"model": model.state_dict(), "class_names": train_dataset.classes}, os.path.join(SAVE_DIR, "weights", "last.pt"))
print(f"Best val_acc: {best_acc:.4f}")