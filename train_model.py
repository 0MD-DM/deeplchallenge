import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
from tqdm import tqdm
import numpy as np

device = torch.device('cpu')
train_dir = './data/train'
val_dir = './data/val'


# Data transformations, augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), 
    transforms.RandomRotation(30),  
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Dataset
train_data = ImageFolder(root=train_dir, transform=train_transform)
val_data = ImageFolder(root=val_dir, transform=val_transform)


train_subset_indices = np.random.choice(len(train_data), int(0.3 * len(train_data)), replace=False)  
train_data = Subset(train_data, train_subset_indices)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)



# Model
model = resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 1)
)
model = model.to(device)


# Loss function, optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 


# Training loop
epochs = 10 
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device).float()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")


    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            val_loss += criterion(outputs.squeeze(), labels).item()
            preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
            correct += (preds.cpu() == labels.cpu().int()).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")



model_path = "fire_classification_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}.")
