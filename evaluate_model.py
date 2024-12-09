import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
from tqdm import tqdm


device = torch.device('cpu')
val_dir = './data/val'
model_path = "fire_classification_model.pth"

# Data transformation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_data = ImageFolder(root=val_dir, transform=val_transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Load the model
model = resnet18(pretrained=False)  
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 1)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


print("Evaluating the model...")
val_loss = 0.0
all_preds = []
all_labels = []

criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images)
        val_loss += criterion(outputs.squeeze(), labels).item()

        preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
