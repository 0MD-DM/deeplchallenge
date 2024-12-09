import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = torch.device("cpu")


model_path = "fire_classification_model.pth"
val_dir = "./data/val"  


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("Loading validation dataset...")
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


print("Loading the model...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded from {model_path}.")


print("Evaluating the model...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        
        
        outputs = model(images)
        preds = (torch.sigmoid(outputs).squeeze() > 0.5).int()
        
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())



accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
