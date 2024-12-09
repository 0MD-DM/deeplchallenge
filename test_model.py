import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import os
import pandas as pd


device = torch.device('cpu')


test_dir = './data/test'
model_path = "fire_classification_model.pth"
output_csv = "test_predictions.csv"


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


model = resnet18(pretrained=False) 
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),  
    nn.Linear(num_features, 1)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


print("Predicting test images...")
results = []

for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg')):  
        filepath = os.path.join(test_dir, filename)
        image = Image.open(filepath).convert("RGB") 
        image = test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prediction = (torch.sigmoid(output).item() > 0.5) 
            results.append({"id": filename, "class": int(prediction)})


df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}.")
