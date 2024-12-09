import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm


device = torch.device("cpu")
model_path = "fire_classification_model.pth"
test_dir = "./data/test"  


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("Loading the model...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded from {model_path}.")



def load_test_images(folder, transform):
    test_images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')): 
            filepath = os.path.join(folder, filename)
            img = Image.open(filepath).convert("RGB") 
            img = transform(img)  
            test_images.append((img, filename))
    return test_images


print("Loading test images...")
test_images = load_test_images(test_dir, transform)


print("Processing test images...")
results = []
with torch.no_grad():
    for img, filename in tqdm(test_images, desc="Predicting"):
        img = img.unsqueeze(0).to(device) 
        output = model(img)
        prediction = (torch.sigmoid(output).item() > 0.5)  
        results.append({"id": filename, "class": int(prediction)})


output_csv = "classification_results.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Test predictions saved to '{output_csv}'.")
