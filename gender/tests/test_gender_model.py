
print('hegllo')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score
# import cv2
import os
from tqdm import tqdm
import numpy as np



def load_model(model_path, device):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Adjust to match binary classification output
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image):
    
    # Convert BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the transformations
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])
    
    # Apply transformations
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(image):
    device = 'cpu'
    model = load_model('models/gender_classifier.pth', device)
    # Preprocess the image
    image = preprocess_image(image)
    image = image.to(device)  # Move to device
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
    
    # Apply sigmoid to get probability
    probability = torch.sigmoid(output).item()
    
    # Print result
    if probability > 0.5:
        print(f'Prediction: Female, Probability: {probability:.4f}')
    else:
        print(f'Prediction: Male, Probability: {1 - probability:.4f}')


image_path = "/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/raw/archive (4)/Training/female/131422.jpg.jpg"
image = cv2.imread(image_path)
predict(image)
print('hegllo')
