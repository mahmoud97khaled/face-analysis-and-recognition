from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import os


class GenderClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    def preprocess_image(self, image):
        # If the input is a NumPy array (OpenCV image), it will already be in BGR format
        if isinstance(image, np.ndarray):
            # Convert from BGR (OpenCV format) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # OpenCV image (NumPy array) preprocessing without converting to PIL
        # Convert NumPy array to PyTorch tensor (use the same transformations as training)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert NumPy array (HWC) to PyTorch tensor (CHW)
            transforms.Resize((80, 80)),  # Resize to the expected input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        # Apply the transformations
        image = transform(image)

        # Add batch dimension (for single image prediction)
        image = image.unsqueeze(0)
        return image

    def predict(self, image):
        image = self.preprocess_image(image)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)

        probability = torch.sigmoid(output).item()
        return probability

        


if __name__ == "__main__": 
    image_path = "/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/Screen Shot 2024-09-10 at 3.26.01 AM.png"
    image = cv2.imread(image_path)
    
    classifier = GenderClassifier(model_path='/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/gender/models/gender_classifier.pth')
    
    prediction = classifier.predict(image)
    if prediction > 0.5:
        print(f'Prediction: Female ({prediction:.2f})')
                
    else:
        print(f'Prediction: Male ({1 - prediction:.2f})')
    