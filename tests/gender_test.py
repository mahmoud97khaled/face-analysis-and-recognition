from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

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
        if image.mode != "RGB":
            image = image.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        image = image.unsqueeze(0)
        return image

    def predict(self, image):
        image = self.preprocess_image(image)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)

        probability = torch.sigmoid(output).item()

        if probability > 0.5:
            print(f'Prediction: Female, Probability: {probability:.4f}')
        else:
            print(f'Prediction: Male, Probability: {1 - probability:.4f}')


if __name__ == "__main__":
    image_path = "/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/raw/archive (4)/Training/male/090548.jpg.jpg"
    image = Image.open(image_path)
    
    classifier = GenderClassifier(model_path='models/gender_classifier.pth')
    
    classifier.predict(image)
    
    image.show()