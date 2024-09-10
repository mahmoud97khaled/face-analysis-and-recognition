import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image

class EmotionDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = self.load_model(model_path, self.device)

    def load_model(self, model_path, device):
        # Load a ResNet34 model
        model = models.resnet34(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  
        )
        

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()  
        return model

    # def preprocess_image(self, image):
    #     if isinstance(image, np.ndarray):
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     elif not isinstance(image, Image.Image):
    #         raise ValueError("Input must be a PIL Image or a NumPy array.")
        
    #     transform_test = transforms.Compose([
    #         transforms.Grayscale(num_output_channels=3),
    #         transforms.Resize((224, 224)),                
    #         transforms.ToTensor(),                      
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    #     ])

    #     img_tensor = transform_test(image).unsqueeze(0)
    #     return img_tensor
    def preprocess_image(self, image):
        # Convert from BGR to RGB using OpenCV
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to grayscale using OpenCV (only 1 channel)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Since ResNet expects 3 channels, stack the grayscale image to have 3 channels
        image_gray_3channel = np.stack((image_gray,) * 3, axis=-1)

        # Now use torchvision transforms for the rest of the preprocessing
        transform_test = transforms.Compose([
            transforms.ToTensor(),  # Convert to a PyTorch tensor
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
        ])

        # Apply the transformations
        img_tensor = transform_test(image_gray_3channel).unsqueeze(0)  # Add batch dimension
        return img_tensor

    def predict(self, image):
        # Preprocess the image
        img_tensor = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

        # Emotion labels (adjust based on your classes)
        emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = emotion_labels[predicted_idx]

        return predicted_emotion, confidence
