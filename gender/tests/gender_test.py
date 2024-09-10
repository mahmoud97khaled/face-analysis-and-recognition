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
        if image.mode != "RGB":
            image = image.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((80, 80)),
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
        return probability

        


if __name__ == "__main__":
    # def get_limited_image_paths_and_labels(folder_path, label, limit):
    #     image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)[:limit]]
    #     labels = [label] * len(image_paths)
    #     return image_paths
    # # /Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/gender/raw/archive (4)/Validation
    # image_folder_path = '/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/gender/raw/archive (4)/Validation'
    # female_folder_path = os.path.join(image_folder_path, 'female')
    # female_image_paths = get_limited_image_paths_and_labels(female_folder_path, label=1, limit=2000)
    
    # # Initialize the classifier
    # classifier = GenderClassifier(model_path='/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/gender/models/gender_classifier.pth')
    
    # count = 0
    # male_count = 0
    # female_count = 0
    # images = get_images()

    # for image_path in images:
    #     # count += 1
    #     # print(count)

    #     # # Open image
    #     # image = Image.open(image_path)

    #     # # Predict gender using the classifier
    #     # prediction = classifier.predict(image)
    #     prediction = classifier.predict(image_path)


    #     if prediction > 0.5:
    #             print(f'Prediction: Female ({prediction:.2f})')
    #             female_count += 1
    #     else:
    #         print(f'Prediction: Male ({1 - prediction:.2f})')
    #         male_count += 1
    #     # # Optionally show the image (remove or comment this if you don't want to open each image)
    #     # image.show()
    # print(male_count,female_count)
    
    image_path = "/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/Screen Shot 2024-09-10 at 3.19.17 AM.png"
    image = Image.open(image_path)
    
    classifier = GenderClassifier(model_path='/Users/Tata/face-analysis-and-recognition/face-analysis-and-recognition/gender/models/gender_classifier.pth')
    
    prediction = classifier.predict(image)
    if prediction > 0.5:
        print(f'Prediction: Female ({prediction:.2f})')
                
    else:
        print(f'Prediction: Male ({1 - prediction:.2f})')
    
    image.show()