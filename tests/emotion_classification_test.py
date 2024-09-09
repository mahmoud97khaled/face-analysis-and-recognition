import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

device = "cpu"


def load_model(model_path, device):
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


def preprocess_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    image = image.astype(np.float32) / 255.0

    transform_test = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

    img_tensor = transform_test(image).unsqueeze(0)
    return img_tensor


def predict(img_tensor, device):
    model = load_model(model_path, device)
    img_tensor = preprocess_image(photo)

    
    img_tensor = img_tensor.to(device)


    with torch.no_grad():
        outputs = model(img_tensor)

    probabilities = torch.nn.functional.softmax(outputs, dim=1)  
    confidence, predicted_idx = torch.max(probabilities, 1)
    predicted_idx = predicted_idx.item()
    confidence = confidence.item()


    emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[predicted_idx]

    return predicted_emotion, confidence


image_path = 'me.jpg'  
model_path = 'res34_res34.pth'  


photo = cv2.imread(image_path)


predicted_emotion, confidence = predict(photo, device)
print(f"Predicted Emotion: {predicted_emotion}, Confidence: {confidence:.2f}")
