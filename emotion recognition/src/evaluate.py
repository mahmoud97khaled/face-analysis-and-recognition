from PIL import Image
import torch
from torchvision import transforms


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB') 
    image = transform(image)
    image = image.unsqueeze(0)  
    return image.to(device)

def predict_emotion(image_path, model_f, class_names):
    
    image = load_image(image_path)
    

    model_f.eval()
    
    with torch.no_grad():
        # Make predictions
        outputs = model_f(image)
        _, predicted = torch.max(outputs, 1)
    

    predicted_class = class_names[predicted.item()]
    return predicted_class

model_f.load_state_dict(torch.load('res34_res34.pth'))  # Load the best model
class_names = train_loader.dataset.classes

# Path to the image to predict
image_path = '.jpg'

# Predict the emotion
predicted_emotion = predict_emotion(image_path, model_f, class_names)
print(f'The predicted emotion is: {predicted_emotion}')
