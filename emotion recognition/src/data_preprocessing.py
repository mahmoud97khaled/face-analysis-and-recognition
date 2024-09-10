import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
from torchvision.transforms import v2
# Define data transformations
transform_train = v2.Compose([
    v2.Grayscale(num_output_channels=3),  # Convert grayscale images to 3-channel images
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=(0,180)),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to 3-channel images
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset
data_path = ''  # Replace with the actual path
train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform_train)
test_dataset = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform_test)


# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'Training samples: {len(train_loader.dataset)}')
print(f'Test samples: {len(test_loader.dataset)}')
