import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score
import cv2
import os
from tqdm import tqdm





# Image preprocessing with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

def get_limited_image_paths_and_labels(folder_path, label, limit):
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)[:limit]]
    labels = [label] * len(image_paths)
    return image_paths, labels


def get_data(image_folder_path):
    male_folder_path = os.path.join(image_folder_path, 'male')
    female_folder_path = os.path.join(image_folder_path, 'female')

    # Take 1000 male and 100 female images for training
    male_image_paths, male_labels = get_limited_image_paths_and_labels(male_folder_path, label=0, limit=2000)
    female_image_paths, female_labels = get_limited_image_paths_and_labels(female_folder_path, label=1, limit=2000)

    image_paths = male_image_paths + female_image_paths
    labels = male_labels + female_labels
    return image_paths,labels





class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.image_paths = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
      img_path = self.image_paths[idx]
      image = cv2.imread(img_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if self.transform:
          image = self.transform(image)  # Convert to tensor
      label = torch.tensor(self.labels[idx], dtype=torch.float32)


      return image, label

def dataload_create(images1,images2):
    train_image_paths,train_labels = get_data(images1)
    val_image_paths,val_labels = get_data(images2)
    train_dataset = ImageDataset(train_image_paths, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=8)

    val_dataset = ImageDataset(val_image_paths, val_labels, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=8)