import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model
model = models.resnet18()

# Modify the final layer to match the number of classes (e.g., 1 for binary classification)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)  # Adjust to match your output classes
# Load the trained weights
model = model.to(device)
def load_model(weight_path,model):
    return model.load_state_dict(torch.load(weight_path))


# Set the model to evaluation mode
model.eval()
