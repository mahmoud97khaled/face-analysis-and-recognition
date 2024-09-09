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
import numpy as np


def calculate_loss(outputs, labels, criterion):
    return criterion(outputs, labels).item()

def calculate_accuracy(preds, labels):
    preds_binary = np.array(preds) > 0.5
    labels_binary = np.array(labels)
    return accuracy_score(labels_binary, preds_binary)