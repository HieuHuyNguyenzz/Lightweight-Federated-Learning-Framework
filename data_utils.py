import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    return train_dataset, test_dataset

def partition_data(dataset, num_clients):
    """Simple IID partition of the dataset."""
    num_items = int(len(dataset))
    split = np.array_split(np.arange(num_items), num_clients)
    
    return [Subset(dataset, indices.tolist()) for indices in split]
