import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataset(dataset_name):
    """
    Dataset factory to load various datasets with appropriate transforms.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test = datasets.MNIST('./data', train=False, transform=transform)
        
    elif dataset_name == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test = datasets.FashionMNIST('./data', train=False, transform=transform)
        
    elif dataset_name == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Using 'balanced' split for EMNIST
        train = datasets.EMNIST('./data', split='balanced', train=True, download=True, transform=transform)
        test = datasets.EMNIST('./data', split='balanced', train=False, transform=transform)
        
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.2010, 0.2023))
        ])
        train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test = datasets.CIFAR10('./data', train=False, transform=transform)
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from: mnist, fmnist, emnist, cifar10")
        
    return train, test

def partition_data(dataset, num_clients, partition_type="iid", alpha=0.5):
    """
    Partition dataset into subsets for clients.
    Supports 'iid', 'non-iid' (sort-by-label), and 'dirichlet'.
    """
    num_items = int(len(dataset))
    targets = np.array(dataset.targets)
    
    if partition_type == "iid":
        split = np.array_split(np.arange(num_items), num_clients)
        return [Subset(dataset, indices.tolist()) for indices in split]
    
    elif partition_type == "non-iid":
        indices = np.argsort(targets)
        split = np.array_split(indices, num_clients)
        return [Subset(dataset, idx.tolist()) for idx in split]
    
    elif partition_type == "dirichlet":
        label_indices = {label: np.where(targets == label)[0] for label in np.unique(targets)}
        client_indices = [[] for _ in range(num_clients)]
        
        for label, indices in label_indices.items():
            p = np.random.dirichlet([alpha] * num_clients)
            proportions = (p * len(indices)).astype(int)
            np.random.shuffle(indices)
            
            start = 0
            for i in range(num_clients):
                end = start + proportions[i]
                client_indices[i].extend(indices[start:end])
                start = end
                
        return [Subset(dataset, idx) for idx in client_indices]
    
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
