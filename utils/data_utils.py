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
        # Dirichlet distribution partitioning
        # 1. Group indices by label
        label_indices = {label: np.where(targets == label)[0] for label in np.unique(targets)}
        
        # 2. For each label, sample from Dirichlet distribution
        # p is the distribution of a label across clients
        client_indices = [[] for _ in range(num_clients)]
        
        for label, indices in label_indices.items():
            # Sample distribution for this label across clients
            # np.random.dirichlet returns a vector that sums to 1
            p = np.random.dirichlet([alpha] * num_clients)
            
            # Proportion of total items for this label
            proportions = (p * len(indices)).astype(int)
            
            # Shuffle indices to ensure randomness
            np.random.shuffle(indices)
            
            # Distribute indices to clients
            start = 0
            for i in range(num_clients):
                end = start + proportions[i]
                client_indices[i].extend(indices[start:end])
                start = end
                
        return [Subset(dataset, idx) for idx in client_indices]
    
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
