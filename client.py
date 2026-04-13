import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, model_class, train_dataset, device):
        self.client_id = client_id
        self.model = model_class().to(device)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True if device.type == 'cuda' else False
        )
        self.device = device

    def train(self, global_weights, epochs=1, lr=0.01):
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Return local weights
        return self.model.state_dict()
