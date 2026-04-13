import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class FLClient:
    """Generic FL Client that can work with any model and dataset."""
    
    def __init__(self, client_id, model_class, train_dataset, config):
        self.client_id = client_id
        self.config = config
        self.device = config.device
        self.model = model_class().to(self.device)
        
        # Optimized DataLoader
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0, # Keep 0 to avoid daemon process errors with mp.Pool
            pin_memory=True if self.device.type == 'cuda' else False
        )

    def train(self, global_weights, epochs=None, lr=None, strategy_type="FedAvg"):
        # Use provided values or fallback to config
        train_epochs = epochs if epochs is not None else self.config.local_epochs
        train_lr = lr if lr is not None else self.config.lr
        
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        # Store global weights for FedProx proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]
        
        optimizer = optim.SGD(self.model.parameters(), lr=train_lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(train_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                
                # Base loss
                loss = criterion(output, target)
                
                # Add proximal term if using FedProx
                if strategy_type == "FedProx":
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += ((param - global_param)**2).sum()
                    loss += (self.config.mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
        
        return self.model.state_dict()

