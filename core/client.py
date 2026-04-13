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
        
        # Dataset parameters
        params = {
            "mnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
            "fmnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
            "emnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
            "cifar10": {"in_channels": 3, "input_size": 32, "num_classes": 10},
        }
        p = params.get(config.dataset_name.lower(), params["mnist"])
        
        # Instantiate model with proper parameters
        self.model = model_class(**p).to(self.device)
        
        # Optimized DataLoader
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

    def train(self, global_weights, epochs=None, lr=None, strategy_type="FedAvg", global_c=None, local_c=None):
        train_epochs = epochs if epochs is not None else self.config.local_epochs
        train_lr = lr if lr is not None else self.config.lr
        
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        global_params = [p.clone().detach() for p in self.model.parameters()]
        optimizer = optim.SGD(self.model.parameters(), lr=train_lr)
        criterion = nn.CrossEntropyLoss()

        # For Scaffold: track original gradients to update control variates
        grad_sums = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        total_steps = 0

        for epoch in range(train_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                
                loss = criterion(output, target)
                
                if strategy_type == "FedProx":
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += ((param - global_param)**2).sum()
                    loss += (self.config.mu / 2) * proximal_term
                
                loss.backward()
                
                # Capture original gradients for Scaffold
                if strategy_type == "Scaffold":
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                grad_sums[name] += param.grad.clone()
                        total_steps += 1
                    
                    # Apply Scaffold correction to the gradient before optimizer step
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            c_i = local_c[name].to(self.device)
                            c_g = global_c[name].to(self.device)
                            param.grad.data.add_(-c_i).add_(c_g)
                
                optimizer.step()
        
        # Compute average gradient for Scaffold
        if strategy_type == "Scaffold":
            avg_grads = {name: g / max(1, total_steps) for name, g in grad_sums.items()}
            return {'weights': self.model.state_dict(), 'grad_avg': avg_grads}
        
        if strategy_type == "FedNova":
            # FedNova needs total local steps for normalization on server side
            total_local_steps = train_epochs * len(self.train_loader)
            return {'weights': self.model.state_dict(), 'local_steps': total_local_steps}
        
        return self.model.state_dict()



