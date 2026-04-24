import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from collections import OrderedDict

class FLClient:
    """Generic FL Client that can work with any model and dataset."""
    
    def __init__(self, client_id, model_class, train_dataset, config, strategy):
        self.client_id = client_id
        self.config = config
        self.device = config.device
        self.strategy = strategy
        
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
            pin_memory=True if self.device.type in ['cuda', 'mps'] else False
        )
        
        # Initialize strategy-specific states
        self.strategy.init_client_state(self)

    def _compute_contrastive_loss(self, z_curr, z_global, z_prev):
        """
        Computes the contrastive loss for MOON.
        Loss = 1/2 * (1 - cos(z_curr, z_global)) + 1/2 * (1 - cos(z_curr, z_prev))
        """
        cos_global = F.cosine_similarity(z_curr, z_global, dim=1)
        cos_prev = F.cosine_similarity(z_curr, z_prev, dim=1)
        
        loss = 0.5 * (1 - cos_global) + 0.5 * (1 - cos_prev)
        return loss.mean()

    def _apply_dp_clipping(self, final_weights, initial_weights):
        """
        Applies L2 norm clipping to the weight updates for Differential Privacy.
        """
        with torch.no_grad():
            delta_w = OrderedDict()
            total_norm = 0.0
            
            # Calculate Delta and total L2 norm
            for name in final_weights.keys():
                # Ensure initial weights are on the correct device
                w_init = initial_weights[name].to(self.device)
                diff = final_weights[name].float() - w_init.float()
                delta_w[name] = diff
                total_norm += torch.norm(diff).item()**2
            
            total_norm = total_norm**0.5
            
            # Clip if norm exceeds threshold
            clip_coeff = min(1.0, self.config.dp_clip_norm / (total_norm + 1e-6))
            
            clipped_weights = OrderedDict()
            for name in final_weights.keys():
                # Ensure initial weights are on the correct device
                w_init = initial_weights[name].to(self.device)
                clipped_weights[name] = w_init + clip_coeff * delta_w[name]
                
            return clipped_weights

    def train(self, global_weights, epochs=None, lr=None, global_c=None, local_c=None, alpha=1.0):
        train_epochs = epochs if epochs is not None else self.config.local_epochs
        train_lr = lr if lr is not None else self.config.lr
        
        # Store current weights for MOON and DP clipping
        prev_weights = copy.deepcopy(self.model.state_dict())
        
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        # Store global params for proximal terms (FedProx, FedDyn)
        self.global_params = [p.clone().detach() for p in self.model.parameters()]
        optimizer = optim.SGD(self.model.parameters(), lr=train_lr)
        criterion = nn.CrossEntropyLoss()

        # Strategy-specific setup (e.g., MOON frozen models)
        if self.strategy.is_moon():
            self.strategy.setup_moon_models(self, prev_weights)

        # For Scaffold: track original gradients to update control variates
        grad_sums = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        total_steps = 0

        for epoch in range(train_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                # Standard Forward Pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Strategy-specific loss modification
                loss = self.strategy.apply_local_loss(self, loss, data, target, alpha=alpha)
                
                loss.backward()
                
                # Strategy-specific gradient modification (e.g., SCAFFOLD)
                self.strategy.modify_gradients(self, self.model)
                
                # Capture original gradients for Scaffold
                if self.strategy.is_scaffold():
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                grad_sums[name] += param.grad.clone()
                        total_steps += 1
                
                optimizer.step()
        
        # Strategy-specific return values
        if self.strategy.is_scaffold():
            avg_grads = {name: g / max(1, total_steps) for name, g in grad_sums.items()}
            return {'weights': self.model.state_dict(), 'grad_avg': avg_grads}
        
        if self.strategy.is_fednova():
            total_local_steps = train_epochs * len(self.train_loader)
            return {'weights': self.model.state_dict(), 'local_steps': total_local_steps}
        
        final_weights = self.model.state_dict()
        if self.config.dp_enabled:
            final_weights = self._apply_dp_clipping(final_weights, global_weights)
            
        return final_weights
