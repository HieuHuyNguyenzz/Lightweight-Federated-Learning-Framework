from collections import OrderedDict
import torch

class FLServer:
    """Generic FL Server that manages the global model and aggregation strategy."""
    
    def __init__(self, model_class, strategy, config):
        self.config = config
        self.device = config.device
        self.strategy = strategy
        
        # Model parameters based on dataset
        params = {
            "mnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
            "fmnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
            "emnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
            "cifar10": {"in_channels": 3, "input_size": 32, "num_classes": 10},
        }
        p = params.get(config.dataset_name.lower(), params["mnist"])
        
        self.global_model = model_class(**p).to(self.device)
        
        # Strategy-specific states are managed within the strategy object
        # or initialized here via the strategy's interface
        self.strategy.init_server_state(self)
        
        # We keep a few generic placeholders for strategies that need them
        self.global_c = None
        self.local_cs = {} 
        self.alphas = {} 
        self.m = None
        self.v = None
        self.t = 0

    def get_global_weights(self):
        return self.global_model.state_dict()

    def _compute_norm(self, weights1, weights2):
        """Computes the L2 norm of the difference between two state_dicts."""
        total_norm = 0.0
        for name, param in weights1.items():
            if param.dtype != torch.long:
                diff = param.float() - weights2[name].float()
                total_norm += torch.norm(diff).item()**2
        return total_norm**0.5

    def aggregate(self, client_updates):
        """
        Aggregates weights using the injected strategy.
        Handles simple weights, Scaffold, FedNova, FedDyn, and FedAdam updates.
        """
        if not client_updates:
            return None
        
        global_weights = self.strategy.aggregate(self, client_updates)
        
        # Differential Privacy: Noise Addition
        if self.config.dp_enabled:
            with torch.no_grad():
                num_clients = len(client_updates)
                # Noise scale: sigma * clip_norm / sqrt(num_clients)
                # This ensures the average has the correct DP property
                noise_std = (self.config.dp_sigma * self.config.dp_clip_norm) / (num_clients**0.5)
                
                for name, param in self.global_model.named_parameters():
                    # Add Gaussian noise to the weights
                    noise = torch.randn_like(param) * noise_std
                    param.data.add_(noise)
                
                # Update global_weights to reflect the noise addition
                global_weights = self.get_global_weights()
                
        return global_weights

    def evaluate(self, test_loader):
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return 100. * correct / total

