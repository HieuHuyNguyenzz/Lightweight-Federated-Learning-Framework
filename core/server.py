from collections import OrderedDict
import torch
from core.strategy import FedDynStrategy

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
        
        # Scaffold states
        self.global_c = None
        self.local_cs = {} 
        
        # FedDyn states: alpha coefficients for each client
        self.alphas = {} # {client_id: float}

    def _init_scaffold_states(self, num_clients):
        """Initialize control variates for Scaffold."""
        self.global_c = OrderedDict({
            name: torch.zeros_like(param).to(self.device) 
            for name, param in self.global_model.named_parameters()
        })
        for i in range(num_clients):
            self.local_cs[i] = OrderedDict({
                name: torch.zeros_like(param).to(self.device) 
                for name, param in self.global_model.named_parameters()
            })

    def _init_feddyn_states(self, num_clients):
        """Initialize alpha coefficients for FedDyn."""
        for i in range(num_clients):
            self.alphas[i] = 1.0

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
        Handles simple weights, Scaffold, FedNova, and FedDyn updates.
        """
        if not client_updates:
            return None

        first_update = client_updates[0]
        is_scaffold = isinstance(first_update, dict) and 'grad_avg' in first_update
        is_fednova = isinstance(first_update, dict) and 'local_steps' in first_update
        
        # For FedDyn, check if the strategy is FedDynStrategy
        is_feddyn = isinstance(self.strategy, FedDynStrategy)
        
        if is_scaffold:
            global_weights, global_c_update = self.strategy.aggregate(client_updates, self.get_global_weights())
            self.global_model.load_state_dict(global_weights)
            with torch.no_grad():
                for name, update in global_c_update.items():
                    self.global_c[name].add_(update)
            for i, update in enumerate(client_updates):
                grad_avg = update['grad_avg']
                for name, g_val in grad_avg.items():
                    self.local_cs[i][name].sub_(g_val)
            return global_weights
            
        elif is_fednova:
            global_weights = self.strategy.aggregate(client_updates, self.get_global_weights())
            self.global_model.load_state_dict(global_weights)
            return global_weights
            
        elif is_feddyn:
            # FedDyn aggregation: Weighted average by alphas
            old_global_weights = self.get_global_weights()
            global_weights = self.strategy.aggregate(client_updates, global_weights=old_global_weights, alphas=self.alphas)
            self.global_model.load_state_dict(global_weights)
            
            # Update alphas for the next round
            # alpha_{i, next} = alpha_{i, curr} * (||w_{next} - w_{curr}|| / ||w_{i, next} - w_{curr}||)
            global_update_norm = self._compute_norm(global_weights, old_global_weights)
            
            for i, update in enumerate(client_updates):
                local_update_norm = self._compute_norm(update, old_global_weights)
                if local_update_norm > 0:
                    self.alphas[i] *= (global_update_norm / local_update_norm)
                
                # Clip alpha to prevent instability
                self.alphas[i] = max(0.01, min(self.alphas[i], 10.0))
            
            return global_weights
            
        else:
            global_weights = self.strategy.aggregate(client_updates, self.get_global_weights())
            self.global_model.load_state_dict(global_weights)
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
