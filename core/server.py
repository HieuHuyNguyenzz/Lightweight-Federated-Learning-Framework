import torch
from collections import OrderedDict

class FLServer:
    """Generic FL Server that manages the global model and aggregation strategy."""
    
    def __init__(self, model_class, strategy, config):
        self.config = config
        self.device = config.device
        self.strategy = strategy
        self.global_model = model_class().to(self.device)
        
        # Scaffold states: global control variate and local control variates
        self.global_c = None
        self.local_cs = {} # {client_id: OrderedDict of tensors}

    def _init_scaffold_states(self, num_clients):
        """Initialize control variates for Scaffold."""
        # Global control variate
        self.global_c = OrderedDict({
            name: torch.zeros_like(param).to(self.device) 
            for name, param in self.global_model.named_parameters()
        })
        # Local control variates for each client
        for i in range(num_clients):
            self.local_cs[i] = OrderedDict({
                name: torch.zeros_like(param).to(self.device) 
                for name, param in self.global_model.named_parameters()
            })

    def get_global_weights(self):
        return self.global_model.state_dict()

    def aggregate(self, client_updates):
        """
        Aggregates weights using the injected strategy.
        Handles both simple weights and Scaffold updates (weights + grad_avg).
        """
        # Check if we are using Scaffold (updates will be list of dicts)
        is_scaffold = isinstance(client_updates[0], dict)
        
        if is_scaffold:
            # Scaffold returns (global_weights, global_c_update)
            global_weights, global_c_update = self.strategy.aggregate(client_updates)
            
            # Update global model
            self.global_model.load_state_dict(global_weights)
            
            # Update global control variate: c_g = c_g + global_c_update
            with torch.no_grad():
                for name, update in global_c_update.items():
                    self.global_c[name].add_(update)
            
            # Update local control variates: c_i = c_i - grad_avg
            # (This part is done during the update of global_c in original Scaffold paper,
            # but usually handled by subtracting the current grad_avg from local c_i)
            for i, update in enumerate(client_updates):
                grad_avg = update['grad_avg']
                for name, g_val in grad_avg.items():
                    self.local_cs[i][name].sub_(g_val)
            
            return global_weights
        else:
            # Standard FedAvg / FedProx
            global_weights = self.strategy.aggregate(client_updates)
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
