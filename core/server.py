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
        Handles simple weights, Scaffold updates, and FedNova updates.
        """
        if not client_updates:
            return None

        # Detect update type based on the first client's update
        first_update = client_updates[0]
        
        # Case 1: Scaffold (weights + grad_avg)
        is_scaffold = isinstance(first_update, dict) and 'grad_avg' in first_update
        # Case 2: FedNova (weights + local_steps)
        is_fednova = isinstance(first_update, dict) and 'local_steps' in first_update
        
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
            
        else:
            # Standard FedAvg / FedProx
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
