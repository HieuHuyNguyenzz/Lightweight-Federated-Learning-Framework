import torch
from collections import OrderedDict
from core.strategies.base import BaseStrategy

class FedAdagradStrategy(BaseStrategy):
    """
    FedAdagrad implementation.
    Adaptive learning rate based on the sum of squared pseudo-gradients.
    """
    def init_server_state(self, server):
        # Initialize the accumulated squared gradients (G)
        server.G = OrderedDict({
            name: torch.zeros_like(param).to(server.device) 
            for name, param in server.global_model.named_parameters()
        })

    def aggregate(self, server, client_updates):
        if not client_updates:
            return None
            
        # 1. Compute the average weights (FedAvg part)
        global_dict = OrderedDict()
        num_clients = len(client_updates)
        for key in client_updates[0].keys():
            stacked = torch.stack([upd[key].float() for upd in client_updates], dim=0)
            global_dict[key] = torch.mean(stacked, dim=0).to(server.device)
        
        # 2. Apply Adagrad update rule on server side
        with torch.no_grad():
            for name, param in server.global_model.named_parameters():
                # Pseudo-gradient delta_t = w_t - avg_weights
                delta_t = param.data - global_dict[name].float()
                
                # Update accumulated squared gradients: G_t = G_{t-1} + delta_t^2
                server.G[name].add_(delta_t ** 2)
                
                # Update global weights: w_{t+1} = w_t - lr / (sqrt(G_t) + eps) * delta_t
                param.data.sub_(delta_t / (torch.sqrt(server.G[name]) + server.config.epsilon), alpha=server.config.fedadagrad_lr)
                
        return server.get_global_weights()
