import torch
from collections import OrderedDict
from core.strategies.base import BaseStrategy

class FedAvgStrategy(BaseStrategy):
    """
    Federated Averaging (FedAvg) implementation.
    Simple weighted average of client weights.
    """
    def aggregate(self, server, client_updates):
        if not client_updates:
            return None
            
        global_dict = OrderedDict()
        num_clients = len(client_updates)
        
        # Standard FedAvg: simple mean of weights
        for key in client_updates[0].keys():
            stacked_weights = torch.stack(
                [client_weights[key].float() for client_weights in client_updates], 
                dim=0
            )
            global_dict[key] = torch.mean(stacked_weights, dim=0).to(server.device)
            
        server.global_model.load_state_dict(global_dict)
        return global_dict
