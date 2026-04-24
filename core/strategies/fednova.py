import torch
from collections import OrderedDict
from core.strategies.base import BaseStrategy

class FedNovaStrategy(BaseStrategy):
    """
    FedNova (Federated Nova) implementation.
    Addresses objective inconsistency by normalizing local updates.
    """
    def is_fednova(self) -> bool:
        return True

    def aggregate(self, server, client_updates):
        if not client_updates:
            return None
            
        global_weights = server.get_global_weights()
        num_clients = len(client_updates)
        
        global_delta = OrderedDict({name: torch.zeros_like(param).to(server.device) for name, param in global_weights.items()})
        client_weight = 1.0 / num_clients
        
        for update in client_updates:
            w_i = update['weights']
            local_steps = update['local_steps']
            
            tau_i = local_steps
            normalization = 1.0 / (1.0 + tau_i)
            
            for name in global_weights.keys():
                if global_weights[name].dtype != torch.long:
                    delta_i = w_i[name].float().to(server.device) - global_weights[name].float()
                    global_delta[name] += client_weight * normalization * delta_i
        
        new_global_weights = OrderedDict()
        for name in global_weights.keys():
            if global_weights[name].dtype != torch.long:
                new_global_weights[name] = global_weights[name] + global_delta[name]
            else:
                new_global_weights[name] = global_weights[name]
        
        server.global_model.load_state_dict(new_global_weights)
        return new_global_weights
