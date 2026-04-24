import torch
from core.strategies.fedavg import FedAvgStrategy

class FedProxStrategy(FedAvgStrategy):
    """
    Federated Proximal (FedProx) implementation.
    Adds a proximal term to the local loss to handle heterogeneity.
    """
    def apply_local_loss(self, client, loss, data, target, alpha=1.0):
        global_params = getattr(client, 'global_params', None)
        if global_params is None:
            return loss
            
        proximal_term = 0.0
        for param, global_param in zip(client.model.parameters(), global_params):
            proximal_term += ((param - global_param)**2).sum()
            
        return loss + (client.config.mu / 2) * proximal_term
