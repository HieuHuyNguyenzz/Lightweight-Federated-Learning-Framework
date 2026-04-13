import torch
from abc import ABC, abstractmethod
from collections import OrderedDict

class AggregationStrategy(ABC):
    """Abstract base class for FL aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, client_weights_list) -> OrderedDict:
        """
        Perform weight aggregation.
        
        Args:
            client_weights_list (list): List of state_dicts from clients.
            
        Returns:
            OrderedDict: The aggregated global weights.
        """
        pass

class FedAvgStrategy(AggregationStrategy):
    """Federated Averaging (FedAvg) implementation."""
    
    def aggregate(self, client_weights_list):
        global_dict = OrderedDict()
        num_clients = len(client_weights_list)
        
        if num_clients == 0:
            return None
            
        for key in client_weights_list[0].keys():
            stacked_weights = torch.stack(
                [client_weights[key].float() for client_weights in client_weights_list], 
                dim=0
            )
            global_dict[key] = torch.mean(stacked_weights, dim=0)
            
        return global_dict

class FedProxStrategy(FedAvgStrategy):
    """
    Federated Proximal (FedProx) implementation.
    FedProx uses the same aggregation as FedAvg, but modifies the local training loss.
    """
    def aggregate(self, client_weights_list):
        return super().aggregate(client_weights_list)

class ScaffoldStrategy(AggregationStrategy):
    """
    SCAFFOLD (Stochastic Controlled Averaging) implementation.
    SCAFFOLD uses control variates to correct client drift.
    """
    def aggregate(self, client_updates):
        global_dict = OrderedDict()
        global_c_update = OrderedDict()
        num_clients = len(client_updates)
        
        if num_clients == 0:
            return None, None
            
        for key in client_updates[0]['weights'].keys():
            stacked_weights = torch.stack(
                [upd['weights'][key].float() for upd in client_updates], 
                dim=0
            )
            global_dict[key] = torch.mean(stacked_weights, dim=0)
            
        for key in client_updates[0]['grad_avg'].keys():
            stacked_grads = torch.stack(
                [upd['grad_avg'][key].float() for upd in client_updates], 
                dim=0
            )
            global_c_update[key] = torch.mean(stacked_grads, dim=0)
            
        return global_dict, global_c_update


