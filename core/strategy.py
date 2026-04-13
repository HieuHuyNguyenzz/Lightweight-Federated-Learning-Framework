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
        # FedProx uses the same averaging aggregation as FedAvg
        return super().aggregate(client_weights_list)
