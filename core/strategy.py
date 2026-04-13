import torch
from abc import ABC, abstractmethod
from collections import OrderedDict

class AggregationStrategy(ABC):
    """Abstract base class for FL aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, client_updates, global_weights=None):
        """
        Perform weight aggregation.
        
        Args:
            client_updates (list): List of updates (state_dicts or dicts) from clients.
            global_weights (OrderedDict, optional): The current global weights.
            
        Returns:
            Any: The aggregated global weights or a tuple (weights, extra_info).
        """
        pass

class FedAvgStrategy(AggregationStrategy):
    """Federated Averaging (FedAvg) implementation."""
    
    def aggregate(self, client_updates, global_weights=None):
        global_dict = OrderedDict()
        num_clients = len(client_updates)
        
        if num_clients == 0:
            return None
            
        for key in client_updates[0].keys():
            stacked_weights = torch.stack(
                [client_weights[key].float() for client_weights in client_updates], 
                dim=0
            )
            global_dict[key] = torch.mean(stacked_weights, dim=0)
            
        return global_dict

class FedProxStrategy(FedAvgStrategy):
    """
    Federated Proximal (FedProx) implementation.
    FedProx uses the same aggregation as FedAvg, but modifies the local training loss.
    """
    def aggregate(self, client_updates, global_weights=None):
        return super().aggregate(client_updates, global_weights)

class ScaffoldStrategy(AggregationStrategy):
    """
    SCAFFOLD (Stochastic Controlled Averaging) implementation.
    SCAFFOLD uses control variates to correct client drift.
    """
    def aggregate(self, client_updates, global_weights=None):
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

class FedNovaStrategy(AggregationStrategy):
    """
    FedNova (Federated Nova) implementation.
    Addresses objective inconsistency by normalizing local updates.
    """
    def aggregate(self, client_updates, global_weights=None):
        if not global_weights:
            raise ValueError("FedNova requires global_weights for normalization.")
            
        num_clients = len(client_updates)
        if num_clients == 0:
            return None
            
        global_delta = OrderedDict({name: torch.zeros_like(param) for name, param in global_weights.items()})
        client_weight = 1.0 / num_clients
        
        for update in client_updates:
            w_i = update['weights']
            local_steps = update['local_steps']
            
            tau_i = local_steps
            normalization = 1.0 / (1.0 + tau_i)
            
            for name in global_weights.keys():
                if global_weights[name].dtype != torch.long:
                    delta_i = w_i[name].float() - global_weights[name].float()
                    global_delta[name] += client_weight * normalization * delta_i
        
        new_global_weights = OrderedDict()
        for name in global_weights.keys():
            if global_weights[name].dtype != torch.long:
                new_global_weights[name] = global_weights[name] + global_delta[name]
            else:
                new_global_weights[name] = global_weights[name]
            
        return new_global_weights

class FedDynStrategy(AggregationStrategy):
    """
    FedDyn (Federated Dynamic) implementation.
    Uses dynamic weighted aggregation based on alpha coefficients.
    """
    def aggregate(self, client_updates, global_weights=None, alphas=None):
        if alphas is None:
            raise ValueError("FedDyn requires alphas for weighted aggregation.")
            
        num_clients = len(client_updates)
        if num_clients == 0:
            return None
            
        global_dict = OrderedDict()
        sum_alphas = sum(alphas.values())
        
        if sum_alphas == 0:
            return global_weights

        for key in client_updates[0].keys():
            # Compute weighted sum: sum(alpha_i * w_i) / sum(alpha_i)
            weighted_sum = torch.zeros_like(client_updates[0][key].float())
            for i in range(num_clients):
                weighted_sum += alphas[i] * client_updates[i][key].float()
            
            global_dict[key] = weighted_sum / sum_alphas
            
        return global_dict


