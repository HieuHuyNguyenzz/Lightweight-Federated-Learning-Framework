import torch
from collections import OrderedDict
from core.strategies.base import BaseStrategy

class FedDynStrategy(BaseStrategy):
    """
    FedDyn (Federated Dynamic) implementation.
    Uses dynamic weighted aggregation based on alpha coefficients.
    """
    def is_feddyn(self) -> bool:
        return True

    def init_server_state(self, server):
        # alpha coefficients for each client
        server.alphas = {} # {client_id: float}
        for i in range(server.config.num_clients):
            server.alphas[i] = 1.0

    def apply_local_loss(self, client, loss, data, target, alpha=1.0):
        # L_reg = alpha/2 * ||w - w_global||^2
        global_params = getattr(client, 'global_params', None)
        if global_params is None:
            return loss
            
        reg_term = 0.0
        for param, global_param in zip(client.model.parameters(), global_params):
            reg_term += ((param - global_param)**2).sum()
            
        return loss + (alpha / 2) * reg_term

    def aggregate(self, server, client_updates):
        if not client_updates:
            return None
            
        old_global_weights = server.get_global_weights()
        num_clients = len(client_updates)
        
        global_dict = OrderedDict()
        sum_alphas = sum(server.alphas.values())
        
        if sum_alphas == 0:
            return old_global_weights

        for key in client_updates[0].keys():
            weighted_sum = torch.zeros_like(client_updates[0][key].float()).to(server.device)
            for i in range(num_clients):
                weighted_sum += server.alphas[i] * client_updates[i][key].float().to(server.device)
            
            global_dict[key] = weighted_sum / sum_alphas
            
        server.global_model.load_state_dict(global_dict)
        
        # Update alphas for the next round
        global_update_norm = server._compute_norm(global_dict, old_global_weights)
        
        for i, update in enumerate(client_updates):
            local_update_norm = server._compute_norm(update, old_global_weights)
            if local_update_norm > 0:
                server.alphas[i] *= (global_update_norm / local_update_norm)
            
            # Clip alpha to prevent instability
            server.alphas[i] = max(0.01, min(server.alphas[i], 10.0))
        
        return global_dict

