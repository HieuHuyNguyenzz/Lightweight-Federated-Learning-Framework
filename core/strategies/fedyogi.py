import torch
from collections import OrderedDict
from core.strategies.base import BaseStrategy

class FedYogiStrategy(BaseStrategy):
    """
    FedYogi implementation.
    An adaptive optimizer for Federated Learning that improves upon FedAdam
    by modifying the variance update rule to be more stable.
    """
    def init_server_state(self, server):
        # Initialize first and second moments
        server.m = OrderedDict({
            name: torch.zeros_like(param).to(server.device) 
            for name, param in server.global_model.named_parameters()
        })
        server.v = OrderedDict({
            name: torch.zeros_like(param).to(server.device) 
            for name, param in server.global_model.named_parameters()
        })
        server.t = 0

    def aggregate(self, server, client_updates):
        if not client_updates:
            return None
            
        # 1. Compute the average weights (FedAvg part)
        global_dict = OrderedDict()
        num_clients = len(client_updates)
        for key in client_updates[0].keys():
            stacked = torch.stack([upd[key].float() for upd in client_updates], dim=0)
            global_dict[key] = torch.mean(stacked, dim=0).to(server.device)
        
        # 2. Apply Yogi update rule on server side
        server.t += 1
        with torch.no_grad():
            for name, param in server.global_model.named_parameters():
                # Pseudo-gradient delta_t = w_t - avg_weights
                delta_t = param.data - global_dict[name].float()
                
                # Update first moment (momentum)
                server.m[name].mul_(server.config.beta1).add_(delta_t, alpha=1 - server.config.beta1)
                
                # Update second moment (Yogi rule)
                # v_t = v_{t-1} + sign(delta_t^2 - v_{t-1}) * sqrt(v_{t-1}) * (1 - beta_2)
                v_prev = server.v[name]
                delta_sq = delta_t ** 2
                sign_diff = torch.sign(delta_sq - v_prev)
                
                server.v[name].add_(sign_diff * torch.sqrt(v_prev) * (1 - server.config.beta2))
                
                # Bias correction for first moment
                m_hat = server.m[name] / (1 - server.config.beta1 ** server.t)
                
                # Update global weights
                # w_{t+1} = w_t - lr * m_hat / (sqrt(v_t) + eps)
                param.data.sub_(m_hat / (torch.sqrt(server.v[name]) + server.config.epsilon), alpha=server.config.fedyogi_lr)
                
        return server.get_global_weights()
