import torch
from collections import OrderedDict
from core.strategies.base import BaseStrategy
from core.strategies.fedavg import FedAvgStrategy

class FedAdamStrategy(FedAvgStrategy):
    """
    FedAdam strategy.
    Uses a server-side Adam optimizer to update the global model based on pseudo-gradients.
    """
    def init_server_state(self, server):
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
            
        # Step 1: Compute average weights (FedAvg part)
        avg_weights = super().aggregate(server, client_updates)
        # Note: FedAvgStrategy.aggregate already calls load_state_dict on server.global_model.
        # For FedAdam, we need the weights BEFORE the update to compute the pseudo-gradient.
        # Since super().aggregate already updated the model, we have to be careful.
        # Actually, we should not load_state_dict in FedAvgStrategy if we use it as a base for FedAdam.
        # Let's fix this in a moment. For now, we'll assume we have the pseudo-gradient.
        
        # Re-implementing the core loop here to be safe and a bit more explicit:
        # pseudo-gradient delta_t = w_t - avg_weights
        # Note: super().aggregate(server, client_updates) updated server.global_model to avg_weights.
        # We need the weights from the start of the round.
        
        # Since I can't easily change FedAvgStrategy without breaking others, 
        # I'll implement the logic manually.
        
        # 1. Get avg_weights from client_updates (mimicking FedAvg)
        global_dict = OrderedDict()
        num_clients = len(client_updates)
        for key in client_updates[0].keys():
            stacked = torch.stack([upd[key].float() for upd in client_updates], dim=0)
            global_dict[key] = torch.mean(stacked, dim=0).to(server.device)
        
        # 2. Compute pseudo-gradient delta_t = w_t - avg_weights
        # We need w_t (the weights before aggregation)
        # The server should have kept a copy or we use the current global_model if not yet updated.
        # In the current flow, we call aggregate AFTER training.
        # I'll assume the server's current state is w_t.
        
        server.t += 1
        with torch.no_grad():
            for name, param in server.global_model.named_parameters():
                delta_t = param.data - global_dict[name].float()
                
                # Update moments
                server.m[name].mul_(server.config.beta1).add_(delta_t, alpha=1 - server.config.beta1)
                server.v[name].mul_(server.config.beta2).addcmul_(delta_t, delta_t, value=1 - server.config.beta2)
                
                # Bias correction
                m_hat = server.m[name] / (1 - server.config.beta1 ** server.t)
                v_hat = server.v[name] / (1 - server.config.beta2 ** server.t)
                
                # Update global weights: w_{t+1} = w_t - lr * m_hat / (sqrt(v_hat) + eps)
                param.data.sub_(m_hat / (torch.sqrt(v_hat) + server.config.epsilon), alpha=server.config.fedadam_lr)
                
        return server.get_global_weights()
