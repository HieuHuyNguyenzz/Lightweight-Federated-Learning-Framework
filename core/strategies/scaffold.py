import torch
from collections import OrderedDict
from core.strategies.base import BaseStrategy

class ScaffoldStrategy(BaseStrategy):
    """
    SCAFFOLD (Stochastic Controlled Averaging) implementation.
    Corrects client drift using control variates.
    """
    def is_scaffold(self) -> bool:
        return True

    def init_server_state(self, server):
        # Initialize global control variate
        server.global_c = OrderedDict({
            name: torch.zeros_like(param).to(server.device) 
            for name, param in server.global_model.named_parameters()
        })
        # Initialize local control variates for all clients to persist across rounds
        server.local_cs = {
            i: OrderedDict({
                name: torch.zeros_like(param).to(server.device) 
                for name, param in server.global_model.named_parameters()
            })
            for i in range(server.config.num_clients)
        }

    def init_client_state(self, client):
        # local_c is now managed by the server to persist across rounds
        pass

    def modify_gradients(self, client, model):
        # Apply Scaffold correction: grad = grad - local_c + global_c
        # Note: client.server_global_c and client.local_c must be set during train()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    c_i = client.local_c[name].to(client.device)
                    c_g = client.server_global_c[name].to(client.device)
                    param.grad.data.add_(-c_i).add_(c_g)

    def aggregate(self, server, client_updates):
        if not client_updates:
            return None
            
        global_weights = OrderedDict()
        global_c_update = OrderedDict()
        num_clients = len(client_updates)
        
        # Average weights
        for key in client_updates[0]['weights'].keys():
            stacked_weights = torch.stack(
                [upd['weights'][key].float() for upd in client_updates], 
                dim=0
            )
            global_weights[key] = torch.mean(stacked_weights, dim=0).to(server.device)
            
        # Average control variates
        for key in client_updates[0]['grad_avg'].keys():
            stacked_grads = torch.stack(
                [upd['grad_avg'][key].float() for upd in client_updates], 
                dim=0
            )
            global_c_update[key] = torch.mean(stacked_grads, dim=0).to(server.device)
            
        # Update server global_c
        with torch.no_grad():
            for name, update in global_c_update.items():
                server.global_c[name].add_(update)
                
        # Update clients local_cs in server state: c_i_next = c_i + (global_c_next - global_c_curr)
        # We'll handle this in the server.aggregate or by sending the update back.
        # The most consistent way is to let the server update its local_cs storage.
        for i in range(num_clients):
            for name, update in global_c_update.items():
                server.local_cs[i][name].add_(update)
        
        server.global_model.load_state_dict(global_weights)
        return global_weights

