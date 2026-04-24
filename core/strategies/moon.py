import torch
import copy
from core.strategies.fedavg import FedAvgStrategy

class MoonStrategy(FedAvgStrategy):
    """
    MOON (Model-Contrastive Federated Learning) implementation.
    Adds a contrastive loss to the local training.
    """
    def is_moon(self) -> bool:
        return True

    def setup_moon_models(self, client, prev_weights):
        # Setup frozen models for representation extraction
        client.global_model_frozen = copy.deepcopy(client.model).eval()
        for p in client.global_model_frozen.parameters():
            p.requires_grad = False
        
        client.prev_model_frozen = copy.deepcopy(client.model).eval()
        client.prev_model_frozen.load_state_dict(prev_weights)
        for p in client.prev_model_frozen.parameters():
            p.requires_grad = False

    def apply_local_loss(self, client, loss, data, target, alpha=None):
        # MOON Contrastive Loss logic
        if not hasattr(client, 'global_model_frozen') or not hasattr(client, 'prev_model_frozen'):
            return loss
            
        with torch.no_grad():
            # Extract representations (features)
            # We use a helper to get the features before the final FC layer
            z_global = self._get_features(client.global_model_frozen, data)
            z_prev = self._get_features(client.prev_model_frozen, data)
        
        z_curr = self._get_features(client.model, data)
        
        # Contrastive loss calculation
        cos_global = torch.nn.functional.cosine_similarity(z_curr, z_global, dim=1)
        cos_prev = torch.nn.functional.cosine_similarity(z_curr, z_prev, dim=1)
        
        contrastive_loss = 0.5 * (1 - cos_global) + 0.5 * (1 - cos_prev)
        contrastive_loss = contrastive_loss.mean()
        
        return loss + client.config.moon_mu * contrastive_loss

    def _get_features(self, model, data):
        # Simple way to get features: remove the last layer
        # For GenericCNN, we can use a hook or just a modified forward
        # To keep it simple and general, we'll assume the model has forward_features 
        # or we use a temporary hack to get the penultimate output.
        if hasattr(model, 'forward_features'):
            return model.forward_features(data)
        
        # Fallback: try to find the last linear layer and remove it
        # This is tricky for different models. 
        # For our GenericCNN, we know it ends with fc2.
        # Let's just assume the model is modified to have forward_features 
        # or we use a simple hook.
        
        # For this project, I'll add forward_features to GenericCNN in a moment.
        return model.forward_features(data)

