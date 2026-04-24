from abc import ABC, abstractmethod
import torch
from collections import OrderedDict

from typing import Any

class BaseStrategy(ABC):
    """
    Abstract base class for all Federated Learning aggregation strategies.
    This class defines the interface for both server-side aggregation and 
    client-side loss modification.
    """

    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, server, client_updates) -> Any:
        """
        Perform the global aggregation on the server.
        """
        pass

    def apply_local_loss(self, client, loss, data, target, alpha=1.0):
        """
        Optional method to modify the local training loss.
        Implement this for strategies like FedProx or MOON.
        
        Args:
            client (FLClient): The client instance.
            loss (torch.Tensor): The standard cross-entropy loss.
            data (torch.Tensor): Current input batch.
            target (torch.Tensor): Current target labels.
            alpha (float): Dynamic weight for strategies like FedDyn.
            
        Returns:
            torch.Tensor: The modified loss.
        """
        return loss


    def modify_gradients(self, client, model):
        """
        Optional method to modify gradients before the optimizer step.
        """
        pass

    def init_server_state(self, server):
        """
        Optional method to initialize server-side states.
        """
        pass

    def init_client_state(self, client):
        """
        Optional method to initialize client-side states.
        """
        pass

    def is_scaffold(self) -> bool:
        return False

    def is_feddyn(self) -> bool:
        return False

    def is_fednova(self) -> bool:
        return False

    def is_moon(self) -> bool:
        return False
