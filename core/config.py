import yaml
import torch
from dataclasses import dataclass

@dataclass
class FLConfig:
    num_clients: int = 10
    rounds: int = 5
    local_epochs: int = 1
    lr: float = 0.01
    max_parallel_clients: int = 2
    device: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 32
    dataset_name: str = "mnist"
    model_type: str = "generic" # Options: "generic", "lenet", "resnet18", "vgg11", "mobilenet"
    partition_type: str = "iid"
    dirichlet_alpha: float = 0.5
    strategy: str = "fedavg" # Options: "fedavg", "fedprox", "scaffold", "fednova", "moon", "feddyn"
    # FedProx specific parameter
    mu: float = 0.01 
    # MOON specific parameter
    moon_mu: float = 0.1
    # FedAdam specific parameters
    fedadam_lr: float = 0.01
    # FedYogi specific parameters
    fedyogi_lr: float = 0.01
    # FedAdagrad specific parameters
    fedadagrad_lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1e-8


    @classmethod
    def load_from_yaml(cls, path="config.yaml"):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert device string to torch.device
        if 'device' in config_dict:
            device_str = config_dict['device']
            if device_str == 'auto':
                config_dict['device'] = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            else:
                config_dict['device'] = torch.device(device_str)
            
        return cls(**config_dict)
