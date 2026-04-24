from core.strategies.fedavg import FedAvgStrategy
from core.strategies.fedprox import FedProxStrategy
from core.strategies.scaffold import ScaffoldStrategy
from core.strategies.fednova import FedNovaStrategy
from core.strategies.moon import MoonStrategy
from core.strategies.feddyn import FedDynStrategy
from core.strategies.fedadam import FedAdamStrategy

def get_strategy(strategy_name):
    """
    Factory function to return the appropriate strategy instance.
    """
    strategies = {
        "fedavg": FedAvgStrategy(),
        "fedprox": FedProxStrategy(),
        "scaffold": ScaffoldStrategy(),
        "fednova": FedNovaStrategy(),
        "moon": MoonStrategy(),
        "feddyn": FedDynStrategy(),
        "fedadam": FedAdamStrategy(),
    }
    
    name = strategy_name.lower()
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(strategies.keys())}")
        
    return strategies[name]
