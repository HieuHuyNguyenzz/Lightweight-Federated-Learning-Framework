import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import sys

from core.config import FLConfig
from core.server import FLServer
from core.client import FLClient
from core.strategy import FedAvgStrategy, FedProxStrategy, ScaffoldStrategy, FedNovaStrategy
from utils.data_utils import get_dataset, partition_data
from utils.logger import setup_logger
from utils.csv_logger import CSVLogger
from models import get_model_for_dataset
from tqdm import tqdm
import gc

# Worker function for parallel training
def train_client_worker(client_id, model_class, train_dataset, global_weights, config, strategy_name, global_c=None, local_c=None):
    client = FLClient(client_id, model_class, train_dataset, config)
    weights = client.train(global_weights, strategy_type=strategy_name, global_c=global_c, local_c=local_c)
    return weights

def main():
    # 1. Load Configuration and Setup Logger
    config = FLConfig.load_from_yaml("config.yaml")
    logger = setup_logger()
    csv_logger = CSVLogger(config)
    
    logger.info("Starting Federated Learning Simulation")
    logger.info(f"Configuration: {config}")
    logger.info(f"Using device: {config.device}")
    logger.info(f"Results will be saved to: {csv_logger.filepath}")

    # Required for multiprocessing with CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 2. Data Preparation
    train_dataset, test_dataset = get_dataset(config.dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Partition data using config settings (supports iid, non-iid, dirichlet)
    client_datasets = partition_data(
        train_dataset, 
        config.num_clients, 
        config.partition_type, 
        config.dirichlet_alpha
    )
    logger.info(f"Dataset partitioned using type: {config.partition_type}")

    # 3. Framework Initialization
    strategy_map = {
        "fedavg": (FedAvgStrategy(), "FedAvg"),
        "fedprox": (FedProxStrategy(), "FedProx"),
        "scaffold": (ScaffoldStrategy(), "Scaffold"),
        "fednova": (FedNovaStrategy(), "FedNova"),
        "moon": (FedAvgStrategy(), "Moon"),
    }
    
    # Priority logic for automatic selection if not specified or based on config
    selected_strategy_key = config.strategy.lower()
    if selected_strategy_key not in strategy_map:
        # Fallback to original priority logic
        if config.partition_type == "dirichlet" or config.mu > 0:
            selected_strategy_key = "scaffold"
        else:
            selected_strategy_key = "fedavg"
            
    strategy, strategy_name = strategy_map[selected_strategy_key]
    
    logger.info(f"Using strategy: {strategy_name}")
    
    # Get model class based on dataset AND model_type
    model_class = get_model_for_dataset(config.dataset_name, config.model_type)
    server = FLServer(model_class, strategy, config)
    
    # Initialize Scaffold states if needed
    if strategy_name == "Scaffold":
        server._init_scaffold_states(config.num_clients)

    # 4. FL Simulation Loop
    for r in range(config.rounds):
        logger.info(f"--- Round {r+1}/{config.rounds} ---")
        local_updates_list = []
        global_weights = server.get_global_weights()
        
        # Process clients in parallel batches
        with tqdm(
            total=config.num_clients, 
            desc=f"Round {r+1} Training", 
            unit="client", 
            file=sys.stdout, 
            dynamic_ncols=True, 
            leave=True,
            ascii=True
        ) as pbar:
            for i in range(0, config.num_clients, config.max_parallel_clients):
                batch_indices = range(i, min(i + config.max_parallel_clients, config.num_clients))
                
                with mp.Pool(processes=len(batch_indices)) as pool:
                    # Prepare args for each client, including control variates for Scaffold
                    args = []
                    # Move global weights to CPU once for all workers in this batch to avoid CUDA IPC issues
                    global_weights_cpu = {k: v.cpu() for k, v in global_weights.items()}
                    
                    for idx in batch_indices:
                        g_c = server.global_c if strategy_name == "Scaffold" else None
                        if g_c:
                            g_c = {k: v.cpu() for k, v in g_c.items()}
                        
                        l_c = server.local_cs[idx] if strategy_name == "Scaffold" else None
                        if l_c:
                            l_c = {k: v.cpu() for k, v in l_c.items()}
                            
                        args.append((idx, model_class, client_datasets[idx], global_weights_cpu, config, strategy_name, g_c, l_c))
                    
                    batch_updates = pool.starmap(train_client_worker, args)
                    local_updates_list.extend(batch_updates)
                
                if config.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
                pbar.update(len(batch_indices))

        # Aggregation
        # server.aggregate now handles both simple weights and Scaffold updates
        server.aggregate(local_updates_list)
        
        # Evaluation
        accuracy = server.evaluate(test_loader)
        logger.info(f"Round {r+1} Global Accuracy: {accuracy:.2f}%")
        csv_logger.log_round(r + 1, accuracy)

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
