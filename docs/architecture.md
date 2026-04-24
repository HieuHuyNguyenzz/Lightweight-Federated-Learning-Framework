# Architecture (Strategy-Centric)

The Lightweight Federated Learning (LFL) framework employs a **Strategy-Centric** modular architecture. This design decouples the training orchestration from the specific mathematical implementation of aggregation algorithms, ensuring high extensibility and RAM efficiency.

## 🏗 Modular Design

The framework is split into three main layers:

### 1. Core Engine (`/core`)
This layer handles the "how" of Federated Learning (communication, parallelization, and state management).
- **`FLServer`**: A lean orchestrator. It no longer contains algorithm-specific logic. Instead, it delegates aggregation and state initialization to the active `Strategy` object.
- **`FLClient`**: A generic trainer. It performs local SGD and delegates loss modification (e.g., proximal terms) and gradient correction to the `Strategy` object.
- **`strategies/`**: The heart of the framework. Each algorithm (FedAvg, FedAdam, etc.) is encapsulated in its own class inheriting from `BaseStrategy`.
- **`FLConfig`**: A centralized configuration object for easy hyperparameter management.

### 2. Utility Layer (`/utils`)
Helper functions for data and logging.
- **`partition_data`**: Supports **IID** and **Non-IID** (Dirichlet distribution) partitioning.
- **`CSVLogger`**: Saves results with the format `{dataset}_{strategy}_{partition}_{clients}.csv`.

### 3. Experiment Layer (`main.py` & `models/`)
- **`models/`**: Library of CNN architectures.
- **`main.py`**: The entry point that wires together the dataset, model, and strategy.

## 🚀 Scalability & Extensibility

### Memory Management
LFL continues to use **Parallel Batching** via `torch.multiprocessing`. By adjusting `max_parallel_clients`, you can scale the simulation to hundreds of clients while keeping RAM usage constant ($O(max\_parallel)$ instead of $O(total\_clients)$).

### The Strategy Pattern
Adding a new algorithm no longer requires modifying the Server or Client code. 
1. Create a new strategy class in `core/strategies/`.
2. Implement `aggregate` (server-side) and `apply_local_loss` (client-side).
3. Register it in `core/strategies/__init__.py`.
4. Select it in `config.yaml`.
