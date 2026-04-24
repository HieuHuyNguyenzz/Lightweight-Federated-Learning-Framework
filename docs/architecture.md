# Architecture (Refactored)

The Lightweight Federated Learning (LFL) framework is now a modular system designed for high extensibility and RAM efficiency.

## 🏗 Modular Design

The framework is split into three main layers to decouple logic from implementation.

### 1. Core Engine (`/core`)
This layer contains the logic that remains constant regardless of the model or dataset used.
- **`FLServer`**: Orchestrates the FL process. It is agnostic to the aggregation algorithm used.
- **`FLClient`**: Handles local training. It can take any `nn.Module` and any `Dataset`.
- **`AggregationStrategy`**: An abstract base class. New algorithms (FedProx, FedAdam) can be added by creating new subclasses without touching the Server code.
- **`FLConfig`**: A centralized configuration object for easy hyperparameter management.

### 2. Utility Layer (`/utils`)
Helper functions for data management.
- **`partition_data`**: Now supports both **IID** and **Non-IID** (Dirichlet distribution) partitioning to simulate real-world data skew.

### 3. Experiment Layer (`main.py` & `models/`)
The user-defined part of the framework.
- **`models/`**: Thư viện các mô hình CNN.
- **`main.py`**: The "glue" code that chooses the model, the dataset, and the strategy, then launches the simulation.

## 🚀 Scalability Features

### Memory Management
LFL continues to use **Parallel Batching** via `torch.multiprocessing`. By adjusting `max_parallel_clients`, you can scale the simulation to hundreds of clients while keeping RAM usage constant ($O(max\_parallel)$ instead of $O(total\_clients)$).

### Extensibility Path
- **To add a new model**: Create it in `models/base.py` or `models/torchvision_wrappers.py` $\rightarrow$ add to mapping in `models/__init__.py` $\rightarrow$ use in `main.py`.
- **To add a new algorithm**: Create a new class in `core/strategy.py` inheriting from `AggregationStrategy` $\rightarrow$ pass it to `FLServer`.
- **To test Non-IID data**: Change `partition_type="non-iid"` in config. Non-IID sử dụng Dirichlet distribution để mô phỏng phân phối dữ liệu thực tế.
