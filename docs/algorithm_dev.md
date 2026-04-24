# Hướng dẫn phát triển thuật toán mới (Aggregation Algorithm)

Một trong những điểm mạnh nhất của LFL framework là tính linh hoạt trong việc mở rộng các thuật toán gộp trọng số nhờ vào **Strategy Pattern**. Dưới đây là hướng dẫn chi tiết để bạn thêm một thuật toán mới.

## 🛠 Quy trình triển khai (3 bước)

### Bước 1: Định nghĩa thuật toán trong `core/strategies/`

Bạn cần tạo một file mới (ví dụ: `my_algo.py`) trong thư mục `core/strategies/` và tạo một class kế thừa từ `BaseStrategy`.

**Các phương thức bạn có thể cần triển khai:**
- `init_server_state(self, server)`: Khởi tạo các biến trạng thái cho Server (ví dụ: momentum trong FedAdam).
- `init_client_state(self, client)`: Khởi tạo các biến cho Client (ví dụ: control variates trong Scaffold).
- `apply_local_loss(self, client, loss, data, target, alpha)`: Thay đổi hàm loss cục bộ (ví dụ: thêm proximal term cho FedProx).
- `modify_gradients(self, client, model)`: Điều chỉnh gradient trước khi update (ví dụ: Scaffold).
- `aggregate(self, server, client_updates)`: Logic gộp trọng số chính tại Server.

**Ví dụ: Triển khai một thuật toán gộp đơn giản (Weighted FedAvg)**
```python
from core.strategies.base import BaseStrategy
from collections import OrderedDict
import torch

class WeightedFedAvgStrategy(BaseStrategy):
    def aggregate(self, server, client_updates):
        global_dict = OrderedDict()
        num_clients = len(client_updates)
        weights = [1.0 / num_clients] * num_clients 
        
        for key in client_updates[0].keys():
            weighted_sum = torch.stack(
                [client_weights[key].float() * w for client_weights, w in zip(client_updates, weights)], 
                dim=0
            ).sum(dim=0)
            global_dict[key] = weighted_sum.to(server.device)
            
        server.global_model.load_state_dict(global_dict)
        return global_dict
```

### Bước 2: Đăng ký thuật toán vào Factory

Mở file `core/strategies/__init__.py`, import class mới của bạn và thêm vào từ điển `strategies`.

```python
from core.strategies.my_algo import WeightedFedAvgStrategy

def get_strategy(strategy_name):
    strategies = {
        "fedavg": FedAvgStrategy(),
        "myalgo": WeightedFedAvgStrategy(), # Đăng ký tại đây
        # ...
    }
    return strategies[strategy_name.lower()]
```

### Bước 3: Kiểm tra và Đánh giá

1. Đặt `strategy: myalgo` trong `config.yaml`.
2. Chạy mô phỏng: `python main.py`.
3. Theo dõi kết quả trong thư mục `results/`.

---

## 💡 Một số gợi ý cho nghiên cứu
Nếu bạn muốn phát triển các thuật toán nâng cao hơn:
- **FedProx**: Implement `apply_local_loss` để thêm proximal term.
- **SCAFFOLD**: Implement `init_server_state`, `init_client_state` và `modify_gradients`.
- **FedAdam**: Implement `init_server_state` và logic update Adam trong `aggregate`.
- **Sparsification**: Tạo một `Strategy` mà chỉ gửi những trọng số quan trọng nhất về Server.
