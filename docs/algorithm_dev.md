# Hướng dẫn phát triển thuật toán mới (Aggregation Algorithm)

Một trong những điểm mạnh nhất của LFL framework là tính linh hoạt trong việc mở rộng các thuật toán gộp trọng số nhờ vào **Strategy Pattern**. Dưới đây là hướng dẫn chi tiết để bạn thêm một thuật toán mới.

## 🛠 Quy trình triển khai (3 bước)

### Bước 1: Định nghĩa thuật toán trong `core/strategy.py`

Bạn cần tạo một class mới kế thừa từ `AggregationStrategy`. Bạn chỉ cần triển khai logic gộp trọng số trong phương thức `aggregate`.

**Ví dụ: Triển khai một thuật toán gộp đơn giản (ví dụ: Weighted FedAvg)**
```python
from core.strategy import AggregationStrategy
from collections import OrderedDict
import torch

class WeightedFedAvgStrategy(AggregationStrategy):
    def aggregate(self, client_weights_list):
        # Giả sử mỗi client đóng góp trọng số khác nhau dựa trên lượng dữ liệu
        # Ở đây chúng ta mô phỏng bằng cách gán trọng số ngẫu nhiên cho ví dụ
        global_dict = OrderedDict()
        num_clients = len(client_weights_list)
        
        # Giả sử trọng số của mỗi client là 1/K (giống FedAvg) 
        # nhưng bạn có thể tùy biến logic tại đây
        weights = [1.0 / num_clients] * num_clients 
        
        for key in client_weights_list[0].keys():
            # Tính tổng có trọng số: sum(weight_i * tensor_i)
            weighted_sum = torch.stack(
                [client_weights[key].float() * w for client_weights, w in zip(client_weights_list, weights)], 
                dim=0
            ).sum(dim=0)
            
            global_dict[key] = weighted_sum
            
        return global_dict
```

### Bước 2: Tích hợp vào `main.py`

Trong file `main.py`, thay vì khởi tạo `FedAvgStrategy()`, bạn hãy khởi tạo class thuật toán mới mà bạn vừa tạo.

```python
# Trong main.py
from core.strategy import WeightedFedAvgStrategy # Import class mới

# ... trong hàm main() ...
strategy = WeightedFedAvgStrategy() # Thay thế FedAvgStrategy
server = FLServer(MNISTNet, strategy, config)
```

### Bước 3: Kiểm tra và Đánh giá
Chạy mô phỏng và theo dõi file `fl_simulation.log` để so sánh độ chính xác giữa thuật toán mới và `FedAvg`.

---

## 💡 Một số gợi ý cho nghiên cứu
Nếu bạn muốn phát triển các thuật toán nâng cao hơn:
- **FedProx**: Bạn cần truyền thêm tham số $\mu$ (proximal term) vào `FLClient` để điều chỉnh hàm loss cục bộ.
- **FedAdam / FedAdagrad**: Bạn cần thay đổi `FLServer` để lưu trữ các state (momentum, velocity) của mô hình toàn cục thay vì chỉ gộp trung bình đơn giản.
- **Sparsification**: Bạn có thể tạo một `Strategy` mà chỉ gửi những trọng số có giá trị thay đổi lớn nhất về Server để giảm băng thông.
