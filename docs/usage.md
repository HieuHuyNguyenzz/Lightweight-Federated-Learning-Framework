# Hướng Dẫn Sử Dụng (Phiên bản Modular)

Tài liệu này hướng dẫn cách vận hành LFL framework.

## Chạy Bản Demo
Để thực thi giả lập:

```bash
python main.py
```

## Quản Lý Cấu Hình (`config.yaml`)
Toàn bộ tham số được quản lý trong `config.yaml`. Bạn có thể thay đổi chúng mà không cần sửa code.

| Tham số | Mô tả | Gợi ý |
| :--- | :--- | :--- |
| `num_clients` | Tổng số client | Tùy quy mô mô phỏng |
| `rounds` | Số vòng giao tiếp | 10 - 100 cho hội tụ |
| `local_epochs` | Epoch mỗi client | 1 - 5 |
| `lr` | Learning rate | 0.01 |
| `batch_size` | Kích thước batch | 32 |
| `max_parallel_clients` | Số client chạy song song | Điều chỉnh theo RAM |
| `dataset_name` | Tập dữ liệu | `"mnist"`, `"fmnist"`, `"emnist"`, `"cifar10"` |
| `model_type` | Kiến trúc model | `"generic"`, `"lenet"`, `"resnet18"`, `"vgg11"`, `"mobilenet"` |
| `partition_type` | Kiểu chia dữ liệu | `"iid"`, `"non-iid"` (Dirichlet) |
| `dirichlet_alpha` | Độ lệch dữ liệu | $\alpha$ thấp $\rightarrow$ Non-IID cao |
| `strategy` | Thuật toán gộp | `"fedavg"`, `"fedprox"`, `"scaffold"`, `"fednova"`, `"moon"`, `"feddyn"`, `"fedadam"`, `"fedyogi"`, `"fedadagrad"` |
| `mu` | Proximal term (FedProx) | 0.01 |
| `moon_mu` | Contrastive weight (MOON) | 0.1 |
| `fedadam_lr` | Server learning rate | 0.01 |
| `fedyogi_lr` | Server learning rate | 0.01 |
| `fedadagrad_lr` | Server learning rate | 0.01 |
| `beta1` | Momentum coefficient | 0.9 |
| `beta2` | Variance coefficient | 0.99 |
| `epsilon` | Stability term | 1e-8 |
| `device` | Thiết bị chạy | `"auto"`, `"cuda"`, `"mps"`, `"cpu"` |
| `dp_enabled` | Bật bảo mật vi sai | `False` |
| `dp_clip_norm` | Ngưỡng cắt gradient | 1.0 |
| `dp_sigma` | Hệ số nhiễu Gaussian | 0.1 |

## Theo Dõi Tiến Độ và Kết Quả
- **Progress Bar**: Framework sử dụng `tqdm` để hiển thị tiến trình huấn luyện của từng Round ngay trên một dòng duy nhất.
- **Kết quả CSV**: Mỗi lần chạy sẽ tạo ra một file `.csv` lưu trữ độ chính xác trong thư mục `results/`.
   - **Quy tắc đặt tên**: `{dataset}_{strategy}_{partition}_{clients}.csv`

## Tùy Chỉnh Framework

### 1. Thêm Mô Hình Mới
1. Định nghĩa model trong `models/base.py` hoặc `models/torchvision_wrappers.py`.
2. Thêm model vào mapping trong `models/__init__.py`.

### 2. Thêm Thuật Toán Gộp Mới
Vui lòng xem hướng dẫn chi tiết tại [Hướng dẫn phát triển thuật toán mới](./algorithm_dev.md).

### 3. Thử Nghiệm Dữ Liệu Non-IID
Đặt `partition_type="non-iid"` trong `config.yaml` để mô phỏng phân phối dữ liệu thực tế (sử dụng Dirichlet distribution).
