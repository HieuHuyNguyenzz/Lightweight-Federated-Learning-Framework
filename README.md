# Framework Federated Learning Siêu Nhẹ (LFL)

Một framework giả lập Học Liên kết modular, tối ưu RAM được triển khai bằng PyTorch.

## 🚀 Bắt Đầu Nhanh

### Cài Đặt
```bash
pip install torch torchvision numpy PyYAML tqdm
```

### Chạy Bản Demo
```bash
python main.//py
```

## 📚 Tài Liệu Chi Tiết

Hệ thống tài liệu hướng dẫn chi tiết cho mọi nhu cầu:

- [Kiến Trúc](./docs/architecture.md) - Thiết kế Hub-and-Spoke, Strategy Pattern và Parallel Batching.
- [Hướng Dẫn Cài Đặt](./docs/installation.md) - Yêu cầu hệ thống và cấu trúc thư mục.
- [Hướng Dẫn Sử Dụng](./docs/usage.md) - Cấu hình `config.yaml`, chọn model, dataset và thuật toán.
- [Hướng dẫn phát triển thuật toán](./docs/algorithm_dev.md) - Cách tạo và tích hợp thuật toán gộp mới.
- [Tối Ưu Hóa Bộ Nhớ](./docs/optimization.md) - Kỹ thuật quản lý RAM và hiệu suất pipeline.

## 🛠 Tính Năng Chính

### 1. Đa Dạng Dataset
Hỗ trợ sẵn các tập dữ liệu phổ biến:
- `mnist`, `fmnist` (Fashion-MNIST), `emnist`, `cifar10`.
- Hỗ trợ chia dữ liệu: `iid`, `non-iid` (sort-by-label), và `dirichlet` (phân phối xác suất).

### 2. Thư Viện Mô Hình (Model Zoo)
Tích hợp các kiến trúc CNN từ cơ bản đến SOTA (được tối ưu cho ảnh nhỏ):
- **GenericCNN**: Linh hoạt theo kích thước ảnh.
- **LeNet-5**: Kinh điển cho MNIST.
- **ResNet-18**: Sâu và mạnh mẽ, giảm overfitting.
- **VGG-11**: Cấu trúc convolution chuẩn.
- **MobileNetV2**: Siêu nhẹ, tối ưu cho thiết bị biên.

### 3. Thuật Toán Gộp (Aggregation)
- **FedAvg**: Baseline tiêu chuẩn.
- **FedProx**: Xử lý dữ liệu Non-IID bằng Proximal Term.
- **SCAFFOLD**: Chống trôi dạt mô hình (Client Drift) bằng biến điều khiển.

## 📂 Cấu Trúc Dự Án (Modular)
- `core/`: Lõi framework (Server, Client, Strategy, Config).
- `models/`: Thư viện các mô hình CNN.
- `utils/`: Công cụ hỗ trợ chia dữ liệu và logging.
- `main.py`: Điểm khởi chạy và cấu hình thí nghiệm.
- `config.yaml`: File cấu hình tham số.
