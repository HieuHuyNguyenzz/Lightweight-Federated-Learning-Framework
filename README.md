# Framework Federated Learning Siêu Nhẹ (LFL)

Một framework giả lập Học Liên kết modular, tối ưu RAM được triển khai bằng PyTorch.

## 🚀 Bắt Đầu Nhanh

### Cài Đặt
```bash
pip install torch torchvision numpy PyYAML tqdm
```

### Chạy Bản Demo
```bash
python main.py
```

## 📚 Tài Liệu Chi Tiết

Hệ thống tài liệu hướng dẫn chi tiết cho mọi nhu cầu:

- [Kiến Trúc](./docs/architecture.md) - Thiết kế Hub-and-Spoke, Strategy Pattern và Parallel Batching.
- [Hướng Dẫn Cài Đặt](./docs/installation.md) - Yêu cầu hệ thống và cấu trúc thư mục.
- [Hướng Dẫn Sử Dụng](./docs/usage.md) - Cấu hình `config.yaml`, theo dõi tiến độ và tùy chỉnh model.
- [Hướng dẫn phát triển thuật toán](./docs/algorithm_dev.md) - Cách tạo và tích hợp thuật toán gộp mới.
- [Tối Ưu Hóa Bộ Nhớ](./docs/optimization.md) - Kỹ thuật quản lý RAM và hiệu suất pipeline.

## 📂 Cấu Trúc Dự Án (Modular)
- `core/`: Lõi framework (Server, Client, Strategy, Config).
- `utils/`: Công cụ hỗ trợ chia dữ liệu và logging.
- `model.py`: Định nghĩa mô hình nơ-ron.
- `main.py`: Điểm khởi chạy và cấu hình thí nghiệm.
- `config.yaml`: File cấu hình tham số.
