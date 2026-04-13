# Hướng Dẫn Cài Đặt

Tài liệu này hướng dẫn cách thiết lập môi trường cho framework Federated Learning Siêu Nhẹ (LFL).

## Yêu Cầu Hệ Thống
- **Hệ điều hành**: Windows, Linux, hoặc macOS.
- **Phiên bản Python**: 3.8 trở lên.
- **Phần cứng**: 
  - RAM: Tối thiểu 4GB.
  - GPU: Tùy chọn (Hỗ trợ CUDA để tăng tốc).

## Cài Đặt Thư Viện Phụ Thuộc

```bash
pip install torch torchvision numpy PyYAML tqdm
```

## Cấu Trúc Thư Mục
- `core/`: Lõi framework (Server, Client, Strategy, Config).
- `models/`: Thư viện các mô hình CNN (Generic, LeNet, ResNet, VGG, MobileNet).
- `utils/`: Công cụ hỗ trợ chia dữ liệu và logging.
- `main.py`: Điểm khởi chạy giả lập.
- `config.yaml`: Cấu hình toàn bộ thí nghiệm.
- `docs/`: Tài liệu hướng dẫn chi tiết.

## Kiểm Tra Cài Đặt

```bash
python main.py
```
