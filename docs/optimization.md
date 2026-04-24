# Tối Ưu Hóa Bộ Nhớ và Hiệu Suất

Framework LFL phiên bản refactor sử dụng kiến trúc modular để tách biệt việc điều phối tài nguyên khỏi logic huấn luyện.

## 1. Quản Lý Tài Nguyên theo Batch (Batch-based Resource Management)
Thay vì khởi tạo toàn bộ clients, LFL chia client thành các nhóm nhỏ dựa trên `max_parallel_clients`.
- **Cơ chế**: Sử dụng `torch.multiprocessing.Pool` để chạy song song một nhóm client.
- **Thu hồi**: Gọi `gc.collect()` và `torch.cuda.empty_cache()` (chỉ dành cho CUDA) ngay sau khi một batch kết thúc.
- **Kết quả**: RAM tiêu thụ là hằng số $O(K)$ với $K$ là số client song song, không phụ thuộc vào tổng số client.

## 2. Tối Ưu hóa Pipeline Dữ Liệu
- **Async Loading**: Sử dụng `pin_memory=True` (chỉ dành cho CUDA) để tăng tốc độ nạp dữ liệu từ CPU $\rightarrow$ GPU.
- **Memory Footprint**: `num_workers` được đặt là `0` trong môi trường đa tiến trình (`mp.Pool`) để tránh lỗi daemon process của Python, nhưng vẫn đảm bảo hiệu suất nhờ vào cơ chế batching.

## 3. Truyền Trọng Số Tối Giản (State Dict Passing)
Framework chỉ truyền `state_dict` (OrderedDict của các tensor) giữa Server và Client.
- Không truyền đối tượng mô hình.
- Không truyền optimizer state (trừ khi thuật toán yêu cầu).
- Điều này giảm thiểu overhead về bộ nhớ và băng thông giao tiếp.

## 4. Chiến Lược Chia Dữ Liệu
- **IID Partitioning**: Chia đều dữ liệu, phù hợp cho baseline.
- **Non-IID Partitioning**: Sử dụng Dirichlet distribution, mô phỏng phân phối dữ liệu thực tế, giúp kiểm tra độ bền vững (robustness) của thuật toán gộp mà không làm tăng RAM.
