# Orchest Computing: Orchestrator MLOps Phi Tập Trung Cho Thiết Bị Biên Đa Chủng Loại

**Orchest Computing** là nền tảng MLOps phân tán dạng Universal App, hỗ trợ kết nối các máy tính không đồng nhất (Windows, macOS, Linux) trong mạng nội bộ (LAN) thành một cụm suy luận (Swarm Inference).

## Tính năng chính
1. **Auto-Discovery & Profiling**: Tự động phát hiện các nút (nodes) trong LAN qua UDP, đo đạc thời gian thực CPU, GPU (NVML/Metal), RAM/VRAM khả dụng và trễ mạng.
2. **Dynamic Programming Model Slicing**: Thuật toán quy hoạch động tự động tìm điểm cắt mô hình (Layer Partitioning) tối ưu dựa trên dung lượng bộ nhớ và băng thông mạng.
3. **High-Speed Binary Tensor Transport**: Truyền nhận Tensor dạng binary không làm suy giảm độ chính xác qua ZeroMQ / Custom TCP sockets, kết hợp mã hóa AES-GCM cho bảo mật cấp biên.
4. **Pipeline Swarm Execution & Micro-batching**: Thực thi gối đầu các khối layer bất đồng bộ giữa các nút trong LAN.
5. **Fault Tolerance**: Tự động phát hiện nút hỏng hóc/ngắt kết nối và phân bổ lại mô hình (Dynamic Re-slicing).

## Cấu trúc thư mục
- `orchest/profiler/`: Hệ thống đo đạc phần cứng & đồ thị mô hình.
- `orchest/partitioner/`: Thuật toán Quy hoạch động cắt nhỏ mô hình.
- `orchest/network/`: Tự động quét mạng (UDP discovery), socket truyền nhận Tensor binary (ZeroMQ/TCP), Micro-batching.
- `orchest/executor/`: Quản lý nút Host (Master) và Worker, luồng thực thi Swarm Pipeline.
- `orchest/models/`: Wrapper chia tách layer cho các mô hình HF (Qwen2, Llama, v.v.).
- `apps/`: CLI (`orchest-cli`) và Web Dashboard quản lý MLOps.
- `tests/`: Bộ kiểm thử tự động.

## Hướng dẫn cài đặt & Chạy thử
```bash
pip install -e .
# Hoặc: pip install -r requirements.txt

# Khởi chạy nút Worker trên máy 1/máy 2:
python apps/worker_node.py --port 5555

# Khởi chạy nút Host trên máy Master:
python apps/host_node.py --model Qwen/Qwen2.5-0.5B-Instruct
```
