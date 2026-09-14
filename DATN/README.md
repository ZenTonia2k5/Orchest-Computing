# Orchest Computing: Orchestrator MLOps Phi Tập Trung Cho Thiết Bị Biên Đa Chủng Loại

> 📖 **Tài liệu Kỹ thuật Chi tiết**: Toàn bộ cấu trúc hệ thống, logic toán học, phương án kỹ thuật và quy trình thực nghiệm được ghi chép đầy đủ tại: [PROJECT_ARCHITECTURE_AND_TECHNICAL_DESIGN.md](file:///Applications/HCMUT/Orchest%20Computing/PROJECT_ARCHITECTURE_AND_TECHNICAL_DESIGN.md).

**Orchest Computing** là nền tảng MLOps phân tán dạng Universal App, hỗ trợ kết nối các máy tính không đồng nhất (Windows, macOS, Linux) trong mạng nội bộ (LAN) thành một cụm suy luận phân tán (Swarm Inference).

## 🚀 Tính năng chính

1. **Universal Unified App (`apps/orchest.py`)**:
   - Một ứng dụng duy nhất dùng chung cho mọi máy tính trong mạng LAN.
   - Chuyển đổi linh hoạt giữa **👑 Host Master Mode**, **⚡ Worker Edge Mode**, và **🛠️ Manage Fleet Tab**.

2. **Auto-Balancing Slicing Bar ($\sum = 100\%$)**:
   - Hỗ trợ cắt mô hình tự động qua thuật toán Quy hoạch động (Auto DP Solver) hoặc tùy chỉnh bằng thanh trượt tỉ lệ (Ratio Cutting).
   - Tự động cân bằng tổng tỉ lệ đúng 100% khi người dùng kéo thanh trượt của bất kỳ máy tính nào.

3. **Worker Connection Notification & Authorization (Accept / Reject)**:
   - Khi Host phân bổ mô hình, Worker nhận thông báo trực quan (Popup Banner) kèm chi tiết số layer và IP của Host.
   - Worker có thể bấm **✅ Accept** để nhận tải mô hình hoặc **❌ Reject** để từ chối.
   - Hỗ trợ chế độ **Auto-Accept** hoặc phê duyệt thủ công.

4. **Connection History & Time Logs**:
   - Lưu trữ nhật ký kết nối theo thời gian thực (Timestamp, Host ID, Host IP, Model Name, Trạng thái `ACCEPTED`/`REJECTED`/`PENDING`, Layer details).

5. **Cluster Fleet Management Tab**:
   - Đổi tên máy tính (Custom Friendly Alias) và lưu cấu hình (`cluster_nodes_config.json`).
   - Bật / Tắt (On/Off) sự tham gia của từng máy tính trong Swarm.
   - Đo độ trễ (Ping / Latency Diagnostics) trực tiếp đến từng nút.
   - Xóa (Delete / Forget) các nút không còn hoạt động.

6. **High-Speed Binary Tensor Transport**:
   - Truyền nhận Tensor dạng binary không làm suy giảm độ chính xác qua Custom TCP sockets, kết hợp mã hóa AES-GCM cho bảo mật cấp biên.

---

## 💻 Hướng dẫn khởi chạy

### Cách 1: Khởi chạy Universal App (Giao diện Web UI)
Chạy ứng dụng Universal App trên bất kỳ máy tính nào (Windows / macOS / Linux):
```bash
python apps/orchest.py
```
Sau đó mở trình duyệt tại: `http://localhost:8080` (hoặc `http://<IP_MAY>:8080`).

### Cách 2: Khởi chạy trên Windows bằng file `.bat`
```cmd
run_win_host.bat
```

---

## 📁 Cấu trúc dự án
- `orchest/executor/`:
  - `host.py`: Host Master Orchestrator, lưu trữ cấu hình fleet `cluster_nodes_config.json`.
  - `worker.py`: Worker Node Process, xử lý thông báo Accept/Reject và ghi Connection History Log.
  - `pipeline.py`: Luồng thực thi Swarm Pipeline phân tán.
- `orchest/partitioner/`:
  - `dynamic_programming.py`: Thuật toán quy hoạch động tối ưu $O(N^2 \cdot K^2)$ phân bổ layer cho $K \ge 1$ máy tính.
- `orchest/network/`:
  - `discovery.py`: UDP Beacon Auto-Discovery.
  - `transport.py`: Zero-overhead Binary Tensor Socket Transport.
- `apps/`:
  - `orchest.py`: Primary Universal App Launcher.
  - `dashboard/`: FastAPI Server và Modern Web UI (Glassmorphism, 3-Tab Navigator, Multi-Stage Pipeline Map).
- `tests/`: Bộ kiểm thử tự động (`pytest tests/`).
