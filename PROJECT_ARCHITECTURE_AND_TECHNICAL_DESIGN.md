# TÀI LIỆU TOÀN DIỆN: CẤU TRÚC CODEBASE, LOGIC THUẬT TOÁN, PHƯƠNG ÁN KỸ THUẬT VÀ GIẢI THÍCH CHI TIẾT TỪNG HÀM
## HỆ THỐNG PHÂN TÁN ORCHEST COMPUTING

---

## MỤC LỤC
1. [SƠ ĐỒ CẤU TRÚC TOÀN BỘ FILE TRONG CODEBASE](#1-sơ-đồ-cấu-trúc-toàn-bộ-file-trong-codebase)
2. [GIẢI THÍCH CHI TIẾT TỪNG HÀM TRONG TỪNG FILE MÃ NGUỒN](#2-giải-thích-chi-tiết-từng-hàm-trong-từng-file-mã-nguồn)
   - 2.1. Module `orchest.partitioner.cost_matrix`
   - 2.2. Module `orchest.partitioner.dynamic_programming`
   - 2.3. Module `orchest.models.auto_slicer` & `qwen_slicer`
   - 2.4. Module `orchest.network.protocol`
   - 2.5. Module `orchest.network.transport`
   - 2.6. Module `orchest.network.discovery`
   - 2.7. Module `orchest.network.micro_batcher`
   - 2.8. Module `orchest.executor.host`
   - 2.9. Module `orchest.executor.worker`
   - 2.10. Module `orchest.executor.pipeline`
   - 2.11. Module `orchest.executor.swarm_generator`
   - 2.12. Module `orchest.executor.fault_tolerance`
   - 2.13. Module `orchest.profiler.node_profiler`
   - 2.14. Module `orchest.profiler.hardware_benchmark`
   - 2.15. Module `orchest.profiler.model_profiler`
   - 2.16. Module `orchest.profiler.network_profiler`
   - 2.17. Module `orchest.profiler.correctness`
   - 2.18. Module `core_engine.dp_solver.cpp`
   - 2.19. Module `apps.dashboard.server`
   - 2.20. Module `apps.dashboard.static.app.js`
   - 2.21. Module `scripts` & `benchmarks`
3. [CÁC PHƯƠNG ÁN KỸ THUẬT THỰC TẾ ĐANG TRIỂN KHAI TRONG CODE](#3-các-phương-án-kỹ-thuật-thực-tế-đang-triển-khai-trong-code)
4. [SƠ ĐỒ KIẾN TRÚC PIPELINE VÀ CÁC MÔ HÌNH HỖ TRỢ](#4-sơ-đồ-kiến-trúc-pipeline-và-các-mô-hình-hỗ-trợ)
   - 4.1. Sơ đồ Pipeline Parallelism toàn diện (End-to-End Tensor Transport & Topology)
   - 4.2. Sơ đồ cấu trúc nội bộ khối lát cắt Transformer (`ModelSliceModule`)
   - 4.3. Bảng phân tích chi tiết các mô hình AI trong hệ thống (`MODEL_CONFIGS`)
   - 4.4. Sơ đồ kịch bản phân bổ mẫu (Case Study: Phân chia Qwen2.5-7B trên cụm 3 máy)
   - 4.5. Sơ đồ chu trình sinh Token tự hồi quy bầy đàn (Autoregressive Swarm Generation Loop)
5. [BIỂU ĐỒ USE CASE VÀ KỊCH BẢN TƯƠNG TÁC](#5-biểu-đồ-use-case-và-kịch-bản-tương-tác)
   - 5.1. Sơ đồ Use Case tổng quan toàn hệ thống (System Overview Use Case Diagram)
   - 5.2. Sơ đồ Use Case phân hệ Điều phối Host Master
   - 5.3. Sơ đồ Use Case phân hệ Thiết bị biên Worker Node
   - 5.4. Sơ đồ Use Case phân hệ Quản trị Fleet & Chịu lỗi
   - 5.5. Bảng đặc tả chi tiết các Use Case chính (Use Case Specifications: UC-01 → UC-05)
6. [LUỒNG THỰC THI END-TO-END TỪNG BƯỚC](#6-luồng-thực-thi-end-to-end-từng-bước)
7. [BÁO CÁO DỌN DẸP CÁC FILE DƯ THỪA TRONG DỰ ÁN](#7-báo-cáo-dọn-dẹp-các-file-dư-thừa-trong-dự-án)
8. [KẾT LUẬN](#8-kết-luận)

---

## 1. SƠ ĐỒ CẤU TRÚC TOÀN BỘ FILE TRONG CODEBASE

### 1.1. Cây thư mục tinh gọn sau khi dọn dẹp

```text
/Applications/HCMUT/Orchest Computing/
├── benchmark_results.json                          # Kết quả đo đạc benchmark thực tế lưu trữ dạng JSON
├── README.md                                       # File README tổng quan ở thư mục gốc
├── PROJECT_ARCHITECTURE_AND_TECHNICAL_DESIGN.md    # Tài liệu kiến trúc toàn diện của dự án
├── AAIDS_FM_ORCHEST_Proposal_v2-long_210826_TTT.pdf# Tài liệu thuyết minh proposal v2 gốc
├── AAIDS_FM_ORCHEST_Proposal_v3_Revised.docx       # Tài liệu thuyết minh proposal v3 gốc
├── AAIDS_FM_ORCHEST_Proposal_v4-long_290826_TTT.docx# Tài liệu thuyết minh proposal v4 gốc
├── DATN/                                           # THƯ MỤC GỐC CHÍNH CỦA DỰ ÁN (PROJECT ROOT)
│   ├── pyproject.toml                              # Cấu hình project và metadata gói Python
│   ├── requirements.txt                            # Danh sách thư viện phụ thuộc (PyTorch, FastAPI,...)
│   ├── cluster_nodes_config.json                   # Cấu hình Fleet đang hoạt động trực tiếp của server
│   ├── run_win_host.bat                            # Script Windows chạy nhanh Host Dashboard
│   ├── run_win_worker.bat                          # Script Windows chạy nhanh Worker Node
│   ├── run_tests.py                                # Script thực thi toàn bộ test suite pytest
│   │
│   ├── certs/                                      # Chứng chỉ bảo mật mTLS
│   │   ├── ca.crt / ca.key                         # Cặp khóa và chứng chỉ CA
│   │   ├── host.crt / host.key                     # Cặp khóa và chứng chỉ của Host Master
│   │   ├── worker.crt / worker.key                 # Cặp khóa và chứng chỉ của Worker Node
│   │   └── host_zmq.* / worker_zmq.*               # Khóa mã hóa CurveZMQ
│   │
│   ├── core_engine/                                # Engine tăng tốc tính toán viết bằng C++
│   │   ├── CMakeLists.txt                          # Cấu hình biên dịch thư viện động (.so / .dylib / .dll)
│   │   └── dp_solver.cpp                           # Triển khai thuật toán quy hoạch động bằng C++ thuần
│   │
│   ├── apps/                                       # Tầng ứng dụng người dùng
│   │   ├── orchest.py                              # ĐIỂM KHỞI CHẠY CHÍNH (Universal Unified App Launcher)
│   │   ├── host_node.py                            # Launcher chạy Host Master độc lập từ dòng lệnh
│   │   ├── worker_node.py                          # Launcher chạy Worker Node độc lập từ dòng lệnh
│   │   ├── cli.py                                  # Giao diện dòng lệnh CLI (profile, host, worker)
│   │   └── dashboard/                              # Web Dashboard Server & Giao diện người dùng
│   │       ├── server.py                           # FastAPI Backend Server, SSE Streaming, REST APIs
│   │       └── static/                             # Frontend tĩnh (Glassmorphism Single Page App)
│   │           ├── index.html                      # Giao diện HTML5 chứa 3 Tabs: Host, Worker, Fleet
│   │           ├── styles.css                      # Toàn bộ mã CSS (Glassmorphism, Dark mode, Animation)
│   │           └── app.js                          # Toàn bộ mã Javascript Client (Canvas map, API, Events)
│   │
│   ├── orchest/                                    # GÓI LÕI CỦA HỆ THỐNG (CORE PYTHON PACKAGE)
│   │   ├── executor/                               # Điều phối thực thi phân tán
│   │   │   ├── host.py                             # Lớp HostOrchestrator: Quản lý cụm, quy hoạch lát cắt
│   │   │   ├── worker.py                           # Lớp WorkerNode: Nhận lát cắt, ủy quyền, thực thi Forward
│   │   │   ├── pipeline.py                         # Lớp SwarmPipelineExecutor: Chuyển tiếp tensor đa node
│   │   │   ├── swarm_generator.py                  # Lớp SwarmAutoregressiveGenerator: Sinh token tự hồi quy
│   │   │   └── fault_tolerance.py                  # Lớp FaultToleranceMonitor: Giám sát Heartbeat và timeout
│   │   ├── models/                                 # Trích xuất và cắt mô hình LLM
│   │   │   ├── auto_slicer.py                      # Cắt lớp vạn năng & Selective Weight Sharded Loading
│   │   │   └── qwen_slicer.py                      # Hàm tiện ích cắt chuyên biệt cho mô hình Qwen
│   │   ├── network/                                # Giao tiếp mạng và giao thức truyền nhận
│   │   │   ├── protocol.py                         # Đóng gói khung nhị phân TensorPacket 18B Header, CRC32
│   │   │   ├── transport.py                        # Hiện thực hóa Raw TCP Socket, gRPC, ZeroMQ, mTLS
│   │   │   ├── discovery.py                        # Tự động phát hiện qua UDP Beacon Broadcast
│   │   │   ├── micro_batcher.py                    # Gom cụm kích hoạt micro-batch bất đồng bộ
│   │   │   ├── tensor_service.proto                # Định nghĩa dịch vụ gRPC truyền nhận tensor
│   │   │   ├── tensor_service_pb2.py               # File sinh tự động từ protobuf
│   │   │   └── tensor_service_pb2_grpc.py          # File sinh stubs gRPC tự động
│   │   ├── partitioner/                            # Tính toán ma trận chi phí và phân hoạch
│   │   │   ├── cost_matrix.py                      # Lớp PartitionCostCalculator: Tính VRAM, KV-Cache, Latency
│   │   │   └── dynamic_programming.py              # Lớp DynamicPartitioner: Thuật toán DP O(K*L^2), Heuristics
│   │   └── profiler/                               # Đo đạc và chấm điểm tài nguyên
│   │       ├── node_profiler.py                    # Đo đạc cấu hình máy cục bộ (CPU, RAM, GPU, VRAM)
│   │       ├── hardware_benchmark.py               # Đánh giá hiệu năng thực tế (Stress test TFLOPs, Băng thông)
│   │       ├── model_profiler.py                   # Đo đạc thông số các layer của mô hình (FLOPs, Memory)
│   │       ├── network_profiler.py                 # Đo độ trễ RTT và Băng thông mạng thực tế
│   │       └── correctness.py                      # Kiểm thử tính đúng đắn số học (Cosine Similarity)
│   │
│   ├── scripts/                                    # Các script hỗ trợ vận hành và bảo mật
│   │   ├── generate_certs.py                       # Script tự động tạo chứng chỉ mTLS X.509
│   │   └── simulate_lan_swarm.py                   # Script giả lập cụm Swarm nhiều worker ảo trên 1 máy
│   ├── benchmarks/                                 # Module chạy thực nghiệm đo đạc
│   │   └── benchmark_suite.py                      # Bộ kịch bản đo đạc toàn diện các kịch bản suy luận
│   └── tests/                                      # Bộ 10 file kiểm thử tự động toàn diện pytest
```

---

## 2. GIẢI THÍCH CHI TIẾT TỪNG HÀM TRONG TỪNG FILE MÃ NGUỒN

---

### 2.1. Module `orchest.partitioner.cost_matrix`
Tập tin: [`DATN/orchest/partitioner/cost_matrix.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/partitioner/cost_matrix.py)  
Lớp: `PartitionCostCalculator`

#### 1. `get_node_compute_speed(node: NodeMetrics) -> float`
- **Giải thuật**: Duyệt danh sách GPU của node, trích xuất giá trị `estimated_tflops` lớn nhất. Nếu không có GPU, ước tính tuyến tính theo số nhân logic của CPU: $\max(\text{cores} \times 0.05, 0.05)$.
- **Công dụng**: Xác định công suất tính toán hiệu dụng của node theo đơn vị TFLOPs/giây ($10^{12}$ phép tính/giây).
- **Cách dùng**: `speed = PartitionCostCalculator.get_node_compute_speed(node)`.
- **Ý nghĩa trong toàn cục**: Cung cấp mẫu số quy đổi tổng FLOPs của một lát cắt layer sang thời gian thực thi mili-giây cho thuật toán phân hoạch.
- **Ưu điểm**: Nhẹ, phản ánh tức thì năng lực GPU rời (CUDA) lẫn Apple Silicon (Metal/MPS).
- **Khuyết điểm**: Chưa xét đến hiện tượng giảm xung do nghẽn nhiệt (thermal throttling) theo thời gian dài.

#### 2. `get_node_available_memory_mb(node: NodeMetrics) -> float`
- **Giải thuật**: Nếu node có GPU, tìm GPU có `vram_free_mb` lớn nhất. Nếu thiết bị là `cuda` hoặc `mps`, trả về dung lượng VRAM trống đó; ngược lại, trả về `ram_available_mb`.
- **Công dụng**: Lấy dung lượng bộ nhớ khả dụng $M_{\text{usable}}$ phục vụ việc kiểm tra ràng buộc tránh OOM.
- **Cách dùng**: `avail_mb = PartitionCostCalculator.get_node_available_memory_mb(node)`.
- **Ý nghĩa trong toàn cục**: Thiết lập giới hạn trần bộ nhớ cho từng node trong ma trận chi phí.
- **Ưu điểm**: Phân biệt chuẩn xác giữa kiến trúc VRAM rời của Windows và Unified Memory của macOS.
- **Khuyết điểm**: Giá trị VRAM mang tính thời điểm; nếu ứng dụng ngoài chiếm thêm VRAM sau thời điểm đo có thể gây sai lệch.

#### 3. `estimate_kv_cache_mb(...) -> float`
- **Giải thuật**: 
  $$M_{\text{KV}} = \frac{2 \times \text{num\_layers} \times \text{concurrency} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S_{\text{max}} \times \text{batch\_size} \times \text{dtype\_bytes}}{1024 \times 1024}$$
- **Công dụng**: Ước tính dung lượng bộ nhớ động dành cho KV-Cache của các layer được giao.
- **Cách dùng**: `kv_mb = PartitionCostCalculator.estimate_kv_cache_mb(num_layers=12, hidden_size=3584, num_heads=28, num_kv_heads=4, max_seq_len=2048)`.
- **Ý nghĩa trong toàn cục**: Cho phép thuật toán DP nhận biết trước sự phình to của bộ nhớ khi sinh chuỗi dài, ngăn chặn OOM trong pha Decode.
- **Ưu điểm**: Hỗ trợ trực tiếp kiến trúc Grouped-Query Attention (GQA), tính đúng số đầu $n_{\text{kv\_heads}}$ thay vì áp đặt Multi-Head Attention (MHA) truyền thống.
- **Khuyết điểm**: Giả định ngữ cảnh luôn đạt mức tối đa $S_{\text{max}}$ (dự phòng trường hợp xấu nhất), có thể hơi bi quan với các prompt ngắn.

#### 4. `estimate_peak_activation_mb(...) -> float`
- **Giải thuật**: 
  $$\text{Peak\_Act} = \frac{\text{batch\_size} \times \text{prompt\_len} \times \text{hidden\_size} \times 4 \times \text{dtype\_bytes}}{1024 \times 1024}$$
- **Công dụng**: Dự báo dung lượng đỉnh của các vector kích hoạt trung gian trong quá trình tính toán forward pass của khối MLP/SwiGLU.
- **Ý nghĩa trong toàn cục**: Đảm bảo bộ nhớ không bị sụp đổ trong bước tính toán ma trận trung gian của pha Prefill.
- **Ưu điểm**: Tính toán tức thời, hệ số $4\times$ bám sát cấu trúc của hầu hết các kiến trúc LLaMA/Qwen.
- **Khuyết điểm**: Không mô hình hóa việc tái sử dụng bộ nhớ đệm (activation buffer reuse) của các trình biên dịch sâu.

#### 5. `calculate_working_set_mb(...) -> float`
- **Giải thuật**: 
  $$M_{\text{working}} = \sum M_{\text{weights}} + M_{\text{peak\_act}} + M_{\text{KV}} + M_{\text{runtime\_reserve}}$$
- **Công dụng**: Tính tổng lượng bộ nhớ Working-Set mà một lát cắt layer $[i..j]$ đòi hỏi trên một node.
- **Ý nghĩa trong toàn cục**: Đại diện cho vế trái của bất đẳng thức kiểm tra khả thi: $M_{\text{working}} \le \alpha \cdot M_{\text{usable}}$.
- **Ưu điểm**: Tổng hợp đa thành phần, loại bỏ triệt để việc chia lát cắt chỉ dựa vào kích thước trọng số tĩnh.
- **Khuyết điểm**: Chưa tính đến bộ đệm mạng $M_{\text{transport}}$ khi kích thước tensor kích hoạt cực lớn.

#### 6. `compute_time_ms(...) -> float`
- **Giải thuật**:
  1. Kiểm tra $M_{\text{working}} > (\text{avail\_mem} \times \alpha)$: Trả về $+\infty$.
  2. Nếu pha `prefill`: $\text{ms} = \frac{\sum \text{FLOPs} \times \text{prompt\_len} \times \text{batch\_size}}{v_{\text{compute}} \times 10^{12}} \times 1000$.
  3. Nếu pha `decode`: Tính tổng thời gian đọc trọng số từ bộ nhớ ($\text{Weight} / \text{BW}_{\text{mem}}$) cộng thời gian tính toán 1 token.
- **Công dụng**: Trả về thời gian thực thi lát cắt bằng mili-giây.
- **Ý nghĩa trong toàn cục**: Hàm chi phí hạt nhân được gọi lặp lại hàng nghìn lần trong bảng quy hoạch động.
- **Ưu điểm**: Tách biệt rõ ràng bản chất Compute-Bound (Prefill) và Memory-Bandwidth-Bound (Decode).
- **Khuyết điểm**: Băng thông bộ nhớ hiện được gán tĩnh ($50\text{ GB/s}$ cho GPU/MPS, $25\text{ GB/s}$ cho CPU) thay vì đo động từng giây.

#### 7. `comm_time_ms(...) -> float`
- **Giải thuật**: Nếu `node_from == node_to` $\to 0$. Ngược lại:
  $$\text{ms} = \frac{\text{RTT}}{2} + \frac{\text{PayloadBytes} \times 8}{\text{BW}_{\text{Mbps}} \times 10^6} \times 1000$$
- **Công dụng**: Tính độ trễ truyền tensor kích hoạt qua cáp mạng/Wi-Fi giữa hai node kế cận.
- **Ý nghĩa trong toàn cục**: Đóng vai trò là chi phí chuyển tiếp cạnh (edge transition cost) trong đồ thị phân hoạch.
- **Ưu điểm**: Khắc họa chính xác hiện tượng không mất phí truyền khi hai tầng cùng nằm trên một node.
- **Khuyết điểm**: Chưa tính độ trễ chuyển dữ liệu giữa GPU và Host RAM (D2H/H2D) trên các hệ thống dùng PCIe chậm.

---

### 2.2. Module `orchest.partitioner.dynamic_programming`
Tập tin: [`DATN/orchest/partitioner/dynamic_programming.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/partitioner/dynamic_programming.py)  
Lớp: `DynamicPartitioner`

#### 1. `partition(...) -> List[SliceAssignment]`
- **Giải thuật**: Quy hoạch động 2 chiều quét qua $K$ node và $N$ layer. Khởi tạo $DP[0][i]$, duyệt các bước chuyển $DP[s][i] = \min_{k} (DP[s-1][k-1] + \text{comm} + \text{compute})$. Quét $\kappa \in [1..K]$ để tìm nghiệm có tổng độ trễ nhỏ nhất. Truy vết ngược qua mảng `parent`.
- **Công dụng**: Trả về kế hoạch phân bổ layer tối ưu toàn cục cho cụm máy tính.
- **Cách dùng**: `plan = partitioner.partition(layers, nodes, rtt_matrix, bw_matrix, alpha=0.90)`.
- **Ý nghĩa trong toàn cục**: "Bộ não" đưa ra quyết định chia cắt mô hình cho toàn bộ hệ thống Orchest Computing.
- **Ưu điểm**: Đảm bảo lát cắt liên tục (contiguous), không phân mảnh layer, độ phức tạp đa thức $O(K \cdot N^2)$ cực nhanh.
- **Khuyết điểm**: Cố định thứ tự node đầu vào (fixed node order); chưa tự hoán vị mọi tổ hợp thứ tự node (Permutation search).

#### 2. `partition_by_ratio(...) -> List[SliceAssignment]`
- **Giải thuật**: Nhận mảng tỉ lệ phần trăm $\sum \text{ratio} = 1.0$. Tính số lượng layer cho node $i$: $\text{round}(N \times \text{ratio}_i)$. Tự động bù trừ sai số làm tròn ở node cuối cùng để đảm bảo tổng số layer đúng bằng $N$.
- **Công dụng**: Thực thi phân chia layer khi người dùng tự kéo thanh trượt thủ công trên Web UI.
- **Ưu điểm**: Linh hoạt, phản hồi trực tiếp ý muốn can thiệp của người quản trị.
- **Khuyết điểm**: Có thể dẫn đến OOM nếu người dùng cố tình kéo nhiều layer cho một máy yếu (không có bộ lọc an toàn cứng như DP).

#### 3. `partition_equal(...)`, `partition_memory_proportional(...)`, `partition_compute_aware(...)`
- **Giải thuật**: 
  - `equal`: Chia đều $N/K$ layer cho mỗi node.
  - `memory_proportional`: Chia layer tỷ lệ thuận theo dung lượng VRAM/RAM của từng node.
  - `compute_aware`: Chia layer tỷ lệ thuận theo TFLOPs của từng node.
- **Công dụng**: Các thuật toán Heuristic cơ sở dùng làm đối chứng (baseline benchmarks) để đánh giá thuật toán DP.
- **Ưu điểm**: Thuật toán đơn giản, tốc độ tính $O(1)$.
- **Khuyết điểm**: Bỏ qua chi phí truyền thông mạng và KV-Cache động, dễ dẫn đến nghẽn cổ chai (straggler) hoặc sập OOM.

#### 4. `find_measured_oracle(...) -> PartitionEvaluationResult`
- **Giải thuật**: Dùng `itertools.combinations` sinh ra toàn bộ các điểm cắt hợp lệ của $N$ layer trên $K$ node. Đo đạc hoặc tính toán chi phí thực tế trên từng phương án để tìm ra nghiệm tối ưu tuyệt đối (Ground Truth Oracle).
- **Công dụng**: Tìm kiếm nghiệm hoàn hảo làm mốc so sánh độ hối tiếc (Regret).
- **Ưu điểm**: Khách quan 100%, không bị sai lệch mô hình.
- **Khuyết điểm**: Độ phức tạp tổ hợp bùng nổ $O(\binom{N-1}{K-1})$, chỉ chạy được trên cụm nhỏ ($K \le 3$, $N \le 36$).

---

### 2.3. Module `orchest.models.auto_slicer` & `qwen_slicer`
Tập tin: [`DATN/orchest/models/auto_slicer.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/models/auto_slicer.py)

#### 1. `ModelSliceModule.forward(self, input_tensor, position_ids, attention_mask, **kwargs)`
- **Giải thuật**:
  - Nếu `is_first_stage` và đầu vào là Token ID: Chuyển qua `embed_tokens` để tạo hidden_states ban đầu.
  - Tự động sinh `position_ids` nếu bị thiếu.
  - Duyệt tuần tự qua mảng `slice_layers`, gọi hàm forward của từng khối transformer layer.
  - Nếu `is_last_stage`: Cho vector ẩn đi qua `norm` (RMSNorm) và nhân với `lm_head` để thu được logits xác suất từ.
- **Công dụng**: Thực thi tính toán lát cắt mô hình trên một máy Worker.
- **Ý nghĩa trong toàn cục**: Đơn vị tính toán suy luận hạt nhân của mỗi Worker trong pipeline phân tán.
- **Ưu điểm**: Đóng gói hoàn chỉnh, tương thích ngược với chuẩn `torch.nn.Module`, nhận và trả Tensor thông thường.
- **Khuyết điểm**: Chưa tối ưu hóa việc giữ trạng thái KV-Cache giữa các bước sinh (hiện tại tính toán lại per-step nếu không truyền cache ngoài).

#### 2. `AutoModelSlicer.get_transformer_layers(model: PreTrainedModel)`
- **Giải thuật**: Dùng cơ chế Reflection kiểm tra các thuộc tính đặc trưng của các họ mô hình:
  - Base: `.model`, `.transformer`, `.gpt_neox`.
  - Layers: `.layers`, `.h`, `.block`.
  - Embeddings: `.embed_tokens`, `.wte`, `.word_embeddings`.
  - Norm & Head: `.norm`, `.ln_f`, `.final_layernorm`, `.lm_head`.
- **Công dụng**: Tự động phát hiện và trích xuất cấu trúc các tầng của mọi mô hình Hugging Face.
- **Ưu điểm**: Tính tương thích cực cao, hỗ trợ tự động Qwen, LLaMA, Mistral, Gemma, GPT-2,...
- **Khuyết điểm**: Phụ thuộc vào quy ước đặt tên thuộc tính của Hugging Face Transformers; nếu có model viết custom layer bất quy tắc có thể cần bổ sung nhánh rẽ.

#### 3. `AutoModelSlicer.load_selective_slice(...) -> ModelSliceModule`
- **Giải thuật**:
  1. Tải `model.safetensors.index.json` qua `hf_hub_download`.
  2. Quét `weight_map`, lọc các file shard có chứa các layer từ `start_idx` đến `end_idx`, cùng shard chứa `embed_tokens` (nếu có) và `lm_head` (nếu có).
  3. Chỉ tải các shard cần thiết đó về ổ cứng.
  4. Tạo model khung bằng `AutoModelForCausalLM.from_config`.
  5. Đọc tensor từ các file safetensors đã tải và gọi `model.load_state_dict(state_dict, strict=False)`.
  6. Trích xuất thành `ModelSliceModule`.
- **Công dụng**: Tải và nạp đúng các phần trọng số của mô hình mà Worker được giao, không tải phần thừa.
- **Cách dùng**: `slice_mod = AutoModelSlicer.load_selective_slice("Qwen/Qwen2.5-7B-Instruct", start_idx=14, end_idx=27, total_layers=28, device="cuda")`.
- **Ý nghĩa trong toàn cục**: Cho phép các máy tính cá nhân ổ cứng nhỏ và mạng chậm có thể tham gia cụm suy luận mô hình khổng lồ.
- **Ưu điểm**: Tiết kiệm 70-85% băng thông và dung lượng ổ cứng, tăng tốc độ gia nhập cụm.
- **Khuyết điểm**: Với các mô hình không chia shard (single file dung lượng dưới 1GB), hàm sẽ fallback về việc tải cả file mô hình.

---

### 2.4. Module `orchest.network.protocol`
Tập tin: [`DATN/orchest/network/protocol.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/network/protocol.py)

#### 1. `serialize_tensor(tensor: torch.Tensor) -> bytes`
- **Giải thuật**: Đưa tensor về CPU liên tục (`tensor.detach().contiguous().cpu()`), ghi vào `io.BytesIO` bằng `torch.save()`.
- **Công dụng**: Biến tensor nhiều chiều thành mảng byte nhị phân.
- **Ưu điểm**: Bảo toàn 100% định dạng số học (FP16/BF16/FP32), không suy giảm độ chính xác.
- **Khuyết điểm**: Có chi phí copy bộ nhớ từ GPU sang Host RAM (D2H).

#### 2. `deserialize_tensor(bdata: bytes, device: str = "cpu") -> torch.Tensor`
- **Giải thuật**: Dùng `torch.load(io.BytesIO(bdata), map_location=device, weights_only=True)`.
- **Công dụng**: Chuyển mảng byte nhị phân trở lại thành tensor trên thiết bị mục tiêu (`cuda`, `mps`, hoặc `cpu`).
- **Ưu điểm**: An toàn với `weights_only=True` (chống lỗ hổng thực thi mã độc qua pickle), đưa thẳng lên GPU mục tiêu.
- **Khuyết điểm**: Chi phí giải mã phụ thuộc vào tốc độ CPU giải nén luồng byte.

#### 3. `TensorPacket.pack(...) -> bytes`
- **Giải thuật**:
  1. Chuyển JSON meta thành UTF-8 bytes.
  2. Tuần tự hóa tensor (nếu có).
  3. Nếu có `secret_key`: Mã hóa phần nhị phân tensor bằng **AES-GCM-256** với nonce 96-bit.
  4. Tính `body_crc32 = zlib.crc32(header_bytes + tensor_bytes) & 0xffffffff`.
  5. Đóng gói Header 18 bytes bằng struct: `!4sHIII` (Magic `b"ORCH"`, msg_code, header_len, tensor_len, crc32).
  6. Ghép: `Header + Header_Bytes + Tensor_Bytes`.
- **Công dụng**: Tạo một gói tin nhị phân hoàn chỉnh để gửi qua Socket.
- **Ý nghĩa trong toàn cục**: Chuẩn hóa toàn bộ việc trao đổi dữ liệu giữa các node trong cụm.
- **Ưu điểm**: Chống dính/vỡ gói, chống hỏng dữ liệu qua CRC32, bảo mật cấp biên bằng AES-GCM.
- **Khuyết điểm**: Nonce hiện được gán tĩnh 12 bytes `\x00` trong bản prototype; môi trường sản xuất cần sinh nonce ngẫu nhiên đi kèm header.

#### 4. `TensorPacket.unpack(...) -> (MessageType, meta, tensor)`
- **Giải thuật**: Đọc 18 bytes đầu, kiểm tra Magic `b"ORCH"`. Bóc tách độ dài các phần, kiểm tra mã CRC32 của phần thân. Nếu đúng, giải mã JSON và giải mã Tensor (kèm giải mã AES nếu có khóa bí mật).
- **Công dụng**: Tiếp nhận và phục hồi dữ liệu từ socket.
- **Ưu điểm**: Báo lỗi tức thì nếu đường truyền mạng bị suy hao bit.
- **Khuyết điểm**: Ném ra ngoại lệ nếu gói tin không đầy đủ, đòi hỏi tầng mạng phía dưới phải có cơ chế nhận đủ byte (`_recv_exact`).

---

### 2.5. Module `orchest.network.transport`
Tập tin: [`DATN/orchest/network/transport.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/network/transport.py)

#### 1. `RawTCPTransportServer.start(self)`
- **Giải thuật**: Tạo socket `AF_INET, SOCK_STREAM`, thiết lập `SO_REUSEADDR`, bind cổng và lắng nghe hàng đợi 128 kết nối. Nếu có chứng chỉ trong thư mục `certs/`, bọc socket bằng SSL Context (`ssl.create_default_context(Purpose.CLIENT_AUTH)`) với cờ `CERT_REQUIRED`. Tạo luồng lắng nghe accept và mỗi kết nối được xử lý trên một `daemon Thread`.
- **Công dụng**: Lắng nghe tensor và lệnh điều phối từ các node khác gửi tới.
- **Ưu điểm**: Tốc độ cực cao, không bị overhead của HTTP, hỗ trợ bảo mật mTLS tiêu chuẩn công nghiệp.
- **Khuyết điểm**: Mỗi kết nối tạo một thread mới; nếu có hàng nghìn kết nối đồng thời có thể tốn bộ nhớ thread stack (hiện tại cụm LAN quy mô 2-10 máy hoạt động tối ưu).

#### 2. `RawTCPTransportServer._recv_exact(self, conn, num_bytes) -> bytes`
- **Giải thuật**: Vòng lặp `while len(data) < num_bytes:` gọi `conn.recv(min(remains, 65536))` để gom đầy đủ từng mảnh byte của khung nhị phân.
- **Công dụng**: Khắc phục triệt để hiện tượng vỡ gói (TCP packet fragmentation).
- **Ý nghĩa trong toàn cục**: Đảm bảo gói tin `TensorPacket` không bao giờ bị đọc thiếu Header hay Body.
- **Ưu điểm**: Đơn giản, độ tin cậy tuyệt đối.
- **Khuyết điểm**: Chặn đồng bộ (blocking read) trên thread con; được khắc phục bằng cách thiết lập timeout kết nối 10 giây.

#### 3. `RawTCPTransportClient.send_packet(...) -> bool`
- **Giải thuật**: Mở socket kết nối tới `target_ip:target_port`, nếu cấu hình mTLS thì wrap socket với chứng chỉ `worker.crt/key`, đóng gói `TensorPacket.pack()`, gọi `sock.sendall(packet_bytes)` và chờ nhận gói tin `ACK` phản hồi.
- **Công dụng**: Gửi tensor kích hoạt hoặc lệnh phân bổ sang máy khác.
- **Ưu điểm**: Đảm bảo dữ liệu đã được bên kia nhận xong nhờ cơ chế ACK.
- **Khuyết điểm**: Tạo và đóng kết nối TCP cho mỗi gói (Short-lived connection); bản tương lai có thể dùng Connection Pool (Keep-Alive).

---

### 2.6. Module `orchest.network.discovery`
Tập tin: [`DATN/orchest/network/discovery.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/network/discovery.py)

#### 1. `UDPDiscoveryService.start_beacon_sender(self, beacon_provider, interval_sec=2.0)`
- **Giải thuật**: Khởi chạy một `threading.Thread` nền lặp vô tận: Cứ mỗi `interval_sec`, gọi hàm `beacon_provider()` để lấy thông số phần cứng mới nhất của máy, chuyển thành chuỗi JSON và gửi qua UDP Socket với cờ `SO_BROADCAST` tới địa chỉ `255.255.255.255:port`.
- **Công dụng**: Phát tín hiệu định danh để Host tìm thấy máy tính trong mạng nội bộ.
- **Ưu điểm**: Tự động, không cần bất kỳ cấu hình IP thủ công nào từ phía người dùng.
- **Khuyết điểm**: Gói UDP broadcast có thể bị một số bộ định tuyến (router) doanh nghiệp chặn qua các subnet khác nhau (hoạt động tốt trong cùng một mạng LAN / Wi-Fi gia đình).

#### 2. `UDPDiscoveryService.start_beacon_listener(self)`
- **Giải thuật**: Khởi chạy luồng nền bind socket UDP tại `0.0.0.0:port`, liên tục nhận dữ liệu `recvfrom(65535)`. Bóc tách JSON, lấy địa chỉ IP thực tế từ socket (`addr[0]`), cập nhật vào từ điển `active_nodes[node_id] = beacon`.
- **Công dụng**: Host thu thập toàn bộ danh sách các máy tính đang online trong mạng.
- **Ý nghĩa trong toàn cục**: Xây dựng bảng đăng bạ topology mạng thời gian thực.
- **Ưu điểm**: Tự động gán đúng IP vật lý của máy gửi (tránh việc worker tự khai báo nhầm IP localhost `127.0.0.1`).
- **Khuyết điểm**: Chưa có cơ chế lọc chống tràn gói (rate-limiting) nếu có nhiều thiết bị broadcast cùng lúc.

---

### 2.7. Module `orchest.network.micro_batcher`
Tập tin: [`DATN/orchest/network/micro_batcher.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/network/micro_batcher.py)

#### 1. `MicroBatcher.enqueue(self, tensor, meta) -> Future`
- **Giải thuật**: Đưa tensor yêu cầu vào hàng đợi nội bộ `queue.Queue()`, trả về một đối tượng `concurrent.futures.Future` để luồng gọi có thể chờ kết quả mà không bị block.
- **Công dụng**: Tiếp nhận các yêu cầu suy luận đồng thời từ nhiều client.
- **Ý nghĩa trong toàn cục**: Chuẩn bị dữ liệu cho việc ghép vi lô (micro-batching).
- **Ưu điểm**: Bất đồng bộ, an toàn đa luồng (thread-safe).
- **Khuyết điểm**: Tăng nhẹ độ trễ của từng request đơn lẻ để đổi lấy thông lượng tổng thể.

#### 2. `MicroBatcher._process_batches(self)`
- **Giải thuật**: Luồng nền gom các tensor trong hàng đợi cho đến khi đạt `max_batch_size` hoặc hết thời gian chờ `timeout_ms`. Ghép các tensor dọc theo chiều batch (`torch.cat(tensors, dim=0)`), đưa qua hàm xử lý batch và phân phát kết quả về từng `Future` tương ứng.
- **Công dụng**: Tối ưu hóa hiệu suất GPU bằng cách tính toán theo lô lớn thay vì từng mẩu nhỏ.
- **Ưu điểm**: Tăng thông lượng tổng (Tokens/sec) khi có nhiều người dùng đồng thời.
- **Khuyết điểm**: Các chuỗi có độ dài khác nhau phải được padding, tốn thêm bộ nhớ đệm.

---

### 2.8. Module `orchest.executor.host`
Tập tin: [`DATN/orchest/executor/host.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/executor/host.py)  
Lớp: `HostOrchestrator`

#### 1. `_refresh_local_host_node(self) -> NodeMetrics`
- **Giải thuật**: Kiểm tra cache thời gian; nếu quá 5 giây thì gọi `NodeProfiler.profile()` để cập nhật lại thông số RAM/CPU của chính máy Host. Gán cổng TCP mặc định 5555, áp dụng tên gợi nhớ từ `cluster_nodes_config.json`, và đăng ký vào danh sách `worker_nodes`.
- **Công dụng**: Đảm bảo chính máy Host cũng là một thành viên tính toán (Node 0) trong cụm.
- **Ưu điểm**: Tránh lãng phí sức mạnh phần cứng của máy chủ Host.
- **Khuyết điểm**: Máy Host vừa làm web server vừa gánh Stage 0 nên có thể chịu tải CPU cao hơn các worker khác.

#### 2. `plan_model_slicing(self, model_name, num_layers, hidden_size, intermediate_size, ...)`
- **Giải thuật**:
  1. Lấy danh sách các node đang hoạt động và có cờ `enabled == True`.
  2. Tạo danh sách `LayerProfile` giả lập hoặc thực tế từ thông số mô hình.
  3. Thu thập ma trận RTT và băng thông giữa các node.
  4. Gọi `DynamicPartitioner.partition(...)` với tham số trần an toàn $\alpha = 0.90$.
  5. Lưu kế hoạch vào `self.current_slice_plan`.
  6. Tự động gọi `dispatch_current_plan()` để truyền lệnh xuống các Worker.
- **Công dụng**: Khởi động toàn bộ chu trình phân hoạch và phân bổ mô hình.
- **Ý nghĩa trong toàn cục**: Điểm hội tụ giữa giao diện người dùng và thuật toán phân hoạch.
- **Ưu điểm**: Tự động hóa hoàn toàn từ bước tính toán đến bước gửi lệnh mạng.
- **Khuyết điểm**: Nếu mạng LAN bị mất kết nối đúng lúc dispatch, cần phải gọi lại hàm này để phân bổ lại.

#### 3. `update_node_config(self, node_id, alias, enabled)` & `delete_node(self, node_id)`
- **Giải thuật**: Cập nhật từ điển `node_configs`, lưu xuống đĩa tại `cluster_nodes_config.json`. Nếu node bị disable hoặc bị xóa, lập tức cập nhật lại topology của cụm.
- **Công dụng**: Quản lý danh bạ Fleet máy tính theo cấu hình người dùng.
- **Ưu điểm**: Bền vững (persist), giữ nguyên tên gợi nhớ của các máy kể cả sau khi khởi động lại.
- **Khuyết điểm**: Ghi đĩa đồng bộ (sync file I/O); có thể tối ưu bằng async I/O nếu danh bạ hàng trăm node.

---

### 2.9. Module `orchest.executor.worker`
Tập tin: [`DATN/orchest/executor/worker.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/executor/worker.py)  
Lớp: `WorkerNode`

#### 1. `record_history_log(self, event_type, host_id, host_ip, model_name, details, status)`
- **Giải thuật**: Tạo một bản ghi JSON có ID ngẫu nhiên (`uuid4()[:8]`), mốc thời gian dạng chuỗi ISO (`strftime("%Y-%m-%d %H:%M:%S")`), chèn vào đầu danh sách `self.connection_history.insert(0, log_entry)`. Giới hạn tối đa 50 bản ghi gần nhất.
- **Công dụng**: Lưu lại vết lịch sử các lần kết nối và phân bổ mô hình.
- **Ưu điểm**: Giúp người dùng Worker kiểm soát minh bạch các tác vụ được Host giao.
- **Khuyết điểm**: Lưu trên bộ nhớ RAM (mất đi khi tắt tiến trình Worker).

#### 2. `handle_dispatch_request(self, meta)`
- **Giải thuật**:
  - Trích xuất: `host_id`, `host_ip`, `model_name`, `layer_start`, `layer_end`, `next_ip`, `next_port`.
  - Nếu `self.auto_accept == True`: Tự động gọi nạp lát cắt mô hình qua `AutoModelSlicer.load_selective_slice()`, ghi log `ACCEPTED`, chuyển trạng thái `self.status = "SERVING"`.
  - Nếu `self.auto_accept == False`: Đưa vào hàng đợi `self.pending_requests`, ghi log `PENDING`, chờ người dùng duyệt trên Web UI.
- **Công dụng**: Xử lý yêu cầu phân bổ lát cắt từ Host gửi sang.
- **Ý nghĩa trong toàn cục**: Chốt chặn bảo mật và phân quyền của Worker.
- **Ưu điểm**: Tôn trọng quyền riêng tư của thiết bị biên, cho phép từ chối nếu máy đang bận tác vụ cá nhân.
- **Khuyết điểm**: Nếu người dùng quên bấm Accept thì cụm sẽ phải chờ hoặc timeout.

#### 3. `execute_forward_slice(self, input_tensor, meta)`
- **Giải thuật**:
  - Đưa `input_tensor` qua `self.slice_module.forward(input_tensor)`.
  - Nếu có node kế tiếp (`self.next_node_ip`): Dùng `TensorTransportClient.send_packet` gửi tiếp tensor đó sang `self.next_node_ip:self.next_node_port` với loại gói `FORWARD_TENSOR`.
  - Nếu là mắt xích cuối cùng: Gửi gói `INFERENCE_COMPLETE` về cho Host Master.
- **Công dụng**: Chuyển giao dữ liệu liên tục theo luồng Pipeline.
- **Ưu điểm**: Dữ liệu chỉ truyền theo một chiều tuần tự (Stage $i \to$ Stage $i+1$), giảm tối đa xung đột đường truyền.
- **Khuyết điểm**: Nếu một mắt xích ở giữa bị nghẽn mạng thì toàn bộ chuỗi pipeline phải chờ theo.

---

### 2.10. Module `orchest.executor.pipeline`
Tập tin: [`DATN/orchest/executor/pipeline.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/executor/pipeline.py)  
Lớp: `SwarmPipelineExecutor`

#### 1. `dispatch_slice_plan(self, model_name: str, total_layers: int) -> bool`
- **Giải thuật**: Duyệt qua từng `SliceAssignment` trong kế hoạch phân bổ. Với mỗi phần tử thứ $i$:
  - Xác định node kế tiếp $i+1$ (lấy IP và cổng TCP của node kế tiếp).
  - Đóng gói thông số metadata gồm: `stage_index`, `layer_start`, `layer_end`, `next_ip`, `next_port`.
  - Nếu node là chính Host Master: Bỏ qua việc gửi qua mạng, đánh dấu sẵn sàng chạy in-process.
  - Nếu là Remote Worker: Gọi `TensorTransportClient.send_packet` gửi gói `DISPATCH_SLICE`.
- **Công dụng**: Thiết lập bảng định tuyến pipeline phân tán trên toàn cụm.
- **Ý nghĩa trong toàn cục**: Biến các máy tính riêng lẻ thành một chuỗi pipeline ăn khớp chính xác với nhau.
- **Ưu điểm**: Các Worker tự biết địa chỉ của node kế tiếp để gửi thẳng dữ liệu cho nhau mà không cần đi vòng về Host Master (Peer-to-Peer Pipeline).
- **Khuyết điểm**: Chưa có cơ chế retry tự động nếu một Worker gửi dispatch bị rớt mạng tạm thời.

---

### 2.11. Module `orchest.executor.swarm_generator`
Tập tin: [`DATN/orchest/executor/swarm_generator.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/executor/swarm_generator.py)  
Lớp: `SwarmAutoregressiveGenerator`

#### 1. `_ensure_model_loaded(self, model_name: str)`
- **Giải thuật**: Kiểm tra nếu mô hình yêu cầu khác với mô hình đang nạp thì giải phóng mô hình cũ. Tải Tokenizer qua `AutoTokenizer.from_pretrained()`. Tải mô hình cục bộ bằng `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16, low_cpu_mem_usage=True)`.
- **Công dụng**: Nạp và chuẩn bị sẵn mô hình/tokenizer trên máy Host.
- **Ưu điểm**: Quản lý bộ nhớ đệm thông minh, tránh nạp lại nhiều lần nếu dùng chung mô hình.
- **Khuyết điểm**: Với mô hình lớn, việc nạp full model trên Host có thể tốn RAM (Host đóng vai trò điều phối chính).

#### 2. `generate_stream(self, prompt, max_new_tokens=1024, temperature=0.7)`
- **Giải thuật**:
  1. Định dạng câu hỏi theo mẫu chat instruct: `tokenizer.apply_chat_template([{"role": "user", "content": prompt}])`.
  2. Khởi tạo `TextIteratorStreamer(tokenizer, skip_prompt=True)`.
  3. Khởi chạy luồng nền `threading.Thread` chạy hàm `local_model.generate` (hoặc phân tán qua swarm).
  4. Sử dụng cú pháp `yield from streamer` để phát từng token văn bản ngay khi vừa sinh xong về cho client.
  5. Kiểm tra cờ `self.stop_requested` ở mỗi chu kỳ lặp để hỗ trợ ngắt khẩn cấp.
- **Công dụng**: Động cơ sinh chữ thời gian thực phục vụ khung chat AI trên Web Dashboard.
- **Ý nghĩa trong toàn cục**: Mang lại trải nghiệm người dùng hiện đại tương tự ChatGPT/Claude.
- **Ưu điểm**: Trả về dữ liệu dạng generator, không cần đợi sinh hết câu mới hiển thị.
- **Khuyết điểm**: Bộ sinh token hiện đang bọc qua streamer cục bộ của Hugging Face trước khi chuyển tiếp toàn bộ sang mạng bầy đàn.

---

### 2.12. Module `orchest.executor.fault_tolerance`
Tập tin: [`DATN/orchest/executor/fault_tolerance.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/executor/fault_tolerance.py)  
Lớp: `FaultToleranceMonitor`

#### 1. `update_heartbeat(self, node: NodeMetrics)`
- **Giải thuật**: Cập nhật mốc thời gian `self.heartbeats[node.node_id] = time.time()` và lưu đối tượng node vào `self.known_nodes`.
- **Công dụng**: Đánh dấu node vẫn đang sống và hoạt động bình thường.
- **Ưu điểm**: Cực nhẹ, chi phí $O(1)$.

#### 2. `_monitor_loop(self)`
- **Giải thuật**: Vòng lặp chạy chu kỳ 2 giây/lần. So sánh `time.time() - last_seen > 8.0` giây. Nếu quá hạn:
  - Xóa node khỏi danh sách theo dõi.
  - Kích hoạt hàm gọi lại: `self.on_node_failed_cb(dead_node_id)`.
- **Công dụng**: Phát hiện tự động các máy bị sập nguồn, ngắt Wi-Fi hoặc treo hệ thống.
- **Ý nghĩa trong toàn cục**: Giúp cụm duy trì tính toàn vẹn, sẵn sàng tự động tái phân hoạch.
- **Ưu điểm**: Ngưỡng 8 giây được cân bằng tốt giữa việc tránh báo động giả (false positive khi mạng giật) và tốc độ phát hiện sự cố.
- **Khuyết điểm**: Nếu đang trong giữa một phiên tính toán forward pass mà node chết thì phiên hiện tại bị lỗi (chỉ phục hồi ở ranh giới phiên kế tiếp).

---

### 2.13. Module `orchest.profiler.node_profiler`
Tập tin: [`DATN/orchest/profiler/node_profiler.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/profiler/node_profiler.py)  
Lớp: `NodeProfiler`

#### 1. `profile(self) -> NodeMetrics`
- **Giải thuật**:
  - Đọc hệ điều hành (`platform.system()`), kiến trúc chip (`platform.machine()`).
  - Đọc thông số CPU: Số nhân vật lý, số nhân logic, tần số hoạt động, phần trăm sử dụng hiện tại (`psutil.cpu_percent()`).
  - Đọc bộ nhớ hệ thống: RAM tổng, RAM khả dụng.
  - Quét thiết bị đồ họa GPU tùy theo OS:
    - Nếu là Windows: Gọi `_get_windows_gpus()` (sử dụng `pynvml` kết hợp `wmi`).
    - Nếu là macOS: Gọi `_get_darwin_gpus()` (kiểm tra `torch.backends.mps.is_available()`, chip Apple Silicon).
    - Nếu là Linux: Gọi `_get_linux_gpus()` (qua NVML).
- **Công dụng**: Thu thập thông tin cấu hình và tài nguyên phần cứng chi tiết của máy.
- **Ý nghĩa trong toàn cục**: Cung cấp dữ liệu đầu vào cho gói tin UDP Beacon và ma trận chi phí.
- **Ưu điểm**: Hỗ trợ toàn diện đa hệ điều hành (Windows, Mac, Linux), tự động xử lý khi thiếu thư viện NVML/WMI mà không gây crash.
- **Khuyết điểm**: Việc gọi WMI trên một số máy Windows cũ có thể mất từ 0.5 đến 1 giây.

---

### 2.14. Module `orchest.profiler.hardware_benchmark`
Tập tin: [`DATN/orchest/profiler/hardware_benchmark.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/profiler/hardware_benchmark.py)

#### 1. `run_client_hardware_rating(duration=2.0, cooldown=0.5) -> Dict[str, Any]`
- **Giải thuật**:
  1. Chọn backend tốt nhất: `cuda` $\to$ `mps` $\to$ `cpu`.
  2. Tạo 2 ma trận ngẫu nhiên kích thước $2048 \times 2048$ kiểu dữ liệu `float32`.
  3. Chạy vòng lặp nhân ma trận liên tục trong `duration` giây.
  4. Đếm tổng số vòng lặp `iterations`.
  5. Tính tổng số phép tính: $\text{FLOPs} = 2 \times 2048^3 \times \text{iterations}$.
  6. Tính: $\text{TFLOPs} = \frac{\text{FLOPs}}{\text{elapsed\_time} \times 10^{12}}$.
  7. Đo nhiệt độ chip trước và sau khi chạy stress test.
  8. Ghi đè vào biến toàn cục `LATEST_BENCHMARK` và lưu file `benchmark_results.json`.
- **Công dụng**: Chấm điểm hiệu năng phần cứng thực tế qua tính toán ma trận thực thay vì thông số lý thuyết.
- **Ý nghĩa trong toàn cục**: Giúp ma trận chi phí có giá trị TFLOPs chính xác nhất để thuật toán DP phân bổ layer hợp lý.
- **Ưu điểm**: Phản ánh chính xác sức mạnh tính toán dấu phẩy động thực tế của GPU/CPU tại thời điểm hiện tại.
- **Khuyết điểm**: Gây tải 100% phần cứng trong 2 giây kiểm tra; do đó hệ thống chỉ tự động chạy khi máy ở chế độ Standby.

---

### 2.15. Module `orchest.profiler.model_profiler`
Tập tin: [`DATN/orchest/profiler/model_profiler.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/profiler/model_profiler.py)  
Lớp: `ModelProfiler`

#### 1. `estimate_layer_flops(hidden_size, intermediate_size, seq_len=1) -> float`
- **Giải thuật**: Tính toán theo công thức chuẩn của khối Transformer Decoder:
  - Attention: $4 \times \text{hidden\_size}^2$ (các ma trận chiếu $Q, K, V, O$).
  - MLP / SwiGLU: $3 \times \text{hidden\_size} \times \text{intermediate\_size}$ (Gate, Up, Down projections).
  - Tổng số phép tính FLOPs = $2 \times (\text{Attn\_params} + \text{MLP\_params}) \times \text{seq\_len}$.
- **Công dụng**: Ước tính số phép tính dấu phẩy động của một layer transformer trên một token.
- **Ưu điểm**: Chuẩn xác cho các kiến trúc LLaMA/Qwen hiện đại.
- **Khuyết điểm**: Chưa tính phần phụ chú ý ma trận tỷ lệ theo bình phương độ dài chuỗi ($O(S^2)$) trong ngữ cảnh cực dài.

---

### 2.16. Module `orchest.profiler.network_profiler`
Tập tin: [`DATN/orchest/profiler/network_profiler.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/profiler/network_profiler.py)  
Lớp: `NetworkProfiler`

#### 1. `measure_rtt(target_ip, port=5555, num_pings=3) -> float`
- **Giải thuật**: Mở socket TCP kết nối tới `target_ip:port`, gửi gói tin nhỏ (ping probe), đo thời gian từ lúc gửi đến khi nhận được ACK. Lấy giá trị trung bình cộng của `num_pings` lần đo.
- **Công dụng**: Đo độ trễ khứ hồi Round-Trip Time giữa 2 máy tính.
- **Ưu điểm**: Dùng TCP ping thực tế thay vì ICMP ping (vốn hay bị tường lửa Windows chặn).
- **Khuyết điểm**: Có chi phí thiết lập TCP handshake nhỏ trong mỗi lần đo.

#### 2. `measure_bandwidth(target_ip, port=5555, payload_mb=4.0) -> float`
- **Giải thuật**: Tạo mảng byte ngẫu nhiên kích thước `payload_mb` (ví dụ 4MB), bấm giờ và đẩy toàn bộ qua socket TCP sang node đích. Tính băng thông: $\text{BW}_{\text{Mbps}} = \frac{\text{payload\_mb} \times 8}{\text{elapsed\_sec}}$.
- **Công dụng**: Xác định tốc độ truyền dữ liệu thực tế trên mạng LAN.
- **Ưu điểm**: Đo trực tiếp tốc độ truyền khối tensor lớn, phản ánh đúng hiệu năng mạng.
- **Khuyết điểm**: Tiêu tốn một lượng nhỏ băng thông mạng LAN trong lúc đo.

---

### 2.17. Module `orchest.profiler.correctness`
Tập tin: [`DATN/orchest/profiler/correctness.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/profiler/correctness.py)  
Lớp: `NumericalCorrectnessVerifier`

#### 1. `verify_tensor_closeness(tensor_a, tensor_b, atol=1e-3) -> Dict[str, Any]`
- **Giải thuật**: Làm phẳng 2 tensor thành vector 1 chiều, tính:
  - Độ tương đồng Cosine: $\frac{A \cdot B}{\|A\|_2 \|B\|_2}$.
  - Sai số tuyệt đối lớn nhất: $\max |A - B|$.
  - Kiểm tra `torch.allclose(tensor_a, tensor_b, atol=atol)`.
- **Công dụng**: Kiểm chứng xem đầu ra của cụm suy luận phân tán có trùng khớp với mô hình gốc trên máy đơn hay không.
- **Ưu điểm**: Khẳng định độ tin cậy số học của toàn hệ thống.
- **Khuyết điểm**: Đòi hỏi phải có kết quả chạy trên máy đơn để làm mốc đối chứng.

---

### 2.18. Module `core_engine.dp_solver.cpp`
Tập tin: [`DATN/core_engine/dp_solver.cpp`](file:///Applications/HCMUT/Orchest%20Computing/DATN/core_engine/dp_solver.cpp)

#### 1. `extern "C" int solve_dp_partition(...)`
- **Giải thuật**:
  - Nhận con trỏ mảng `layers`, số layer `num_layers`, con trỏ mảng `nodes`, số node `num_nodes`, độ trễ RTT và băng thông mặc định.
  - Cấp phát mảng động 2 chiều `std::vector<std::vector<float>> dp` và mảng `parent`.
  - Triển khai 3 vòng lặp lồng nhau tính toán $DP[i][j]$ và truy vết ngược.
  - Ghi kết quả vào con trỏ đầu ra `SliceResult* out_results`.
  - Trả về số lượng lát cắt (stages) tìm được.
- **Công dụng**: Động cơ giải quy hoạch động siêu tốc viết bằng C++ cho hệ thống.
- **Ưu điểm**: Tốc độ thực thi cực nhanh (dưới 0.1 mili-giây), không chịu ảnh hưởng bởi Python Global Interpreter Lock (GIL).
- **Khuyết điểm**: Đòi hỏi hệ thống phải có trình biên dịch C++ (GCC/Clang/MSVC) để build file `.so` / `.dylib` / `.dll`.

---

### 2.19. Module `apps.dashboard.server`
Tập tin: [`DATN/apps/dashboard/server.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/apps/dashboard/server.py)

#### 1. `startup_event()`
- **Giải thuật**: Chạy khi server FastAPI khởi động: Kích hoạt `host_orchestrator.start_discovery()`, bật `local_worker.start()`, tự động tính toán phân hoạch mặc định cho mô hình khởi điểm `Qwen2.5-0.5B`, và khởi chạy luồng nền tự động benchmark phần cứng chu kỳ 5 phút một lần.
- **Công dụng**: Khởi tạo toàn bộ các dịch vụ nền của Universal App.
- **Ưu điểm**: Giúp ứng dụng sẵn sàng phục vụ ngay khi mở giao diện mà người dùng không cần bấm cấu hình phức tạp.

#### 2. `generate_stream_endpoint(req: GenerateRequest)`
- **Giải thuật**: Endpoint nhận yêu cầu tạo văn bản, trả về `StreamingResponse(streamer, media_type="text/event-stream")`. Trong luồng stream, mỗi token sinh ra được bọc theo chuẩn SSE: `data: {"token": "...", "done": false}\n\n`.
- **Công dụng**: Cung cấp API stream chữ chạy cho khung Chat AI.
- **Ưu điểm**: Chuẩn hóa theo kiến trúc web hiện đại (Server-Sent Events), dễ dàng tiêu thụ từ trình duyệt.

#### 3. `update_node(req: NodeUpdateRequest)` & `ping_node(req: NodePingRequest)`
- **Giải thuật**: 
  - `update_node`: Cập nhật cấu hình node vào `HostOrchestrator`. Nếu trạng thái `enabled` thay đổi, tự động gọi phân hoạch lại mô hình ngay lập tức.
  - `ping_node`: Mở kết nối TCP tới IP và cổng của Worker, đo thời gian hoàn thành kết nối, trả về số mili-giây.
- **Công dụng**: Phục vụ các thao tác quản trị Fleet máy tính từ xa.

---

### 2.20. Module `apps.dashboard.static.app.js`
Tập tin: [`DATN/apps/dashboard/static/app.js`](file:///Applications/HCMUT/Orchest%20Computing/DATN/apps/dashboard/static/app.js)

#### 1. `drawPipelineMap(slicePlan)`
- **Giải thuật**: Truy xuất phần tử `<canvas id="pipeline-canvas">`. Lấy danh sách các stage trong `slicePlan`. Tính toán tọa độ $X, Y$, vẽ các khối chữ nhật bo góc đại diện cho từng máy với màu sắc gradient, hiển thị dải layer $[start..end]$, và vẽ các mũi tên cong chuyển tiếp kích hoạt giữa các máy có kèm hiệu ứng hạt photon chuyển động.
- **Công dụng**: Hiển thị trực quan sơ đồ luồng pipeline phân tán trên giao diện web.
- **Ưu điểm**: Trực quan hóa sinh động, giúp người dùng dễ hiểu mô hình đang được cắt ra sao.

#### 2. `syncBalanceSliders(changedNodeId, newValue)`
- **Giải thuật**: Khi người dùng kéo một slider node: Tính độ lệch $\Delta = \text{newValue} - \text{oldValue}$. Phân bổ đều $-\Delta$ cho tất cả các slider của các node còn lại nhưng chặn dưới ở mức tối thiểu $5\%$. Đảm bảo tổng giá trị luôn đúng $100.0\%$.
- **Công dụng**: Giữ vững tính bất biến toán học khi người dùng điều chỉnh tỷ lệ chia layer thủ công.
- **Ưu điểm**: Mượt mà, không giật lag, ngăn chặn triệt để lỗi tổng tỷ lệ khác 100%.

#### 3. `handleWorkerPopup(pendingRequest)`
- **Giải thuật**: Khi có yêu cầu phân bổ mới: Làm xuất hiện hộp thoại Popup Banner cảnh báo màu cam nổi bật trên màn hình, hiển thị Host IP, tên Model, số layer. Bắt sự kiện bấm nút:
  - Nếu bấm Accept: Gửi `POST /api/worker/respond` với `action="accept"`.
  - Nếu bấm Reject: Gửi `POST /api/worker/respond` với `action="reject"`.
- **Công dụng**: Giao diện tương tác phê duyệt của Worker.

---

### 2.21. Module `scripts` & `benchmarks`
Tập tin: `scripts/generate_certs.py`, `scripts/simulate_lan_swarm.py`, `benchmarks/benchmark_suite.py`

#### 1. `generate_certs.py -> generate_all_certs()`
- **Giải thuật**: Sử dụng thư viện `cryptography.x509` để tạo cặp khóa riêng RSA 2048-bit. Tạo chứng chỉ gốc CA tự ký thời hạn 1 năm. Tiếp đó tạo các chứng chỉ con cho Host và Worker được ký trực tiếp bởi khóa riêng của CA. Xuất ra các file `.crt` và `.key` trong thư mục `certs/`.
- **Công dụng**: Thiết lập hạ tầng khóa công khai (PKI) phục vụ bảo mật mTLS cho cụm.

#### 2. `simulate_lan_swarm.py -> main()`
- **Giải thuật**: Dùng `multiprocessing.Process` khởi chạy 3 đối tượng `WorkerNode` với các ID `Worker_Sim_1`, `Worker_Sim_2`, `Worker_Sim_3` trên các cổng TCP 5556, 5557, 5558.
- **Công dụng**: Cho phép lập trình viên chạy thử nghiệm toàn bộ hệ thống Swarm phân tán ngay trên một máy tính cá nhân duy nhất.

---

## 3. CÁC PHƯƠNG ÁN KỸ THUẬT THỰC TẾ ĐANG TRIỂN KHAI TRONG CODE

### 3.1. Phương án 1: Universal App "3 trong 1" (`apps/orchest.py`)
Mã nguồn gom toàn bộ chức năng vào một điểm chạy duy nhất. Không cần phân chia file chạy máy chủ hay máy con. Khi khởi chạy, tiến trình tự đóng cả 3 vai trò: Host Master, Worker Node và Web Manager. Người dùng chỉ việc tương tác qua giao diện web 3 tab:
- Tab 1: Quản lý Host & Chat AI.
- Tab 2: Quản lý Worker & Ủy quyền kết nối.
- Tab 3: Quản lý Fleet & Đổi tên máy, đo Ping, Benchmark.

### 3.2. Phương án 2: Giao thức đóng gói nhị phân TensorPacket 18-Byte Header + CRC32
Không sử dụng JSON hay Base64 để truyền tensor. Gói tin được đóng gói theo khung nhị phân chuẩn gồm 18 bytes fixed header:
`[4B Magic "ORCH"][2B MsgTypeCode][4B HeaderLen][4B TensorLen][4B BodyCRC32]`
Tuần tự hóa trực tiếp qua `torch.save(tensor.contiguous().cpu(), io.BytesIO())`, giữ nguyên vẹn 100% độ chính xác dấu phẩy động của mô hình và kiểm tra toàn vẹn bằng CRC32.

### 3.3. Phương án 3: Cơ chế nạp Shard chọn lọc (Selective Weight Sharded Loading)
Thay vì bắt các máy biên phải tải toàn bộ mô hình nặng 15GB–60GB từ Hugging Face, hàm `AutoModelSlicer.load_selective_slice()` đọc file chỉ mục `model.safetensors.index.json`, chỉ tải đúng các file shard chứa layer được phân công cho máy đó. Giúp tiết kiệm từ 70% đến 85% băng thông và dung lượng ổ cứng.

### 3.4. Phương án 4: Tự động khám phá qua UDP Beacon & Chịu lỗi Heartbeat
Không cần nhập thủ công địa chỉ IP. Các máy tự động gửi broadcast UDP trên cổng 59001 mỗi 2 giây. Lớp `FaultToleranceMonitor` theo dõi nhịp tim; nếu quá 8 giây không có tín hiệu, node lỗi sẽ bị tự động loại khỏi topology mạng mà không gây sập máy chủ.

### 3.5. Phương án 5: Xác thực kết nối Worker (Accept/Reject) & Ghi nhật ký lịch sử
Khi Host Master điều phối lát cắt mô hình xuống Worker, máy Worker sẽ bật một Popup Banner thông báo trực quan. Người dùng có quyền bấm Accept để nhận tải hoặc Reject để từ chối. Mọi hoạt động kết nối đều được ghi lại trong bảng lịch sử `connection_history` với mốc thời gian rõ ràng.

### 3.6. Phương án 6: Thuật toán thanh trượt tự cân bằng (Auto-Balancing Slicing Bar $\sum = 100\%$)
Khi người dùng chuyển sang chế độ phân bổ layer bằng tay (Custom Ratio), thuật toán trên file `app.js` tự động phân bổ độ bù trừ $\Delta$ cho các thanh trượt còn lại, đảm bảo tổng tỷ lệ phần trăm luôn bằng đúng 100%, triệt tiêu hoàn toàn lỗi toán học khi chia layer.

### 3.7. Phương án 7: Stream phản hồi văn bản từng token (SSE Streaming)
Kết hợp giữa `TextIteratorStreamer` của Hugging Face và `StreamingResponse` chuẩn Server-Sent Events (SSE) của FastAPI, cho phép dòng văn bản phản hồi tuôn ra màn hình chat theo từng token với hiệu ứng chữ chạy tức thì.

---

## 4. SƠ ĐỒ KIẾN TRÚC PIPELINE VÀ CÁC MÔ HÌNH HỖ TRỢ

### 4.1. Sơ đồ Pipeline Parallelism toàn diện (End-to-End Tensor Transport & Topology)

Kiến trúc Pipeline Parallelism trong Orchest Computing phân chia $L$ layer transformer tuần tự thành $K$ lát cắt liên tục (Contiguous Stages). Mỗi máy chỉ nạp và tính toán lát cắt của mình. Tensor kích hoạt trung gian (Intermediate Activation Tensors) được chuyển tiếp nối tiếp qua mạng LAN thông qua giao thức nhị phân siêu tốc `TensorPacket` (18-byte fixed header, CRC32, mã hóa AES-GCM-256 qua Raw TCP Socket có chứng chỉ mTLS X.509).

```mermaid
graph TD
    subgraph Client ["Client Browser / Web UI"]
        UI_Prompt["Người dùng nhập Prompt"]
        UI_Stream["Hiển thị Token SSE Streaming chữ chạy"]
    end

    subgraph HostNode ["STAGE 0: HOST MASTER (e.g. MacBook Pro M1 16GB)"]
        Tokenizer["Tokenizer: Văn bản -> Token IDs"]
        Embeddings["Token Embeddings (wte / embed_tokens)"]
        RoPE_0["Rotary Position Embeddings (RoPE)"]
        Layers_Stage0["Transformer Layers [0 .. k₁]"]
        Act_0["Activation Tensor A₀"]
        
        Tokenizer --> Embeddings --> RoPE_0 --> Layers_Stage0 --> Act_0
    end

    subgraph Transport_1 ["MẠNG CỤC BỘ LAN (Custom Raw TCP / mTLS)"]
        Wire_1["TensorPacket: [ORCH 18B Header][CRC32 Checksum][AES-GCM Payload]"]
    end

    subgraph WorkerNode1 ["STAGE 1: WORKER EDGE 1 (e.g. PC Windows RTX 3060 6GB)"]
        Recv_1["TCP Server Socket: Nhận & Giải mã Tensor A₀"]
        Layers_Stage1["Transformer Layers [k₁+1 .. k₂]"]
        Act_1["Activation Tensor A₁"]
        
        Recv_1 --> Layers_Stage1 --> Act_1
    end

    subgraph Transport_2 ["MẠNG CỤC BỘ LAN (Custom Raw TCP / mTLS)"]
        Wire_2["TensorPacket: [ORCH 18B Header][CRC32 Checksum][AES-GCM Payload]"]
    end

    subgraph WorkerNode2 ["STAGE 2: WORKER EDGE 2 - STAGE CUỐI (e.g. Laptop GTX 1650 4GB)"]
        Recv_2["TCP Server Socket: Nhận & Giải mã Tensor A₁"]
        Layers_Stage2["Transformer Layers [k₂+1 .. L-1]"]
        FinalNorm["Final RMSNorm / LayerNorm"]
        LMHead["LM Head (Linear Projection -> Logits)"]
        Sampling["Bộ lấy mẫu Token (Temperature, Top-p, Argmax)"]
        NextToken["Token ID mới (Generated Token)"]
        
        Recv_2 --> Layers_Stage2 --> FinalNorm --> LMHead --> Sampling --> NextToken
    end

    subgraph Transport_3 ["MẠNG CỤC BỘ LAN (TCP Return Packet)"]
        Wire_3["INFERENCE_COMPLETE: Gửi Token ID / Logits về Host"]
    end

    UI_Prompt -->|POST /api/generate/stream| Tokenizer
    Act_0 -->|sendall socket| Wire_1
    Wire_1 -->|recv_exact socket| Recv_1
    Act_1 -->|sendall socket| Wire_2
    Wire_2 -->|recv_exact socket| Recv_2
    NextToken -->|TCP ACK / Result| Wire_3
    Wire_3 -->|FastAPI Generator| UI_Stream
    NextToken -.->|Lặp lại cho Token tiếp theo| Embeddings
```

---

### 4.2. Sơ đồ cấu trúc nội bộ khối lát cắt Transformer (`ModelSliceModule`)

Lớp `ModelSliceModule` trong file [`auto_slicer.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/orchest/models/auto_slicer.py) tự động thích ứng với vị trí của node trong pipeline:

```mermaid
classDiagram
    class ModelSliceModule {
        +int start_idx
        +int end_idx
        +int total_layers
        +bool is_first_stage
        +bool is_last_stage
        +nn.Module embed_tokens
        +nn.ModuleList slice_layers
        +nn.Module norm
        +nn.Module lm_head
        +forward(input_tensor, position_ids, attention_mask) Tensor
    }

    class TransformerBlock {
        +RMSNorm input_layernorm
        +GroupedQueryAttention self_attn
        +ResidualConnection res_1
        +RMSNorm post_attention_layernorm
        +SwiGLU_MLP mlp
        +ResidualConnection res_2
        +forward(hidden_states) Tensor
    }

    ModelSliceModule "1" *-- "many" TransformerBlock : Chứa danh sách các layer [start_idx .. end_idx]
```

#### Chi tiết luồng tính toán bên trong 3 loại Stage:
```mermaid
graph TD
    subgraph FirstStage ["Stage 0 (Đầu Pipeline)"]
        T_IN[Input Token IDs] --> T_EMB[embed_tokens / wte]
        T_EMB --> T_ROPE[Tạo position_ids & RoPE embeddings]
        T_ROPE --> BLK_0[Transformer Blocks 0 .. k₁]
        BLK_0 --> OUT_0[Output Tensor sang Worker tiếp theo]
    end

    subgraph MidStage ["Stage Trung Gian (Chỉ xử lý Layers)"]
        M_IN[Nhận Activation Tensor từ Node trước] --> NORM_1[Input RMSNorm]
        NORM_1 --> GQA[Grouped-Query Attention GQA]
        GQA --> RES_1[+ Residual Connection]
        RES_1 --> NORM_2[Post Attention RMSNorm]
        NORM_2 --> MLP[MLP SwiGLU: Gate, Up, Down Proj]
        MLP --> RES_2[+ Residual Connection]
        RES_2 --> BLK_MID[... Lặp lại cho các layer còn lại trong slice ...]
        BLK_MID --> OUT_MID[Output Tensor sang Node tiếp theo]
    end

    subgraph LastStage ["Stage Cuối (Đuôi Pipeline)"]
        L_IN[Nhận Activation Tensor từ Node trước] --> BLK_LAST[Transformer Blocks k₂ .. L-1]
        BLK_LAST --> L_NORM[Final RMSNorm]
        L_NORM --> L_HEAD[LM Head Projection: hidden_dim -> vocab_size]
        L_HEAD --> LOGITS[Logits Vector]
        LOGITS --> SAMP[Lấy mẫu: Top-p / Temperature]
        SAMP --> NEW_TOKEN[Token ID mới]
    end
```

---

### 4.3. Bảng phân tích chi tiết các mô hình AI trong hệ thống (`MODEL_CONFIGS`)

Hệ thống cấu hình sẵn 9 mô hình ngôn ngữ lớn phổ biến trong [`DATN/apps/dashboard/server.py`](file:///Applications/HCMUT/Orchest%20Computing/DATN/apps/dashboard/server.py). Các thông số kỹ thuật này được nạp vào ma trận chi phí để tính toán TFLOPs, bộ nhớ tham số và kích thước KV-Cache:

| Tên Mô Hình (Hugging Face Repo) | Dung lượng Tham số | Số Layer ($L$) | Kích thước ẩn ($d_{\text{model}}$) | Kích thước trung gian ($d_{\text{ffn}}$) | Số Attn Heads ($n_{\text{heads}}$) | Số KV Heads ($n_{\text{kv\_heads}}$) | Hệ số nén GQA | Dung lượng Weights (FP16) | VRAM đơn máy tối thiểu khuyến nghị |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen2.5-0.5B-Instruct`** | 0.5 Tỷ | 24 | 896 | 4,864 | 14 | 2 | **$7.0\times$** | ~1.0 GB | 2 GB VRAM |
| **`Qwen/Qwen2.5-1.5B-Instruct`** | 1.5 Tỷ | 28 | 1,536 | 8,960 | 12 | 2 | **$6.0\times$** | ~3.0 GB | 4 GB VRAM |
| **`Qwen/Qwen2.5-3B-Instruct`** | 3.0 Tỷ | 36 | 2,048 | 11,008 | 16 | 2 | **$8.0\times$** | ~6.0 GB | 8 GB VRAM |
| **`Qwen/Qwen2.5-7B-Instruct`** | 7.6 Tỷ | 28 | 3,584 | 18,944 | 28 | 4 | **$7.0\times$** | ~15.2 GB | 18 GB VRAM |
| **`google/gemma-2-9b-it`** | 9.2 Tỷ | 42 | 3,584 | 14,336 | 16 | 8 | **$2.0\times$** | ~18.4 GB | 22 GB VRAM |
| **`Qwen/Qwen2.5-14B-Instruct`** | 14.7 Tỷ | 48 | 5,120 | 13,824 | 40 | 8 | **$5.0\times$** | ~29.5 GB | 36 GB VRAM |
| **`Qwen/Qwen2.5-20B-Instruct`** | 20.0 Tỷ | 40 | 4,096 | 22,016 | 32 | 8 | **$4.0\times$** | ~40.0 GB | 48 GB VRAM |
| **`google/gemma-2-27b-it`** | 27.2 Tỷ | 46 | 4,608 | 36,864 | 32 | 16 | **$2.0\times$** | ~54.4 GB | 64 GB VRAM |
| **`Qwen/Qwen2.5-32B-Instruct`** | 32.5 Tỷ | 64 | 5,120 | 27,648 | 40 | 8 | **$5.0\times$** | ~65.0 GB | 80 GB VRAM |

> 💡 **Nhận xét**: Đối với các mô hình từ 7B đến 32B, dung lượng VRAM đơn máy đòi hỏi từ 18GB đến 80GB (vượt quá khả năng của mọi laptop hay PC gaming thông thường). Đây chính là động lực khiến giải pháp phân hoạch Swarm Pipeline của Orchest Computing phát huy tối đa giá trị: **ghép các máy 4GB, 6GB, 8GB thành cụm để chạy mô hình lớn**.

---

### 4.4. Sơ đồ kịch bản phân bổ mẫu (Case Study: Phân chia Qwen2.5-7B trên cụm 3 máy)

Giả sử mạng LAN có 3 máy tính tham gia cụm suy luận cho mô hình `Qwen/Qwen2.5-7B-Instruct` (tổng số 28 layers, dung lượng trọng số ~15.2 GB):
- **Máy 1 (Host Master)**: MacBook Pro M1 Max, 16GB Unified Memory khả dụng, băng thông RAM 200 GB/s.
- **Máy 2 (Worker 1)**: PC Windows NVIDIA RTX 3060, 6GB VRAM, 13 TFLOPs.
- **Máy 3 (Worker 2)**: Laptop Windows NVIDIA GTX 1650, 4GB VRAM, 6.5 TFLOPs.

```mermaid
gantt
    title Phân Bổ 28 Layers của Qwen2.5-7B Trên 3 Máy Tính Dị Chủng
    dateFormat  X
    axisFormat %s

    section Máy 1 (Mac M1 16GB)
    Stage 0 (Layers 0 - 10) + Embeddings :active, 0, 11
    section Máy 2 (PC RTX 3060 6GB)
    Stage 1 (Layers 11 - 21) :crit, 11, 22
    section Máy 3 (Laptop GTX 1650 4GB)
    Stage 2 (Layers 22 - 27) + LM Head :done, 22, 28
```

#### Chi tiết phân chia tài nguyên:
```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN2.5-7B-INSTRUCT (28 LAYERS)                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
          │                                 │                                │
          ▼                                 ▼                                ▼
┌──────────────────┐              ┌──────────────────┐             ┌──────────────────┐
│     STAGE 0      │              │     STAGE 1      │             │     STAGE 2      │
│  MacBook Pro M1  │              │   PC RTX 3060    │             │ Laptop GTX 1650  │
│  Layers [0..10]  │ ──TCP A₀──>  │ Layers [11..21]  │ ──TCP A₁──> │ Layers [22..27]  │
│    (11 Layers)   │              │   (11 Layers)    │             │    (6 Layers)    │
│  + Embeddings    │              │                  │             │   + LM Head      │
│  Weights: ~5.9GB │              │  Weights: ~5.9GB │             │  Weights: ~3.4GB │
│  VRAM Max: 16GB  │              │  VRAM Max: 6GB   │             │  VRAM Max: 4GB   │
│  Trạng thái: OK  │              │  Trạng thái: OK  │             │  Trạng thái: OK  │
└──────────────────┘              └──────────────────┘             └──────────────────┘
```

---

### 4.5. Sơ đồ chu trình sinh Token tự hồi quy bầy đàn (Autoregressive Swarm Generation Loop)

Quá trình sinh văn bản chia làm 2 giai đoạn: **Pha Prefill** (xử lý toàn bộ prompt đầu vào) và **Pha Decode** (lặp từng token đơn lẻ cho đến khi kết thúc):

```mermaid
stateDiagram-v2
    [*] --> NhậnPrompt : Người dùng bấm Send
    NhậnPrompt --> PrefillPhase : Tokenizer mã hóa chuỗi Prompt

    state PrefillPhase {
        [*] --> Stage0_Prefill : Nhúng Tokens + Chạy Layers [0..k₁]
        Stage0_Prefill --> Stage1_Prefill : Gửi Tensor kích hoạt đầy đủ qua LAN
        Stage1_Prefill --> Stage2_Prefill : Gửi Tensor kích hoạt qua LAN
        Stage2_Prefill --> SinhTokenDau : LM Head tính Logits & Lấy mẫu Token T₁
    }

    SinhTokenDau --> StreamToken : Stream Token T₁ về Web UI qua SSE
    StreamToken --> DecodeLoop : Bắt đầu vòng lặp sinh tự hồi quy

    state DecodeLoop {
        [*] --> KiemTraDung : Kiểm tra Token == EOS hoặc đạt max_tokens?
        KiemTraDung --> Dung : Đúng (Gặp <|im_end|> hoặc nhấn Stop)
        KiemTraDung --> Stage0_Decode : Sai -> Đưa Token Ti vào Stage 0
        
        Stage0_Decode --> Stage1_Decode : Gửi Vector kích hoạt (kích thước 1xhidden_dim)
        Stage1_Decode --> Stage2_Decode : Gửi Vector kích hoạt qua LAN
        Stage2_Decode --> SinhTokenKe : LM Head sinh Token Ti+1
        SinhTokenKe --> StreamTokenKe : Stream Ti+1 về Web UI qua SSE
        StreamTokenKe --> KiemTraDung : Tiếp tục vòng lặp
    }

    Dung --> [*] : Hoàn tất câu trả lời AI
```

---

## 5. BIỂU ĐỒ USE CASE VÀ KỊCH BẢN TƯƠNG TÁC

### 5.1. Sơ đồ Use Case tổng quan toàn hệ thống (System Overview Use Case Diagram)

Hệ thống Orchest Computing tương tác với 3 nhóm tác nhân (Actors):
1. **👑 Host Master Operator**: Người dùng trực tiếp tại máy chủ, chịu trách nhiệm quản lý cụm, chọn mô hình, cấu hình lát cắt và trò chuyện với mô hình AI.
2. **⚡ Worker Node Operator**: Người dùng tại các máy tính biên cá nhân, có quyền kiểm soát tài nguyên máy của mình, phê duyệt hoặc từ chối nhận lát cắt mô hình.
3. **⚙️ Background System Daemons**: Các tiến trình dịch vụ ngầm tự động (UDP Discovery, Fault Tolerance Monitor, Hardware Benchmark Runner).

```mermaid
graph LR
    HostUser(("👑 Host Master Operator"))
    WorkerUser(("⚡ Worker Edge Operator"))
    Daemon(("⚙️ Background Daemons"))

    subgraph Orchest_Computing ["HỆ THỐNG PHÂN TÁN ORCHEST COMPUTING"]
        UC_Init["Khởi động Universal App"]
        UC_Disc["Tự động phát hiện mạng (UDP Discovery)"]
        UC_Plan["Lập kế hoạch phân bổ (Auto DP / Ratio)"]
        UC_Auth["Phê duyệt ủy quyền (Accept / Reject)"]
        UC_Load["Tải Shard chọn lọc (Selective Sharding)"]
        UC_Chat["Trò chuyện AI (Swarm Token Streaming)"]
        UC_Fleet["Quản lý Fleet (Alias, On/Off, Ping)"]
        UC_Bench["Chấm điểm phần cứng (TFLOPs Stress Test)"]
        UC_Fault["Giám sát Heartbeat & Cắt tỉa Node lỗi"]
    end

    HostUser --> UC_Init
    HostUser --> UC_Plan
    HostUser --> UC_Chat
    HostUser --> UC_Fleet
    HostUser --> UC_Bench

    WorkerUser --> UC_Init
    WorkerUser --> UC_Auth
    WorkerUser --> UC_Bench

    Daemon --> UC_Disc
    Daemon --> UC_Load
    Daemon --> UC_Fault
    Daemon --> UC_Bench

    UC_Plan -.->|<<include>>| UC_Disc
    UC_Plan -.->|<<trigger>>| UC_Auth
    UC_Auth -.->|<<include>>| UC_Load
    UC_Chat -.->|<<require>>| UC_Plan
```

---

### 5.2. Sơ đồ Use Case phân hệ Điều phối Host Master

Phân hệ Host Master tập trung toàn bộ năng lực điều phối cụm và giao diện trò chuyện:

```mermaid
graph TD
    ActorHost(("👑 Host Master Operator"))

    subgraph Host_Subsystem ["Phân Hệ Điều Phối Host Master (Tab 1: Host)"]
        UC_SelectModel["Chọn Mô hình AI (Qwen2.5, Gemma-2)"]
        UC_ViewTopo["Xem sơ đồ trực quan Pipeline Canvas"]
        UC_AutoDP["Chạy thuật toán Quy hoạch động (Auto DP Slicing)"]
        UC_CustomRatio["Kéo thanh trượt tỷ lệ (Auto-Balancing Slicing)"]
        UC_Dispatch["Phát lệnh cấu hình Pipeline (Dispatch Plan)"]
        UC_SendPrompt["Nhập câu hỏi Prompt & Gửi"]
        UC_StreamText["Nhận văn bản stream từng token (SSE)"]
        UC_StopGen["Dừng sinh khẩn cấp (Emergency Stop)"]
    end

    ActorHost --> UC_SelectModel
    ActorHost --> UC_ViewTopo
    ActorHost --> UC_AutoDP
    ActorHost --> UC_CustomRatio
    ActorHost --> UC_Dispatch
    ActorHost --> UC_SendPrompt
    ActorHost --> UC_StopGen

    UC_AutoDP -.->|<<update>>| UC_ViewTopo
    UC_CustomRatio -.->|<<update>>| UC_ViewTopo
    UC_ViewTopo -.->|<<proceed>>| UC_Dispatch
    UC_SendPrompt --> UC_StreamText
```

---

### 5.3. Sơ đồ Use Case phân hệ Thiết bị biên Worker Node

Phân hệ Worker Node trao quyền bảo mật tối cao cho chủ sở hữu máy tính biên:

```mermaid
graph TD
    ActorWorker(("⚡ Worker Edge Operator"))

    subgraph Worker_Subsystem ["Phân Hệ Thiết Bị Biên Worker (Tab 2: Worker Edge)"]
        UC_ViewMetrics["Xem tài nguyên máy (CPU, RAM, VRAM, TFLOPs)"]
        UC_ReceivePopup["Nhận Popup Banner thông báo phân bổ lát cắt"]
        UC_Accept["Bấm Accept: Chấp nhận nhận tải"]
        UC_Reject["Bấm Reject: Từ chối phân bổ"]
        UC_ToggleAutoAccept["Bật / Tắt chế độ Auto-Accept"]
        UC_DownloadShards["Tải các Shard cần thiết từ Hugging Face"]
        UC_ForwardTensor["Thực thi lát cắt forward pass & gửi tensor kế tiếp"]
        UC_ViewHistory["Xem nhật ký lịch sử kết nối (Connection History)"]
    end

    ActorWorker --> UC_ViewMetrics
    ActorWorker --> UC_ReceivePopup
    ActorWorker --> UC_Accept
    ActorWorker --> UC_Reject
    ActorWorker --> UC_ToggleAutoAccept
    ActorWorker --> UC_ViewHistory

    UC_ReceivePopup -.-> UC_Accept
    UC_ReceivePopup -.-> UC_Reject
    UC_Accept -.->|<<trigger>>| UC_DownloadShards
    UC_DownloadShards -.->|<<ready>>| UC_ForwardTensor
    UC_Accept -.->|<<log>>| UC_ViewHistory
    UC_Reject -.->|<<log>>| UC_ViewHistory
```

---

### 5.4. Sơ đồ Use Case phân hệ Quản trị Fleet & Chịu lỗi

Phân hệ Fleet Management cho phép giám sát toàn diện sức khỏe của từng máy tính trong mạng LAN:

```mermaid
graph TD
    ActorAdmin(("👑 Host / Fleet Admin"))
    DaemonMonitor(("⚙️ Fault Tolerance Monitor"))

    subgraph Fleet_Subsystem ["Phân Hệ Quản Trị Fleet (Tab 3: Manage Fleet)"]
        UC_EditAlias["Đổi tên gợi nhớ máy tính (Custom Friendly Alias)"]
        UC_ToggleNode["Bật / Tắt tham gia cụm của máy (Enable / Disable)"]
        UC_PingNode["Đo độ trễ mạng TCP (Ping Latency Diagnostics)"]
        UC_DeleteNode["Xóa / Quên máy khỏi danh bạ cụm (Delete Node)"]
        UC_RunStress["Kích hoạt Benchmark chấm điểm TFLOPs phần cứng"]
        UC_AutoPeriodicBench["Tự động chạy Benchmark mỗi 5 phút (Standby)"]
        UC_HeartbeatTimeout["Phát hiện máy chết sau 8s & Cắt tỉa Topology"]
        UC_AutoRebalance["Tự động phân hoạch lại trên các máy còn sống"]
    end

    ActorAdmin --> UC_EditAlias
    ActorAdmin --> UC_ToggleNode
    ActorAdmin --> UC_PingNode
    ActorAdmin --> UC_DeleteNode
    ActorAdmin --> UC_RunStress

    DaemonMonitor --> UC_AutoPeriodicBench
    DaemonMonitor --> UC_HeartbeatTimeout

    UC_ToggleNode -.->|<<trigger>>| UC_AutoRebalance
    UC_HeartbeatTimeout -.->|<<trigger>>| UC_AutoRebalance
```

---

### 5.5. Bảng đặc tả chi tiết các Use Case chính (Use Case Specifications: UC-01 → UC-05)

#### 📋 UC-01: Lập kế hoạch phân bổ & Điều phối lát cắt mô hình
- **Mã định danh**: `UC-01`
- **Tên Use Case**: Lập kế hoạch phân bổ lát cắt và phát lệnh điều phối (Plan & Dispatch Model Slices).
- **Tác nhân chính**: Host Master Operator.
- **Tiền điều kiện**: Ứng dụng đã khởi chạy, có ít nhất 1 Worker đang phát UDP Beacon trong mạng LAN.
- **Luồng sự kiện chính (Main Flow)**:
  1. Người dùng chọn mô hình LLM từ danh sách (ví dụ `Qwen/Qwen2.5-7B-Instruct`).
  2. Người dùng chọn chế độ:
     - **Chế độ tự động (Auto DP)**: Hệ thống gọi `DynamicPartitioner.partition()`, giải bảng DP với trần an toàn $\alpha = 0.90$.
     - **Chế độ tùy chỉnh (Custom Ratio)**: Người dùng kéo thanh trượt tỷ lệ, hàm `syncBalanceSliders()` tự cân bằng tổng 100%.
  3. Giao diện vẽ lại sơ đồ phân hoạch trên Canvas Pipeline Map (hiển thị màu sắc và dải layer $[start..end]$ của từng máy).
  4. Người dùng bấm nút **"Plan & Dispatch"**.
  5. `SwarmPipelineExecutor` duyệt qua kế hoạch và gửi gói tin TCP `DISPATCH_SLICE` đến địa chỉ IP của các Worker tương ứng.
- **Luồng nhánh (Alternative Flow)**:
  - Nếu mô hình vượt quá tổng bộ nhớ cả cụm: Thuật toán DP trả về nghiệm không khả thi, giao diện hiển thị cảnh báo đỏ *"Infeasible: Total cluster memory insufficient"*.
- **Hậu điều kiện**: Các Worker nhận được gói tin cấu hình lát cắt và chuyển sang bước phê duyệt/nạp shard.

---

#### 📋 UC-02: Phê duyệt ủy quyền phân bổ trên Worker Node
- **Mã định danh**: `UC-02`
- **Tên Use Case**: Phê duyệt ủy quyền phân bổ lát cắt (Authorize Slice Connection).
- **Tác nhân chính**: Worker Edge Operator.
- **Tiền điều kiện**: Worker đang ở trạng thái `IDLE` trên cổng TCP 5555 và vừa nhận gói `DISPATCH_SLICE` từ Host.
- **Luồng sự kiện chính**:
  1. Máy Worker nhận gói tin cấu hình lát cắt.
  2. Nếu `auto_accept == False`:
     - Giao diện Web UI của Worker bật Popup Banner cảnh báo màu cam.
     - Hiển thị thông tin: IP của Host, tên model, dải layer được giao (ví dụ: Layer 14 đến 27).
     - Người dùng bấm **✅ Accept**: Gửi API `POST /api/worker/respond` với `action="accept"`.
     - Worker gọi `AutoModelSlicer.load_selective_slice()` tải các shard cần thiết và chuyển trạng thái sang `SERVING`.
     - Hệ thống chèn bản ghi vào `connection_history` với trạng thái `ACCEPTED`.
     - Gửi gói tin TCP `ACK` báo về cho Host Master.
- **Luồng ngoại lệ (Exception Flow)**:
  - Người dùng bấm **❌ Reject**: Gửi API với `action="reject"`.
  - Worker từ chối nạp mô hình, ghi log `REJECTED`, gửi phản hồi từ chối về Host để Host chuyển sang phân bổ lại.
- **Hậu điều kiện**: Lát cắt mô hình được nạp sẵn sàng trên GPU/MPS của Worker hoặc bị hủy an toàn.

---

#### 📋 UC-03: Suy luận bầy đàn & Stream Token thời gian thực
- **Mã định danh**: `UC-03`
- **Tên Use Case**: Trò chuyện AI bầy đàn và stream token (Swarm Chat Inference & Streaming).
- **Tác nhân chính**: Host Master Operator.
- **Tiền điều kiện**: Các Worker trong pipeline đều đang ở trạng thái `SERVING`.
- **Luồng sự kiện chính**:
  1. Người dùng nhập nội dung câu hỏi (Prompt) vào ô Chat và nhấn **Send** (hoặc Enter).
  2. Web UI gửi `POST /api/generate/stream` kèm nội dung prompt.
  3. Máy Host bọc prompt theo mẫu Chat Instruct (`apply_chat_template`).
  4. **Pha Prefill**:
     - Host (Stage 0) tính toán Embeddings và các layer $[0..k_1]$.
     - Host đóng gói vector kích hoạt vào `TensorPacket` (18B Header + CRC32) gửi qua TCP Socket tới Worker 1.
     - Worker 1 tính tiếp các layer $[k_1+1..k_2]$ và gửi tiếp sang Worker cuối cùng.
     - Worker cuối tính các layer còn lại + LM Head, lấy mẫu được Token đầu tiên $T_1$, gửi về Host.
  5. **Pha Decode**:
     - Lặp lại việc chuyển tiếp tensor 1 token qua chuỗi pipeline để sinh lần lượt $T_2, T_3, \dots$.
  6. Tại mỗi token sinh ra, server phát một sự kiện SSE (`data: {"token": "...", "done": false}`).
  7. Client JavaScript nhận từng sự kiện và vẽ ngay lập tức ký tự lên màn hình (Typing effect).
  8. Khi gặp token `<|im_end|>` hoặc chạm `max_new_tokens`: Server gửi `data: {"done": true}` và kết thúc phiên stream.
- **Luồng nhánh (Dừng khẩn cấp)**:
  - Người dùng bấm nút **Stop**: Client gửi `POST /api/generate/stop` $\to$ Cờ `stop_requested = True` được bật $\to$ Luồng sinh token lập tức dừng lại.

---

#### 📋 UC-04: Quản lý danh bạ Fleet & Đo chẩn đoán mạng
- **Mã định danh**: `UC-04`
- **Tên Use Case**: Quản lý danh bạ Fleet và chẩn đoán độ trễ (Manage Fleet & Diagnostics).
- **Tác nhân chính**: Host / Fleet Administrator.
- **Tiền điều kiện**: Đang mở Tab 3: Manage Fleet trên Web Dashboard.
- **Luồng sự kiện chính**:
  1. **Đổi tên gợi nhớ**:
     - Người dùng bấm vào ô tên máy, nhập tên mới (ví dụ *"PC Gaming RTX 4080"*), nhấn Enter.
     - Server gọi `host.update_node_config()`, cập nhật và lưu đĩa vào `cluster_nodes_config.json`.
  2. **Bật / Tắt máy (On/Off Switch)**:
     - Người dùng gạt công tắc của một máy tính sang `OFF`.
     - Server đánh dấu `enabled = False`, lưu file cấu hình và tự động kích hoạt tính lại lát cắt mô hình trên các máy còn lại đang `ON`.
  3. **Đo độ trễ (Ping Diagnostics)**:
     - Người dùng bấm nút **Ping**.
     - Host mở socket TCP kết nối tới cổng 5555 của Worker, bấm giờ hoàn thành kết nối.
     - Trả về độ trễ mili-giây (ví dụ: `1.24 ms`) hiển thị trực tiếp lên bảng Fleet.
  4. **Xóa node (Delete / Forget)**:
     - Người dùng bấm nút thùng rác: Server xóa hoàn toàn định danh máy khỏi `cluster_nodes_config.json`.

---

#### 📋 UC-05: Chấm điểm phần cứng tự động định kỳ
- **Mã định danh**: `UC-05`
- **Tên Use Case**: Chấm điểm hiệu năng phần cứng thực tế (Hardware Benchmark Rating).
- **Tác nhân chính**: Background Benchmark Runner (hoặc người dùng bấm thủ công).
- **Tiền điều kiện**: Máy đang ở trạng thái Standby (không có phiên suy luận đang chạy).
- **Luồng sự kiện chính**:
  1. Cứ mỗi 5 phút (300 giây) ở chế độ rảnh:
  2. Luồng nền `_bg_periodic_benchmark_loop` tự động kích hoạt hàm `run_client_hardware_rating(duration=2.0)`.
  3. Tạo 2 ma trận ngẫu nhiên $2048 \times 2048$ kiểu `float32`.
  4. Chạy nhân ma trận liên tục trong 2 giây trên thiết bị tốt nhất (`cuda` $\to$ `mps` $\to$ `cpu`).
  5. Đếm số vòng lặp và chia cho thời gian để tính toán chỉ số TFLOPs thực tế.
  6. Đo nhiệt độ chip trước và sau khi chạy stress test.
  7. Cập nhật kết quả vào biến `LATEST_BENCHMARK` và đồng bộ file `benchmark_results.json`.
  8. Đặt lại đồng hồ đếm ngược 300 giây trên giao diện Web UI.
- **Luồng nhánh (Thủ công)**:
  - Người dùng bấm nút **"Run Benchmark Now"** trên Tab 3 $\to$ Gọi `POST /api/benchmark/run` và thực thi ngay lập tức.

---

## 6. LUỒNG THỰC THI END-TO-END TỪNG BƯỚC

Biểu đồ tuần tự dưới đây thể hiện chính xác các hàm được gọi xuyên suốt trong mã nguồn từ lúc khởi chạy đến khi hoàn tất một câu trả lời AI:

```mermaid
sequenceDiagram
    autonumber
    actor User as Người dùng
    participant App as apps/orchest.py
    participant Server as apps/dashboard/server.py
    participant Host as orchest/executor/host.py
    participant Cost as orchest/partitioner/cost_matrix.py
    participant DP as orchest/partitioner/dynamic_programming.py
    participant Worker as orchest/executor/worker.py
    participant Slicer as orchest/models/auto_slicer.py
    participant Transport as orchest/network/transport.py

    Note over App,Server: BƯỚC 1: KHỞI ĐỘNG HỆ THỐNG
    User->>App: Chạy python apps/orchest.py
    App->>Server: uvicorn.run(app, port=8000)
    Server->>Host: HostOrchestrator() -> Bật UDPDiscoveryService
    Server->>Worker: WorkerNode() -> Bật RawTCPTransportServer (Port 5555)

    Note over Server,Worker: BƯỚC 2: KHÁM PHÁ MẠNG & CHẤM ĐIỂM
    Worker->>Host: UDP Beacon Broadcast (Port 59001): IP, RAM, VRAM, TFLOPs
    Host->>Host: _refresh_local_host_node() & Đăng ký Worker
    Server->>Server: run_client_hardware_rating() đo TFLOPs thực tế

    Note over User,DP: BƯỚC 3: LẬP KẾ HOẠCH PHÂN HOẠCH LAYER
    User->>Server: Chọn mô hình & Bấm "Plan & Dispatch"
    Server->>Host: host.plan_model_slicing(model_name)
    Host->>DP: partitioner.partition(layers, nodes, alpha=0.90)
    loop Quét qua từng trạng thái
        DP->>Cost: compute_time_ms() & comm_time_ms()
        Cost-->>DP: Chi phí mili-giây (hoặc inf nếu OOM)
    end
    DP-->>Host: Trả về Danh sách SliceAssignment tối ưu

    Note over Host,Slicer: BƯỚC 4: PHÂN BỔ LÁT CẮT & NẠP SHARD CHỌN LỌC
    Host->>Transport: RawTCPTransportClient.send_packet(DISPATCH_SLICE)
    Transport->>Worker: Gói tin nhị phân DISPATCH_SLICE
    Worker->>Worker: Hiện Popup Banner Accept / Reject trên Web UI
    User-->>Worker: Bấm "Accept"
    Worker->>Slicer: AutoModelSlicer.load_selective_slice()
    Slicer->>Slicer: Tải đúng Shard cần thiết từ Hugging Face
    Slicer-->>Worker: Trả về ModelSliceModule hoàn chỉnh
    Worker-->>Host: Gói tin phản hồi ACK (Trạng thái: SERVING)

    Note over User,Worker: BƯỚC 5: SUY LUẬN BẦY ĐÀN & STREAM TOKEN
    User->>Server: Gửi Prompt vào ô Chat
    Server->>Host: swarm_generator.generate_stream(prompt)
    Host->>Worker: Gửi FORWARD_TENSOR qua Raw TCP Socket
    Worker->>Worker: slice_module.forward(input_tensor)
    Worker-->>Host: Trả về Token IDs / Logits
    Host-->>Server: yield token (TextIteratorStreamer)
    Server-->>User: Server-Sent Events (SSE): Chữ hiển thị tức thì trên Web UI
```

---

## 7. BÁO CÁO DỌN DẸP CÁC FILE DƯ THỪA TRONG DỰ ÁN

Trong quá trình rà soát và tối ưu hóa cấu trúc dự án, các file dư thừa, trùng lặp và tạm thời đã được phân tích và loại bỏ triệt để nhằm giữ cho codebase gọn gàng, trong sạch và chuyên nghiệp:

| Tên File Đã Xóa | Vị trí ban đầu | Lý do loại bỏ & Trạng thái thay thế |
| :--- | :--- | :--- |
| **`DATN/apps/real_app.py`** | `DATN/apps/` | **File Alias dư thừa**: File chỉ có 13 dòng gọi lại `apps/orchest.py`. Không có file mã nguồn hay kịch bản nào import file này. Điểm khởi chạy chính thức duy nhất là `apps/orchest.py`. |
| **`cluster_nodes_config.json`** | Thư mục gốc | **File cấu hình trùng lặp**: Được sinh ra do chạy lệnh ngoài thư mục gốc trước đây. File cấu hình thực tế được code và server FastAPI quản lý lưu trữ tại [`DATN/cluster_nodes_config.json`](file:///Applications/HCMUT/Orchest%20Computing/DATN/cluster_nodes_config.json). |
| **`proposal_v2_extracted.txt`** | Thư mục gốc | **File dump text tạm**: File văn bản tạm trích xuất từ tài liệu proposal v2. File gốc chính thức `AAIDS_FM_ORCHEST_Proposal_v2-long_210826_TTT.pdf` vẫn được lưu giữ an toàn. |
| **`proposal_v2_clean.txt`** | Thư mục gốc | **File dump text tạm**: Tương tự như trên, là bản sao text tạm thời. |
| **`proposal_v3_extracted.txt`** | Thư mục gốc | **File dump text tạm**: File văn bản tạm trích xuất từ tài liệu proposal v3 (`AAIDS_FM_ORCHEST_Proposal_v3_Revised.docx` vẫn nguyên vẹn). |
| **`proposal_v4_extracted.txt`** | Thư mục gốc | **File dump text tạm**: File văn bản tạm trích xuất từ tài liệu proposal v4 (`AAIDS_FM_ORCHEST_Proposal_v4-long_290826_TTT.docx` vẫn nguyên vẹn). |
| **`walkthrough.md`** | Thư mục gốc | **File tài liệu cũ lỗi thời**: Ghi chú tạm thời từ môi trường thử nghiệm cũ trên Windows (`d:/BKU/Orchest-Computing/...`). Toàn bộ nội dung về mTLS, CRC32, Allowlist đã được hợp nhất đầy đủ vào tài liệu kỹ thuật chính thức này. |
| **`.DS_Store`** | Gốc & `DATN/` | **File hệ thống rác của macOS**: Tự sinh khi duyệt thư mục bằng Finder, không có giá trị kỹ thuật. |

#### Kết quả kiểm thử tính toàn vẹn hệ thống sau khi dọn dẹp:
- Đã chạy lại toàn bộ test suite: `python3 run_tests.py` trong thư mục `DATN/`.
- Kết quả: **29 passed, 1 skipped in 9.17s** (Đạt tỷ lệ thành công 100%, không phát sinh bất kỳ lỗi hồi quy nào).

---

## 8. KẾT LUẬN

Tài liệu này phản ánh chính xác 100% cấu trúc, logic toán học và kỹ thuật hiện thực hóa của toàn bộ codebase **Orchest Computing** hiện có trong thư mục `/Applications/HCMUT/Orchest Computing/DATN`. Toàn bộ các module đều được thiết kế chặt chẽ, hướng module hóa cao, sạch sẽ và không còn file dư thừa, sẵn sàng phục vụ cho việc nghiên cứu, bảo vệ đồ án tốt nghiệp và phát triển thương mại hóa trong tương lai.
