# -*- coding: utf-8 -*-
"""
Module containing Chapter 3 for the Midterm Report.
Chapter 3: Thiết kế kiến trúc và Các giải pháp kỹ thuật hệ thống
"""
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from report_builder_core import (
    add_heading_1, add_heading_2, add_heading_3, add_body_paragraph, add_bullet_item,
    add_callout_box, format_styled_table,
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TEXT, COLOR_MUTED,
    HEX_PRIMARY, HEX_SECONDARY, HEX_LIGHT_BG, HEX_BORDER, HEX_WHITE
)

def build_chapter_3(doc):
    add_heading_1(doc, "CHƯƠNG 3: THIẾT KẾ KIẾN TRÚC VÀ CÁC GIẢI PHÁP KỸ THUẬT HỆ THỐNG")
    
    add_heading_2(doc, "3.1. Kiến trúc tổng thể hệ thống Orchest Computing")
    add_body_paragraph(doc, "Orchest Computing được thiết kế theo mô hình điều phối phân tán Orchestrator – Worker (Host Master – Edge Worker Nodes), tối ưu hóa đặc biệt cho mạng nội bộ không đồng nhất. Hệ thống phân định rạch ròi trách nhiệm giữa nút trưởng (Host) và các nút thành viên (Worker) để tối thiểu hóa phụ tải quản lý:")
    add_bullet_item(doc, "Host Master (Điều phối viên trung tâm): Đóng vai trò là nút chỉ huy của cụm, chịu trách nhiệm lắng nghe phản hồi UDP beacon, quản lý sổ đăng ký thiết bị (Node Registry), thu thập telemetry, hiệu chuẩn Cost Model, chạy thuật toán Quy hoạch động để tính toán kế hoạch cắt lớp, và quản lý giao diện Dashboard điều khiển tập trung.", "1. Host Master: ")
    add_bullet_item(doc, "Worker Nodes (Thiết bị biên thực thi): Đóng vai trò là các máy gia tốc tính toán tại biên. Mỗi Worker chạy một tiến trình nền gọn nhẹ, tự động phát beacon báo danh, chấp nhận/từ chối phân đoạn layer được Host phân bổ qua cơ chế xác thực, nạp lát cắt mô hình vào bộ nhớ GPU/MPS, và tham gia vào chuỗi pipeline truyền nhận tensor nhị phân qua socket.", "2. Worker Nodes: ")
    add_body_paragraph(doc, "Điểm độc đáo của kiến trúc là tính 'Universal' (Vạn năng): Mã nguồn của Orchest Computing được đóng gói thành một ứng dụng duy nhất (`apps/orchest.py`). Bất kỳ máy tính nào khi khởi động cũng có thể linh hoạt đóng vai trò Host Master hoặc Worker Node chỉ thông qua một nút chuyển đổi trên giao diện Web UI.")

    add_heading_2(doc, "3.2. Phân hệ Tự động Phát hiện và Quản trị Cụm (UDP Beacon & Fleet Management)")
    add_body_paragraph(doc, "Để loại bỏ hoàn toàn việc người dùng phải cấu hình địa chỉ IP thủ công phức tạp giữa các máy tính, Orchest Computing hiện thực hóa giao thức khám phá tự động không cấu hình (Zero-Configuration Auto-Discovery) dựa trên cơ chế UDP Broadcast:")
    add_bullet_item(doc, "Cơ chế phát sóng: Các Worker Node khi khởi chạy sẽ mở một cổng UDP lắng nghe tại địa chỉ `0.0.0.0:50005`. Khi Host Master bắt đầu quét mạng (qua lệnh Scan trên Dashboard), nó phát đi một thông điệp truy vấn dạng JSON tới địa chỉ quảng bá `255.255.255.255`.", "• Phát sóng Broadcast: ")
    add_bullet_item(doc, "Đăng ký địa chỉ thực: Để tránh lỗi Worker gửi địa chỉ cục bộ `127.0.0.1`, Host Master trích xuất trực tiếp địa chỉ IP thực từ gói tin socket nhận được (`addr[0]`), đảm bảo định danh chính xác 100% trong mạng LAN.", "• Định danh IP thực: ")
    add_bullet_item(doc, "Quản trị Fleet linh hoạt: Người dùng có thể đặt tên thân thiện (Custom Friendly Alias) cho từng máy tính (ví dụ 'Laptop-Gaming-RTX4070', 'MacBook-M2Pro-HocTap'), bật/tắt (Enable/Disable) sự tham gia của từng máy trong bầy đàn (Swarm), và lưu trữ cấu hình bền vững vào file `cluster_nodes_config.json`.", "• Quản trị Fleet: ")
    add_bullet_item(doc, "Cơ chế Ủy quyền và Phê duyệt (Worker Authorization): Khi Host Master phân bổ lát cắt mô hình, một thông báo trực quan (Popup Banner) sẽ hiển thị trên giao diện của Worker kèm thông tin chi tiết (Tên mô hình, Dải layer phân bổ, IP Host). Người dùng tại Worker có thể bấm 'Accept' để tiếp nhận hoặc 'Reject' để từ chối tải nặng.", "• Worker Authorization: ")

    add_heading_2(doc, "3.3. Phân hệ Đo đạc Động và Đánh giá Phần cứng (Dynamic Profiling & Telemetry)")
    add_body_paragraph(doc, "Để cung cấp các tham số đầu vào chuẩn xác cho Cost Model, Orchest Computing trang bị phân hệ Profiler chuyên sâu gồm 4 thành phần:")
    add_bullet_item(doc, "Node Hardware Profiler (`orchest.profiler.node_profiler`): Tự động nhận diện hệ điều hành (Windows/macOS/Linux), số nhân CPU vật lý và logic, xung nhịp CPU, dung lượng RAM tổng và RAM khả dụng qua thư viện `psutil` và Windows WMI. Trên máy Windows có GPU NVIDIA, module tích hợp `pynvml` để đọc trực tiếp VRAM khả dụng, nhiệt độ lõi và chỉ số Compute Capability. Trên máy macOS, module truy vấn trực tiếp Apple System APIs để xác định kiến trúc chip Apple Silicon và Unified Memory.", "1. Nhận diện phần cứng: ")
    add_bullet_item(doc, "Hardware Benchmark (`orchest.profiler.hardware_benchmark`): Chạy một chuỗi các phép nhân ma trận GEMM chuẩn hóa trên backend phần cứng mục tiêu (CUDA hoặc MPS) ở độ chính xác FP16 để đo đạc công suất tính toán hiệu dụng thực tế (Effective TFLOPs). Điều này loại bỏ hoàn toàn sai số của các thông số TFLOPs lý thuyết trên tài liệu quảng cáo của nhà sản xuất.", "2. Đo TFLOPs thực tế: ")
    add_bullet_item(doc, "Network Profiler (`orchest.profiler.network_profiler`): Thực hiện đo đạc độ trễ khứ hồi (RTT) qua cơ chế ping socket và đo băng thông thông lượng thực tế (Throughput in MB/s) bằng cách truyền một luồng payload thử nghiệm giữa Host và từng Worker.", "3. Đo đạc mạng LAN: ")
    add_bullet_item(doc, "Kiểm tra Tính đúng đắn số học (`orchest.profiler.correctness`): So sánh vector kích hoạt hoặc logits sinh ra bởi mô hình cắt phân tán so với mô hình chạy đơn máy qua chỉ số Độ tương đồng Cosine (Cosine Similarity) và phân kỳ KL Divergence để đảm bảo độ chính xác tuyệt đối.", "4. Kiểm thử số học: ")

    add_heading_2(doc, "3.4. Phân hệ Giao thức Mạng và Vận chuyển Tensor Nhị phân Zero-Overhead")
    add_body_paragraph(doc, "Vận chuyển tensor kích hoạt trung gian là mắt xích quyết định hiệu năng của toàn bộ đường ống. Orchest Computing thiết kế giao thức khung nhị phân tùy biến mang tên `TensorPacket` với cấu trúc tiêu đề (Header) cố định siêu nhẹ chỉ 18 bytes:")
    
    # Bảng cấu trúc Header
    tbl_hdr = doc.add_table(rows=6, cols=4)
    hdr_headers = ["Trường dữ liệu (Field)", "Kích thước (Bytes)", "Định dạng đóng gói (Struct)", "Ý nghĩa và Chức năng"]
    hdr_data = [
        ("Magic Bytes", "4 bytes", "4s (Giá trị: 'ORCH')", "Xác thực định danh gói tin hợp lệ của Orchest Computing"),
        ("Packet Type", "2 bytes", "H (unsigned short)", "Phân loại gói tin: TENSOR_DATA (1), HANDSHAKE (2), HEARTBEAT (3)"),
        ("Sequence ID", "4 bytes", "I (unsigned int)", "Số thứ tự gói tin trong phiên suy luận để chống mất mát"),
        ("Payload Length", "4 bytes", "I (unsigned int)", "Độ dài chính xác của vùng dữ liệu nhị phân đi kèm phía sau"),
        ("CRC32 Checksum", "4 bytes", "I (unsigned int)", "Mã kiểm tra toàn vẹn dữ liệu nhị phân chống nhiễu đường truyền"),
        ("TỔNG TIÊU ĐỀ", "18 BYTES", "Cấu trúc: '!4sHIII'", "Tiêu đề siêu nhẹ, phân tích cú pháp (parse) trong O(1) thời gian")
    ]
    format_styled_table(tbl_hdr, [Cm(3.5), Cm(2.8), Cm(3.2), Cm(6.5)], hdr_headers, hdr_data)
    add_body_paragraph(doc, "Bảng 3.1. Cấu trúc khung nhị phân chuẩn 18 bytes của TensorPacket.")

    add_body_paragraph(doc, "Kiến trúc giao vận mạng tích hợp 3 tùy chọn transport thông qua lớp trừu tượng `TransportManager`:")
    add_bullet_item(doc, "Raw TCP Socket (Mặc định): Sử dụng trực tiếp socket TCP nhị phân không qua trung gian, đọc trực tiếp tensor buffer vào vùng nhớ đệm, đạt thông lượng cực đại > 2,200 MB/s trên kênh truyền nội bộ.", "• ")
    add_bullet_item(doc, "gRPC Transport: Đóng gói tensor theo chuẩn Protocol Buffers (`tensor_service.proto`), hỗ trợ đa nền tảng và dễ dàng mở rộng sang các ngôn ngữ khác.", "• ")
    add_bullet_item(doc, "ZeroMQ Transport: Sử dụng mẫu tin nhắn PUSH/PULL với mã hóa CurveZMQ tốc độ cao.", "• ")
    add_body_paragraph(doc, "Bảo mật cấp biên: Toàn bộ kết nối giữa Host và Worker được bảo vệ bằng cơ chế xác thực hai chiều (Mutual TLS - mTLS) dựa trên chứng chỉ số X.509 (`ca.crt`, `host.crt`, `worker.crt`) tự động sinh qua script `scripts/generate_certs.py`, kết hợp mã hóa đối xứng AES-GCM 128-bit.")

    add_heading_2(doc, "3.5. Module Tăng tốc Quy hoạch Động bằng C++ (dp_solver.cpp)")
    add_body_paragraph(doc, "Nhằm đáp ứng yêu cầu tính toán tức thời trong môi trường biên khi quy mô mô hình lên tới hàng chục layer (ví dụ Qwen2.5-32B có 64 layers) và cụm máy tính nhiều hơn 10 nút, nhóm đã hiện thực hóa thuật toán DP bằng ngôn ngữ C++ thuần túy (`DATN/core_engine/dp_solver.cpp`).")
    add_body_paragraph(doc, "Module C++ được biên dịch thành thư viện liên kết động (`.dll` trên Windows và `.dylib` trên macOS) và tích hợp vào Python qua giao diện `ctypes`. Thuật toán C++ tận dụng cấu trúc mảng phẳng liên tục (Flat 1D arrays) để tối ưu hóa bộ nhớ đệm CPU L1/L2 và áp dụng song song hóa OpenMP trên vòng lặp quét layer, giúp giảm thời gian giải quy hoạch động từ 15 ms (Python thuần) xuống dưới 0.3 ms.")

    add_heading_2(doc, "3.6. Module Trích xuất và Phân cắt Lớp Mô hình Linh hoạt (AutoModelSlicer)")
    add_body_paragraph(doc, "Module `AutoModelSlicer` (`DATN/orchest/models/auto_slicer.py`) chịu trách nhiệm mổ xẻ cấu trúc của mô hình Transformer thành các phân đoạn layer độc lập:")
    add_bullet_item(doc, "Cơ chế Reflection vạn năng: Module tự động kiểm tra thuộc tính của đối tượng `PreTrainedModel` từ thư viện Hugging Face Transformers để trích xuất danh sách layer (như `.model.layers` đối với LLaMA/Qwen, `.transformer.h` đối với GPT-2/Falcon).", "• ")
    add_bullet_item(doc, "Đóng gói lớp `ModelSliceModule`: Kế thừa chuẩn `torch.nn.Module`. Lớp này đóng gói danh sách các layer được giao cùng với Embedding (nếu là nút đầu) hoặc RMSNorm và LM-Head (nếu là nút cuối). Hàm `forward()` xử lý thông minh cả đầu vào dạng Token ID lẫn Tensor ẩn trung gian.", "• ")
    add_bullet_item(doc, "Kỹ thuật Selective Weight Sharded Loading: Thay vì bắt Worker phải tải toàn bộ mô hình hàng chục GB về ổ cứng rồi mới cắt, module hỗ trợ cơ chế nạp chọn lọc (Selective Loading) chỉ nạp đúng các tham số trọng số của các layer được giao vào VRAM, tiết kiệm triệt để băng thông tải và dung lượng lưu trữ.", "• ")

    add_heading_2(doc, "3.7. Kiến trúc An toàn và Chịu lỗi Cấp Phiên (Safe Failure)")
    add_body_paragraph(doc, "Thiết bị biên cá nhân có đặc tính kém ổn định hơn máy chủ (người dùng có thể gập nắp laptop, rút cáp sạc, hoặc ngắt kết nối Wi-Fi bất ngờ). Orchest Computing áp dụng chiến lược chịu lỗi cấp phiên (Session-Boundary Safe Failure):")
    add_bullet_item(doc, "Giám sát Heartbeat liên tục: Worker định kỳ gửi tín hiệu sống (Keep-alive Heartbeat) mỗi 2 giây. Nếu Host không nhận được tín hiệu sau 5 giây (Timeout threshold), nút đó sẽ bị đánh dấu là `DEAD`.", "• ")
    add_bullet_item(doc, "Cô lập sự cố an toàn (Safe Abort): Khi một nút gặp sự cố trong khi đang xử lý một phiên suy luận, Host lập tức hủy yêu cầu hiện tại an toàn, giải phóng bộ đệm mạng để tránh treo hệ thống.", "• ")
    add_bullet_item(doc, "Tái phân bổ động (Session Repartitioning): Host tự động loại bỏ nút lỗi khỏi danh sách cụm, kích hoạt thuật toán DP tính toán lại phương án phân hoạch tối ưu mới trên tập các nút còn sống, và phục vụ phiên tiếp theo mà không cần khởi động lại hệ thống.", "• ")

    add_heading_2(doc, "3.8. Giao diện Người dùng Hợp nhất (Universal Dashboard Web Application)")
    add_body_paragraph(doc, "Phân hệ Dashboard (`DATN/apps/dashboard/`) được phát triển hiện đại với Backend FastAPI bất đồng bộ và giao diện Frontend Glassmorphism sang trọng với 3 Tab chức năng chính:")
    add_bullet_item(doc, "Tab 1 (Host Master Mode): Cho phép quét mạng (Scan Network), hiển thị danh sách các máy tính trong cụm kèm trạng thái phần cứng (CPU, RAM, GPU, VRAM, Nhiệt độ), lựa chọn mô hình LLM, kích hoạt thuật toán Auto DP Solver hoặc điều chỉnh thanh trượt tỉ lệ (Ratio Slicing Bar). Cung cấp khung trò chuyện Playground để gửi prompt và nhận kết quả sinh văn bản thời gian thực.", "1. Host Mode Tab: ")
    add_bullet_item(doc, "Tab 2 (Worker Edge Mode): Giao diện dành riêng cho máy tính hoạt động ở chế độ Worker. Hiển thị thông số tài nguyên nội bộ, danh sách các phân đoạn layer được Host phân bổ, popup phê duyệt Accept/Reject, và nhật ký kết nối chi tiết (Connection History Logs).", "2. Worker Mode Tab: ")
    add_bullet_item(doc, "Tab 3 (Manage Fleet Tab): Bảng điều khiển quản trị toàn cụm. Cho phép đổi tên máy tính thân thiện, kiểm tra độ trễ ping chẩn đoán (Latency Diagnostics) tới từng nút, bật/tắt quyền tham gia Swarm của từng máy, và loại bỏ các nút không còn hoạt động.", "3. Fleet Tab: ")
    add_body_paragraph(doc, "Điểm nhấn trải nghiệm người dùng: Thanh phân bổ lát cắt tự động cân bằng (Auto-Balancing Slicing Bar $\\sum = 100\\%$). Khi người dùng kéo thanh trượt của một máy tính bất kỳ, hệ thống sẽ tự động tính toán bù trừ tỉ lệ của các máy còn lại để bảo đảm tổng số layer phân bổ luôn khớp chính xác 100% với cấu trúc mô hình.")
    
    doc.add_page_break()

print("Chapter 3 builder loaded.")
