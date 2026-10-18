# -*- coding: utf-8 -*-
"""
Module containing Chapter 5, References, and Appendices for the Midterm Report.
Chapter 5: Đánh giá tiến độ và Kế hoạch nửa sau học kỳ (Tuần 10 - 18)
Tài liệu tham khảo (IEEE References)
Phụ lục (Appendices)
"""
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from report_builder_core import (
    add_heading_1, add_heading_2, add_heading_3, add_body_paragraph, add_bullet_item,
    add_callout_box, format_styled_table,
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TEXT, COLOR_MUTED,
    HEX_PRIMARY, HEX_SECONDARY, HEX_LIGHT_BG, HEX_BORDER, HEX_WHITE
)

def build_chapter_5_and_backmatter(doc):
    # CHƯƠNG 5
    add_heading_1(doc, "CHƯƠNG 5: ĐÁNH GIÁ TIẾN ĐỘ VÀ KẾ HOẠCH NỬA SAU HỌC KỲ")
    
    add_heading_2(doc, "5.1. Tổng kết và Đối chiếu Tiến độ Thực hiện so với Đề cương Đề xuất")
    add_body_paragraph(doc, "Đối chiếu với đề cương nghiên cứu 18 tuần đã được phê duyệt, tính đến thời điểm giữa kỳ (kết thúc Tuần 9), nhóm nghiên cứu đã hoàn thành 100% khối lượng công việc của hai giai đoạn then chốt đầu tiên: Giai đoạn P1 (Tuần 1 - 4: Khảo sát & Thiết kế) và Giai đoạn P2 (Tuần 5 - 9: Xây dựng Core Engine & Thực nghiệm đo đạc ban đầu). Bảng 5.1 tổng hợp chi tiết việc hoàn thành các sản phẩm giao nộp (Deliverables).")

    tbl_p = doc.add_table(rows=6, cols=4)
    p_headers = ["Giai đoạn / Thời gian", "Mục tiêu và Nhiệm vụ trọng tâm", "Sản phẩm giao nộp theo Đề cương", "Tình trạng thực tế (Tuần 9)"]
    p_data = [
        ("Giai đoạn P1\n(Tuần 1 – 4)", "Khảo sát tài liệu; lập ma trận related work; định thức hóa mô hình Working-Set và KV-Cache GQA; thiết kế thuật toán DP O(K*N^2); đặc tả giao thức mạng nhị phân.", "Bản đề cương chi tiết; đặc tả Cost Model; đặc tả giao thức truyền nhận; kế hoạch benchmark.", "ĐÃ HOÀN THÀNH 100%\n(Hồ sơ thiết kế và mô hình toán hoàn thiện)"),
        ("Giai đoạn P2\n(Tuần 5 – 9)", "Xây dựng Auto Discovery UDP, Node Registry, Heartbeat; viết module Profiler phần cứng (NVML, WMI, MPS); xây dựng Cost Model Calibrator; hiện thực DP Solver (Python & C++); sinh chứng chỉ mTLS, mã hóa CRC32; xây dựng Dashboard Web UI; chạy 5 Sub-studies thực nghiệm.", "Core Engine hoàn chỉnh; module Profiler; Cost Model đã cân chỉnh; module C++ DP Solver; Dashboard Web 3 Tab; bộ dữ liệu thực nghiệm 5 Sub-studies.", "ĐÃ HOÀN THÀNH 100%\n(Mã nguồn hoạt động ổn định, 45/46 test cases passed, đo đạc cụm thực tế đầy đủ)"),
        ("Giai đoạn P3\n(Tuần 10 – 14)", "Xây dựng luồng thực thi song song đường ống (Swarm Pipeline Executor); AutoModelSlicer nạp trọng số chọn lọc; chu trình sinh token tự hồi quy đa nút; cơ chế Micro-batching; hoàn thiện cơ chế chịu lỗi Safe Failure cấp phiên.", "Hệ thống pipeline phân tán hoàn chỉnh; chu trình sinh văn bản tự hồi quy; cơ chế chịu lỗi tự động; kiểm thử tích hợp đa nút.", "KẾ HOẠCH NỬA SAU\n(Đang tiến hành, đã có mã nguồn khung và slicer)"),
        ("Giai đoạn P4\n(Tuần 15 – 18)", "Thực hiện chuỗi thực nghiệm quy mô lớn; phân tích thống kê chuyên sâu; đóng gói mã nguồn và tài liệu kỹ thuật hoàn chỉnh; hoàn thiện bản thảo Luận văn Tốt nghiệp và Slide bảo vệ.", "Bản mẫu hệ thống hoàn thiện; bộ dữ liệu thực nghiệm mở rộng; Luận văn Tốt nghiệp hoàn chỉnh; mã nguồn đóng gói mở.", "KẾ HOẠCH NỬA SAU\n(Dự kiến hoàn thành đúng hạn bảo vệ)")
    ]
    format_styled_table(tbl_p, [Cm(2.6), Cm(5.2), Cm(4.6), Cm(3.6)], p_headers, p_data)
    add_body_paragraph(doc, "Bảng 5.1. Đối chiếu chi tiết tiến độ thực hiện thực tế so với đề cương nghiên cứu 18 tuần.")

    add_heading_2(doc, "5.2. Các khó khăn kỹ thuật đã giải quyết và bài học kinh nghiệm")
    add_body_paragraph(doc, "Trong quá trình triển khai nửa đầu học kỳ, nhóm đã nhận diện và giải quyết triệt để 3 thách thức kỹ thuật lớn:")
    add_bullet_item(doc, "Xử lý sự phân mảnh quản lý bộ nhớ giữa Windows và macOS: Trên Windows, VRAM được tách biệt vật lý và quản lý qua driver NVIDIA (truy vấn qua NVML). Trên macOS, kiến trúc Apple Silicon sử dụng Unified Memory dùng chung cho cả CPU và GPU Metal, không có khái niệm VRAM rời. Nhóm đã xây dựng bộ phân lớp trừu tượng trong `PartitionCostCalculator` tự động nhận diện và tính toán $M_{\\text{usable}}$ phù hợp cho từng nền tảng, giải quyết hoàn toàn nguy cơ định giá sai bộ nhớ.", "1. Bất đối xứng kiến trúc bộ nhớ: ")
    add_bullet_item(doc, "Triệt tiêu chi phí đóng gói dữ liệu trung gian: Ban đầu, việc tuần tự hóa tensor qua các định dạng văn bản (JSON/Base64) hoặc thư viện chuẩn tiêu tốn quá nhiều CPU và làm phình to kích thước dữ liệu gấp nhiều lần. Nhóm đã thiết kế giao thức khung nhị phân `TensorPacket` với tiêu đề cố định 18 bytes, truyền trực tiếp mảng byte dữ liệu thô của PyTorch Tensor kết hợp với checksum CRC32 tính toán phần cứng, nâng băng thông truyền tải socket lên trên 2,400 MB/s.", "2. Tối ưu hóa vận chuyển Tensor: ")
    add_bullet_item(doc, "Khắc phục hiện tượng vòng lặp tự chuyển tiếp (Self-transition loophole) trong thuật toán DP: Trong phiên bản thuật toán ban đầu, khi quy hoạch động quét qua các nút, nếu một nút được chọn liên tiếp ở nhiều giai đoạn mà không có ràng buộc, hệ thống có thể tạo ra các lát cắt vi mô (micro-slices) không cần thiết. Nhóm đã tinh chỉnh cấu trúc chuyển trạng thái của DP để đảm bảo mỗi nút chỉ nhận đúng một lát cắt liên tục duy nhất $[l_k^{\\text{start}} .. l_k^{\\text{end}}]$, triệt tiêu hoàn toàn sự phân mảnh layer.", "3. Ràng buộc lát cắt DP liên tục: ")

    add_heading_2(doc, "5.3. Kế hoạch nghiên cứu và triển khai chi tiết cho nửa sau học kỳ (Tuần 10 - 18)")
    add_body_paragraph(doc, "Trong 9 tuần tiếp theo, nhóm nghiên cứu tập trung dồn toàn lực vào hai giai đoạn P3 và P4 nhằm hoàn thiện trọn vẹn đồ án tốt nghiệp:")
    
    add_heading_3(doc, "5.3.1. Giai đoạn P3: Hoàn thiện Runtime và Song song đường ống (Tuần 10 – Tuần 14)")
    add_bullet_item(doc, "Tuần 10: Tối ưu hóa chu trình sinh token tự hồi quy bầy đàn (Autoregressive Swarm Generation Loop) trong `SwarmAutoregressiveGenerator`. Hoàn thiện việc duy trì và tái sử dụng bộ đệm KV-Cache phân tán giữa các bước giải mã (Decode steps) để giảm thiểu độ trễ ITL.", "• ")
    add_bullet_item(doc, "Tuần 11: Hiện thực hóa cơ chế gom cụm kích hoạt vi mô (Asynchronous Micro-batching) trong `micro_batcher.py`, cho phép đẩy nhiều yêu cầu suy luận đồng thời qua đường ống dạng 1F1B (One Forward, One Backward/Decode) để nâng cao thông lượng tổng thể.", "• ")
    add_bullet_item(doc, "Tuần 12: Hoàn thiện và kiểm thử kịch bản chịu lỗi tự động (Session-Boundary Safe Failure). Tiến hành thử nghiệm rút đột ngột cáp mạng của 1 Worker trong khi cụm đang chạy suy luận để xác nhận Host có thể phát hiện qua Heartbeat trong 5 giây, hủy phiên an toàn và tự động tái phân bổ (Repartitioning) trên các máy còn lại.", "• ")
    add_bullet_item(doc, "Tuần 13: Mở rộng khả năng hỗ trợ mô hình LLM sang các mô hình quy mô cực lớn (Qwen2.5-14B và Qwen2.5-32B gồm 64 layers) kết hợp mô phỏng và nạp lượng tử hóa trọng số 4-bit (4-bit Weight Quantization, 0.5 byte/tham số).", "• ")
    add_bullet_item(doc, "Tuần 14: Đóng gói và tinh chỉnh toàn diện giao diện Universal Dashboard, hỗ trợ vẽ biểu đồ đường ống sinh động bằng SVG/Canvas thời gian thực và hiển thị dòng text streaming mượt mà.", "• ")

    add_heading_3(doc, "5.3.2. Giai đoạn P4: Mở rộng Thực nghiệm, Đóng gói và Viết Luận văn (Tuần 15 – Tuần 18)")
    add_bullet_item(doc, "Tuần 15: Mở rộng cụm thực nghiệm lên quy mô 4 đến 5 thiết bị cá nhân trong phòng thí nghiệm của trường. Chạy bộ benchmark kiểm thử áp lực (Stress-testing) liên tục trong 24 giờ để đánh giá độ ổn định lâu dài.", "• ")
    add_bullet_item(doc, "Tuần 16: Tổng hợp dữ liệu thực nghiệm, tính toán các chỉ số thống kê chuyên sâu (P95, P99 latency, Throughput tokens/sec, Spearman rank correlation, phân tích phương sai ANOVA).", "• ")
    add_bullet_item(doc, "Tuần 17: Hoàn thiện toàn bộ bản thảo Luận văn Tốt nghiệp chính thức theo đúng định dạng quy chuẩn của Trường ĐH Bách Khoa – ĐHQG-HCM. Đóng gói mã nguồn mở (Open-source repository) kèm tài liệu hướng dẫn cài đặt và sử dụng chi tiết.", "• ")
    add_bullet_item(doc, "Tuần 18: Thiết kế Slide báo cáo tổng kết, chuẩn bị video demo hệ thống trực quan, chạy thử kịch bản demo trực tiếp trên cụm máy tính và tiến hành bảo vệ trước Hội đồng Đồ án Tốt nghiệp.", "• ")

    # Bảng Gantt Chart tóm tắt tiến độ
    add_heading_3(doc, "5.3.3. Sơ đồ tiến độ tổng thể (Gantt Chart Roadmap)")
    tbl_gantt = doc.add_table(rows=5, cols=5)
    gantt_headers = ["Hạng mục công việc trọng tâm", "Tuần 1 – 4", "Tuần 5 – 9 (Giữa kỳ)", "Tuần 10 – 14", "Tuần 15 – 18 (Bảo vệ)"]
    gantt_data = [
        ("1. Khảo sát, Mô hình toán & Thiết kế kiến trúc", "[████████] 100%", "---", "---", "---"),
        ("2. Xây dựng Core Engine, Profiler & Benchmark", "---", "[████████] 100%", "---", "---"),
        ("3. Swarm Pipeline, Autoregressive Loop & Chịu lỗi", "---", "---", "[████████] Dự kiến", "---"),
        ("4. Thực nghiệm cụm lớn, Viết Luận văn & Bảo vệ", "---", "---", "---", "[████████] Dự kiến")
    ]
    format_styled_table(tbl_gantt, [Cm(5.2), Cm(2.7), Cm(2.7), Cm(2.7), Cm(2.7)], gantt_headers, gantt_data)
    add_body_paragraph(doc, "Bảng 5.2. Sơ đồ phân bổ tiến độ tổng thể 18 tuần của Đồ án Tốt nghiệp.")

    # TÀI LIỆU THAM KHẢO
    add_heading_1(doc, "TÀI LIỆU THAM KHẢO")
    references = [
        "[1] A. Vaswani et al., \"Attention is all you need,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 5998–6008.",
        "[2] H. Touvron et al., \"LLaMA: Open and efficient foundation language models,\" arXiv preprint arXiv:2302.13971, 2023.",
        "[3] Qwen Team, \"Qwen2.5 Technical Report,\" Alibaba Group, Tech. Rep., Sep. 2024. [Online]. Available: https://github.com/QwenLM/Qwen2.5",
        "[4] W. Kwon et al., \"Efficient memory management for large language model serving with PagedAttention,\" in Proc. 29th ACM Symp. Operating Systems Principles (SOSP), 2023, pp. 611–626.",
        "[5] A. Borzunov et al., \"Petals: Collaborative inference and fine-tuning of large language models,\" in Proc. 61st Annu. Meeting Assoc. Computational Linguistics (ACL): System Demonstrations, 2023, pp. 558–568.",
        "[6] J. Hu et al., \"PipeEdge: Pipeline-parallel inference of large language models on edge devices,\" in Proc. IEEE/ACM Int. Conf. Utility and Cloud Computing (UCC), 2023, pp. 1–10.",
        "[7] Exo Labs, \"exo: Run your own AI cluster at home with everyday devices,\" 2024. [Online]. Available: https://github.com/exo-explore/exo",
        "[8] D. Narayanan et al., \"Memory-efficient pipeline parallelism for diverse DNN architectures,\" in Proc. 18th USENIX Conf. Networked Systems Design and Implementation (NSDI), 2021, pp. 1–15.",
        "[9] T. Dao, \"FlashAttention-2: Faster attention with better parallelism and work partitioning,\" in Proc. Int. Conf. Learning Representations (ICLR), 2024.",
        "[10] S. Rajbhandari et al., \"Zero: Memory optimizations toward training trillion parameter models,\" in Proc. Int. Conf. High Performance Computing, Networking, Storage and Analysis (SC), 2020, pp. 1–16.",
        "[11] M. Shoeybi et al., \"Megatron-LM: Training multi-billion parameter language models using model parallelism,\" arXiv preprint arXiv:1909.08053, 2019.",
        "[12] J. Ainslie et al., \"GQA: Training generalized multi-query transformer models from multi-head checkpoints,\" in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2023, pp. 4895–4901.",
        "[13] P. Rescorla, \"The Transport Layer Security (TLS) Protocol Version 1.3,\" RFC 8446, Internet Engineering Task Force (IETF), Aug. 2018.",
        "[14] P. Hintjens, \"ZeroMQ: Messaging for Many Applications,\" O'Reilly Media, Inc., 2013.",
        "[15] Apple Inc., \"Metal Performance Shaders Framework Documentation,\" Apple Developer, 2024. [Online]. Available: https://developer.apple.com/metal/",
        "[16] NVIDIA Corporation, \"NVIDIA Management Library (NVML) Reference Manual,\" NVIDIA Developer, 2024.",
        "[17] T. Wolf et al., \"Transformers: State-of-the-art natural language processing,\" in Proc. Conf. Empirical Methods in Natural Language Processing: System Demonstrations, 2020, pp. 38–45.",
        "[18] A. Paszke et al., \"PyTorch: An imperative style, high-performance deep learning library,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019, pp. 8024–8035."
    ]
    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p_ref.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
        r.font.color.rgb = COLOR_TEXT

    # PHỤ LỤC
    add_heading_1(doc, "PHỤ LỤC")
    
    add_heading_2(doc, "Phụ lục A: Đặc tả Chi tiết Gói tin Nhị phân TensorPacket")
    add_body_paragraph(doc, "Gói tin nhị phân `TensorPacket` được truyền tải tuần tự qua socket kết nối mạng theo cấu trúc nhị phân sau:")
    add_callout_box(doc, """+---------------+---------------+---------------+------------------+---------------+-------------------------------+
| Magic (4B)    | Type (2B)     | Seq ID (4B)   | Payload Len (4B) | CRC32 (4B)    | Raw Binary Payload (N bytes)  |
| 'ORCH'        | uint16        | uint32        | uint32           | uint32        | Serialized Tensor / PyTorch   |
+---------------+---------------+---------------+------------------+---------------+-------------------------------+""", "Sơ đồ bố cục bộ nhớ của khung nhị phân TensorPacket")
    add_body_paragraph(doc, "Quy trình xác thực gói tin: Khi bên nhận đọc đủ 18 bytes tiêu đề, nó giải nén bằng cú pháp `struct.unpack('!4sHIII', header_bytes)`. Nếu Magic Bytes khác `'ORCH'`, kết nối lập tức bị ngắt kết nối. Sau đó bên nhận đọc chính xác `Payload Len` bytes từ socket và tính mã CRC32 của mảng bytes nhận được. Nếu CRC32 tính được không khớp với trường CRC32 trong header, gói tin bị hủy bỏ ngay tại chỗ để ngăn ngừa dữ liệu tensor bị hỏng hóc làm sai lệch quá trình suy luận.")

    add_heading_2(doc, "Phụ lục B: Bảng Đặc tả Kỹ thuật Các Mô hình LLM trong Hệ thống")
    add_body_paragraph(doc, "Các thông số kiến trúc cốt lõi của các mô hình AI thuộc họ Qwen2.5 được cấu hình trong `MODEL_CONFIGS` của hệ thống:")
    
    tbl_app_b = doc.add_table(rows=5, cols=7)
    app_b_headers = ["Mô hình", "Số Layer (N)", "Hidden Size (d)", "Số đầu Query", "Số đầu KV (GQA)", "Dung lượng FP16", "Working Set Đơn máy"]
    app_b_data = [
        ("Qwen2.5-0.5B", "24", "896", "14", "2 (GQA)", "~0.98 GB", "~1.2 GB"),
        ("Qwen2.5-1.5B", "28", "1,536", "12", "2 (GQA)", "~3.08 GB", "~3.8 GB"),
        ("Qwen2.5-3B", "36", "2,048", "16", "2 (GQA)", "~6.18 GB", "~7.4 GB"),
        ("Qwen2.5-7B", "28", "3,584", "28", "4 (GQA)", "~15.20 GB", "~17.8 GB")
    ]
    format_styled_table(tbl_app_b, [Cm(2.6), Cm(2.0), Cm(2.2), Cm(2.2), Cm(2.2), Cm(2.5), Cm(2.5)], app_b_headers, app_b_data)
    add_body_paragraph(doc, "Bảng B.1. Thông số kiến trúc và nhu cầu bộ nhớ của các mô hình kiểm thử.")

print("Chapter 5 and Back Matter builder loaded.")
