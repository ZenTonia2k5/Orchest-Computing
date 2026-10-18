# -*- coding: utf-8 -*-
"""
Module containing Chapter 1 and Chapter 2 for the Midterm Report.
Chapter 1: Tổng quan và Tính cấp thiết của đề tài
Chapter 2: Cơ sở lý thuyết và Mô hình toán học
"""
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from report_builder_core import (
    add_heading_1, add_heading_2, add_heading_3, add_body_paragraph, add_bullet_item,
    add_callout_box, format_styled_table,
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TEXT, COLOR_MUTED,
    HEX_PRIMARY, HEX_SECONDARY, HEX_LIGHT_BG, HEX_BORDER, HEX_WHITE
)

def build_chapter_1(doc):
    add_heading_1(doc, "CHƯƠNG 1: TỔNG QUAN VÀ TÍNH CẤP THIẾT CỦA ĐỀ TÀI")
    
    add_heading_2(doc, "1.1. Bối cảnh nghiên cứu: Sự bùng nổ của Mô hình Ngôn ngữ Lớn và bài toán suy luận tại biên")
    add_body_paragraph(doc, "Trong những năm gần đây, sự đột phá của Trí tuệ Nhân tạo Tạo sinh (Generative AI), đặc biệt là các Mô hình Ngôn ngữ Lớn (Large Language Models - LLMs) như GPT-4, LLaMA-3, Qwen-2.5, Mistral và DeepSeek, đã tạo ra một cuộc cách mạng sâu sắc trong xử lý ngôn ngữ tự nhiên, lập trình tự động, phân tích tài liệu và hỗ trợ ra quyết định. Tuy nhiên, phần lớn các mô hình này hiện nay đều được triển khai tập trung trên các hạ tầng đám mây (Cloud Datacenters) quy mô lớn được trang bị các cụm siêu máy chủ GPU NVIDIA H100, A100 chuyên dụng.")
    add_body_paragraph(doc, "Mặc dù mô hình điện toán đám mây mang lại khả năng xử lý mạnh mẽ, nó đang bộc lộ những rào cản và nguy cơ nghiêm trọng trong thực tế ứng dụng:")
    add_bullet_item(doc, "Vi phạm quyền riêng tư và an toàn dữ liệu: Dữ liệu nhạy cảm của cá nhân, doanh nghiệp hoặc cơ quan chính phủ phải gửi trực tiếp lên máy chủ bên thứ ba, tiềm ẩn nguy cơ rò rỉ thông tin mật và vi phạm các quy định bảo vệ dữ liệu nghiêm ngặt như GDPR.", "1. Rủi ro quyền riêng tư: ")
    add_bullet_item(doc, "Chi phí vận hành và bản quyền đắt đỏ: Chi phí thuê máy chủ đám mây hoặc mua token API tăng theo cấp số nhân khi lưu lượng truy cập gia tăng, trở thành gánh nặng tài chính khổng lồ cho các doanh nghiệp vừa và nhỏ cũng như giới nghiên cứu.", "2. Gánh nặng chi phí: ")
    add_bullet_item(doc, "Sự phụ thuộc tuyệt đối vào kết nối Internet: Hệ thống đám mây không thể hoạt động tại các môi trường mạng bị cô lập (air-gapped networks), các vùng sâu vùng xa, tàu biển ngoài khơi, hoặc trong các tình huống ứng cứu thảm họa khi kết nối viễn thông bị gián đoạn.", "3. Tính sẵn sàng kết nối: ")
    add_body_paragraph(doc, "Chính vì vậy, xu hướng chuyển dịch tính toán AI về thiết bị biên (Edge Computing) và mạng cục bộ (Local Area Network - LAN) đang trở thành một trọng tâm nghiên cứu mang tính chiến lược toàn cầu. Việc tận dụng trực tiếp năng lực phần cứng sẵn có tại các văn phòng, phòng thí nghiệm trường học hoặc hộ gia đình (bao gồm máy tính xách tay cá nhân, máy tính để bàn văn phòng) để chạy suy luận mô hình ngôn ngữ lớn là một hướng tiếp cận đầy hứa hẹn, giải quyết triệt để bài toán bảo mật và chi phí.")

    add_heading_2(doc, "1.2. Thách thức cốt lõi trong suy luận phân tán tại biên")
    add_body_paragraph(doc, "Dù mang lại nhiều tiềm năng to lớn, việc hiện thực hóa suy luận LLMs trên các cụm thiết bị biên cá nhân phải đối mặt với 4 thách thức kỹ thuật cơ bản sau:")
    
    add_heading_3(doc, "1.2.1. \"Bức tường bộ nhớ\" (The Memory Wall) và giới hạn VRAM thiết bị biên")
    add_body_paragraph(doc, "Mô hình ngôn ngữ lớn đòi hỏi dung lượng bộ nhớ khổng lồ để lưu trữ các tham số trọng số (Weights) và duy trì trạng thái ngữ cảnh suy luận (KV-Cache). Một mô hình ở quy mô phổ biến hiện nay như Qwen2.5-7B khi nạp ở định dạng bán chính xác (FP16/BF16) cần tối thiểu 14 GB bộ nhớ chỉ riêng cho trọng số; đối với mô hình Qwen2.5-14B hoặc 32B, con số này tăng vọt lên tương ứng 28 GB và 64 GB.")
    add_body_paragraph(doc, "Trong khi đó, hầu hết máy tính cá nhân của người dùng biên chỉ sở hữu card đồ họa rời phổ thông với dung lượng VRAM rất hạn chế: thông thường từ 4 GB (RTX 3050) đến 8 GB (RTX 4060/4070 Laptop), hoặc chia sẻ bộ nhớ hệ thống. Hiện tượng này tạo nên một rào cản tuyệt đối: một thiết bị biên đơn lẻ hoàn toàn không thể nạp đủ mô hình vào bộ nhớ, dẫn đến sự cố tràn bộ nhớ (Out-Of-Memory - OOM) ngay tại bước khởi tạo.")

    add_heading_3(doc, "1.2.2. Tính dị chủng sâu sắc về phần cứng (Hardware Heterogeneity)")
    add_body_paragraph(doc, "Khác với các trung tâm dữ liệu đồng nhất (nơi hàng nghìn GPU cùng loại được kết nối qua bus NVLink tốc độ cao hàng trăm GB/s), cụm thiết bị biên trong mạng LAN gia đình hoặc văn phòng có tính dị chủng cực kỳ phức tạp:")
    add_bullet_item(doc, "Hệ điều hành và nền tảng máy gia tốc: Sự pha trộn giữa máy tính Windows trang bị GPU NVIDIA rời sử dụng nền tảng tính toán CUDA, máy tính Mac của Apple trang bị chip SoC Apple Silicon (M1/M2/M3) sử dụng kiến trúc Bộ nhớ Hợp nhất (Unified Memory) và framework Metal Performance Shaders (MPS), cùng các máy tính chỉ chạy CPU x86-64 hoặc ARM.", "• Nền tảng dị chủng: ")
    add_bullet_item(doc, "Sự chênh lệch công suất tính toán: Một máy trạm có thể đạt hiệu năng 12.0 TFLOPs, trong khi một máy văn phòng lân cận chỉ đạt 2.0 TFLOPs hoặc chạy CPU thuần. Nếu phân bổ tải tính toán không cân đối, hiện tượng nút nghẽn cổ chai (Straggler problem) sẽ xảy ra, kéo sập thông lượng của toàn bộ hệ thống.", "• Chênh lệch tính toán: ")

    add_heading_3(doc, "1.2.3. Sự bất đối xứng và biến động của môi trường mạng nội bộ (LAN / Wi-Fi)")
    add_body_paragraph(doc, "Các thiết bị biên liên kết với nhau qua mạng cáp Ethernet gia dụng hoặc sóng Wi-Fi thương mại. Băng thông khả dụng thường dao động trong khoảng 100 Mbps đến 1 Gbps (thấp hơn hàng trăm lần so với băng thông bus nội bộ máy chủ), và độ trễ khứ hồi (Round Trip Time - RTT) có thể biến động từ 0.5 ms đến trên 20 ms. Do đó, bất kỳ mô hình phân tán nào làm phát sinh quá nhiều lưu lượng trao đổi tensor qua mạng đều sẽ thất bại vì chi phí truyền thông (Communication Overhead) sẽ áp đảo hoàn toàn thời gian tính toán.")

    add_heading_3(doc, "1.2.4. Bản chất phân kỳ giữa hai pha suy luận: Prefill và Decode")
    add_body_paragraph(doc, "Quá trình suy luận LLM không phải là một luồng tính toán đồng nhất, mà tách biệt thành hai pha có bản chất tính toán hoàn toàn trái ngược nhau:")
    add_bullet_item(doc, "Pha nạp ngữ cảnh (Prefill Phase): Xử lý toàn bộ prompt đầu vào gồm hàng trăm hoặc hàng nghìn tokens cùng một lúc. Pha này có bản chất Compute-Bound (đòi hỏi công suất tính toán TFLOPs cao), quyết định trực tiếp độ trễ sinh token đầu tiên (Time to First Token - TTFT).", "• Pha Prefill: ")
    add_bullet_item(doc, "Pha sinh tự hồi quy (Decode Phase): Sinh tuần tự từng token một theo cơ chế tự hồi quy. Mỗi bước sinh token chỉ nạp 1 vector kích hoạt nhưng phải quét qua toàn bộ trọng số mô hình và đọc KV-cache, khiến pha này có bản chất Memory-Bandwidth-Bound và cực kỳ nhạy cảm với độ trễ mạng khứ hồi (RTT), quyết định độ trễ giữa các token (Inter-Token Latency - ITL).", "• Pha Decode: ")

    add_heading_2(doc, "1.3. Khảo sát các công trình liên quan và khoảng trống nghiên cứu (Related-Work Analysis)")
    add_body_paragraph(doc, "Để giải quyết bài toán suy luận phân tán, cộng đồng khoa học đã đề xuất một số hệ thống tiêu biểu. Bảng 1.1 tổng hợp ma trận phân tích so sánh toàn diện giữa Orchest Computing và các giải pháp tiên tiến nhất hiện nay.")

    table_comp = doc.add_table(rows=7, cols=6)
    comp_headers = ["Tiêu chí phân tích", "PipeEdge", "exo", "Petals", "PiPPy (PyTorch)", "Orchest Computing"]
    comp_data = [
        ("Mục tiêu tối ưu hóa chính", "Phân hoạch nhận thức tài nguyên biên", "Phân bổ động theo topo và bộ nhớ", "Chia sẻ suy luận bầy đàn (Swarm)", "Thực thi pipeline runtime cơ bản", "Orchestrator biên dị chủng có Cost Model đo lường"),
        ("Tách biệt Prefill / Decode", "Suy luận tổng quát, chưa tách biệt", "Nhận thức thực thi chung", "Tối ưu thông lượng phục vụ đám đông", "Chỉ thực thi pipeline tĩnh", "Tách biệt rõ ràng: Prefill tối ưu TTFT, Decode đo ITL"),
        ("Mô hình động KV-Cache", "Chưa đưa vào hàm mục tiêu", "Nhận thức tài nguyên tĩnh", "Lưu trạng thái phân tán", "Trạng thái runtime cơ bản", "Mô hình hóa động tương thích kiến trúc GQA"),
        ("Hỗ trợ dị chủng sâu sắc", "Tính toán, bộ nhớ, mạng mô phỏng", "Apple Silicon và thiết bị cá nhân", "Cộng đồng tình nguyện heterogeneous", "Tập trung cụm máy chủ đồng nhất", "Windows CUDA + macOS MPS đo đạc thực tế Telemetry"),
        ("Thứ tự nút & Phân bổ", "Theo cấu hình hệ thống", "Tự động theo đồ thị topo", "Định tuyến động theo mạng P2P", "Gán tĩnh theo stage", "Thứ tự tối ưu trong cụm, phân rã Contiguous DP O(K*N^2)"),
        ("Đánh giá độ tin cậy", "Độ trễ và thông lượng hệ thống", "Thông lượng và độ trễ đo đạc", "Thông lượng bầy đàn quy mô lớn", "Bubble pipeline và vết thực thi", "Sai số Cost Model (MAPE/RMSE) & Measured Oracle Regret")
    ]
    format_styled_table(table_comp, [Cm(2.6), Cm(2.4), Cm(2.4), Cm(2.4), Cm(2.4), Cm(3.0)], comp_headers, comp_data)

    add_body_paragraph(doc, "Bảng 1.1. Ma trận so sánh toàn diện giữa Orchest Computing và các công trình liên quan.")
    add_body_paragraph(doc, "Từ bảng phân tích trên, nhóm nghiên cứu nhận diện các khoảng trống nghiên cứu (Research Gaps) lớn mà các hệ thống hiện tại chưa giải quyết triệt để:")
    add_bullet_item(doc, "Khoảng trống 1 (Thiếu mô hình bộ nhớ động nhận thức GQA): Hầu hết các hệ thống phân hoạch layer hiện tại chỉ xem xét kích thước trọng số tĩnh của mô hình mà bỏ qua dung lượng KV-cache phình to nhanh chóng khi context length tăng dài, dẫn đến nguy cơ sập OOM bất ngờ trong pha Decode.", "1. Bỏ quên KV-Cache động: ")
    add_bullet_item(doc, "Khoảng trống 2 (Không phân tách bản chất toán học hai pha): Việc gộp chung pha Prefill và Decode vào một hàm chi phí duy nhất khiến các thuật toán phân bổ đưa ra quyết định sai lầm, bởi vì pha Prefill bị nghẽn bởi TFLOPs còn pha Decode bị nghẽn bởi băng thông bộ nhớ và RTT mạng.", "2. Sai lệch bản chất hai pha: ")
    add_bullet_item(doc, "Khoảng trống 3 (Thiếu đánh giá độ hối tiếc so với nghiệm chuẩn): Các nghiên cứu trước chỉ so sánh hệ thống của mình với các Heuristics đơn giản, hoàn toàn thiếu vắng việc đối chiếu với Nghiệm tối ưu tuyệt đối (Ground Truth Measured Oracle) để định lượng độ hối tiếc (Regret) của thuật toán phân hoạch.", "3. Thiếu chuẩn mực đánh giá: ")

    add_heading_2(doc, "1.4. Mục tiêu nghiên cứu và định vị đóng góp mới của đề tài")
    add_heading_3(doc, "1.4.1. Mục tiêu tổng quát và mục tiêu cụ thể")
    add_body_paragraph(doc, "Mục tiêu tổng quát của đề tài là xây dựng hoàn chỉnh hệ thống Orchest Computing — một nền tảng điều phối (Orchestrator) gọn nhẹ, an toàn và tối ưu cho phép liên kết các máy tính cá nhân dị chủng (Windows CUDA và macOS Apple Silicon MPS) trong mạng LAN thành một cụm suy luận LLM phân tán hiệu năng cao.")
    add_body_paragraph(doc, "Các mục tiêu cụ thể được xác định rõ ràng bao gồm:")
    add_bullet_item(doc, "Thiết lập mô hình toán học dự báo chính xác độ trễ pha Prefill và dung lượng Working Set bộ nhớ.", "• Mục tiêu 1 (Toán học): ")
    add_bullet_item(doc, r"Thiết kế và hiện thực hóa thuật toán Quy hoạch động (DP) phân hoạch lát cắt layer liên tục $O(K \cdot N^2)$.", "• Mục tiêu 2 (Thuật toán): ")
    add_bullet_item(doc, "Hiện thực hóa tầng giao vận nhị phân Zero-Overhead bảo mật mTLS/CurveZMQ và toàn vẹn CRC32.", "• Mục tiêu 3 (Giao vận mạng): ")
    add_bullet_item(doc, "Xây dựng ứng dụng Universal App (Web Dashboard) hỗ trợ tự phát hiện cụm và quản trị trực quan.", "• Mục tiêu 4 (Ứng dụng): ")
    add_bullet_item(doc, "Thực hiện đo đạc thực nghiệm chuẩn mực trên cụm vật lý thực tế với 5 Sub-studies và đánh giá Partition Regret.", "• Mục tiêu 5 (Thực nghiệm): ")

    add_heading_3(doc, "1.4.2. Định vị đóng góp mới (Novel Contributions)")
    add_body_paragraph(doc, "Đề tài mang lại 4 đóng góp cốt lõi cho lĩnh vực suy luận phân tán tại biên:")
    add_bullet_item(doc, "Đóng góp 1 (Prefill/Decode- & KV-Cache-Aware Cost Model): Xây dựng mô hình chi phí toán học phân tách rõ ràng hai pha; tích hợp động dung lượng KV-cache theo kiến trúc Grouped-Query Attention (GQA) và thiết lập trần an toàn bộ nhớ $\\alpha = 0.90$ trên từng nút phần cứng.", "1. Cost Model đột phá: ")
    add_bullet_item(doc, "Đóng góp 2 (Contiguous Dynamic Programming Partitioning): Giải quyết triệt để bài toán phân chia lát cắt layer liên tục trên chuỗi nút cố định trong thời gian đa thức $O(K \\cdot N^2)$, loại bỏ hoàn toàn các vòng lặp tự chuyển tiếp (self-transition loopholes).", "2. Thuật toán DP tối ưu: ")
    add_bullet_item(doc, "Đóng góp 3 (Thực nghiệm chuẩn mực với Measured Oracle Regret): Đánh giá định lượng chất lượng phân hoạch qua sai số dự báo (MAPE/RMSE) trên tập dữ liệu cô lập (Held-Out Evaluation) và chứng minh độ hối tiếc thực tế Realized Regret = 0.0 ms so với nghiệm vét cạn tối ưu tuyệt đối.", "3. Chuẩn mực đánh giá cao: ")
    add_bullet_item(doc, "Đóng góp 4 (Nền tảng Universal App trọn vẹn): Hệ thống duy nhất cung cấp trải nghiệm 'Zero-Config' từ khâu phát hiện qua UDP Beacon, cấp phát chứng chỉ mTLS, phân cắt động mô hình đến giao diện Web trực quan 3 Tab.", "4. Sản phẩm hoàn chỉnh: ")

    add_heading_2(doc, "1.5. Các câu hỏi nghiên cứu trọng tâm (Research Questions)")
    add_body_paragraph(doc, "Để dẫn dắt toàn bộ quá trình nghiên cứu và thực nghiệm, đề tài xác lập 8 câu hỏi nghiên cứu cụ thể:")
    add_bullet_item(doc, "RQ1 (Working Set Formulation): Cần mô hình hóa các thành phần bộ nhớ nào để bảo đảm không xảy ra OOM trên cả Windows CUDA và macOS MPS?", "• ")
    add_bullet_item(doc, "RQ2 (Prefill Cost Model): Làm thế nào để dự báo chính xác độ trễ pha Prefill dựa trên công suất tính toán TFLOPs và chi phí truyền thông đa thành phần?", "• ")
    add_bullet_item(doc, "RQ3 (DP Partitioning Optimality): Thuật toán Quy hoạch động có đảm bảo tìm ra lát cắt tối ưu trong không gian tìm kiếm liên tục dưới ràng buộc bộ nhớ cứng hay không?", "• ")
    add_bullet_item(doc, "RQ4 (Binary Tensor Transport): Giao thức nhị phân Custom TCP kết hợp mTLS và CRC32 đạt thông lượng bao nhiêu và có đáp ứng yêu cầu truyền tensor độ chính xác cao hay không?", "• ")
    add_bullet_item(doc, "RQ5 (Auto-Discovery & Telemetry): Cơ chế UDP Beacon có tự động thu thập chính xác cấu hình phần cứng và đo đạc băng thông mạng thực tế mà không cần cấu hình thủ công?", "• ")
    add_bullet_item(doc, "RQ6 (Distributed Break-even): Dưới kích thước mô hình, độ dài prompt và điều kiện mạng nào thì suy luận phân tán mang lại lợi thế vượt trội so với chạy đơn máy?", "• ")
    add_bullet_item(doc, "RQ7 (Cost Model Fidelity): Sai số dự báo của Cost Model trên tập dữ liệu kiểm thử cô lập (Held-Out) đạt mức bao nhiêu phần trăm?", "• ")
    add_bullet_item(doc, "RQ8 (Safety-Ceiling Sensitivity): Ngưỡng an toàn bộ nhớ $\\alpha \\in [0.80, 0.95]$ ảnh hưởng như thế nào đến tính khả thi của phân hoạch và nguy cơ tràn bộ nhớ OOM?", "• ")

    add_heading_2(doc, "1.6. Phạm vi và giới hạn của đề tài")
    add_body_paragraph(doc, "Trong khuôn khổ Đề án Tốt nghiệp với lộ trình 18 tuần (hiện tại ở mốc Giữa kỳ Tuần 8 - 9), phạm vi nghiên cứu được xác định rõ ràng:")
    add_bullet_item(doc, "Môi trường thực thi: Khóa cứng trên nền tảng runtime đồng nhất PyTorch. Trên Windows sử dụng backend PyTorch CUDA kết hợp NVML/WMI; trên macOS sử dụng backend PyTorch MPS (Metal Performance Shaders). Các framework khác như MLX được quy hoạch cho giai đoạn phát triển tương lai.", "• Về Runtime: ")
    add_bullet_item(doc, "Mô hình mục tiêu: Tối ưu hóa pha Prefill đơn truy vấn (single-request TTFT) dưới cơ chế 1 microbatch cho mô hình dạng Transformer Decoder-only (họ Qwen2.5 từ 0.5B đến 7B).", "• Về Mô hình: ")
    add_bullet_item(doc, "Cơ chế song song: Tập trung vào Song song đường ống (Pipeline Parallelism) với các lát cắt layer liên tục (Contiguous Layers). Không triển khai Song song Tensor (Tensor Parallelism) nội bộ qua mạng LAN do giới hạn băng thông mạng dân dụng.", "• Về Song song hóa: ")
    add_bullet_item(doc, "Xử lý lỗi: Áp dụng cơ chế Phân hoạch lại cấp phiên (Session-Boundary Repartitioning / Safe Failure). Khi có nút gặp sự cố, hệ thống hủy phiên hiện tại an toàn, đo lại cụm và tái phân bổ, không áp dụng cơ chế di chuyển trạng thái trực tiếp (live migration) phức tạp.", "• Về Chịu lỗi: ")

    add_heading_2(doc, "1.7. Bố cục của báo cáo tiến độ giữa kỳ")
    add_body_paragraph(doc, "Báo cáo tiến độ giữa kỳ được cấu trúc thành 5 chương chính:")
    add_bullet_item(doc, "Chương 1: Tổng quan và Tính cấp thiết của đề tài.", "• ")
    add_bullet_item(doc, "Chương 2: Cơ sở lý thuyết và Mô hình toán học.", "• ")
    add_bullet_item(doc, "Chương 3: Thiết kế kiến trúc và Các giải pháp kỹ thuật hệ thống.", "• ")
    add_bullet_item(doc, "Chương 4: Kết quả thực hiện đến giữa kỳ (Tuần 8 - 9) (Hiện trạng mã nguồn, kiểm thử tự động, 5 Sub-studies thực nghiệm).", "• ")
    add_bullet_item(doc, "Chương 5: Đánh giá tiến độ và Kế hoạch nghiên cứu nửa sau học kỳ (Tuần 10 - 18).", "• ")
    
    doc.add_page_break()

def build_chapter_2(doc):
    add_heading_1(doc, "CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ MÔ HÌNH TOÁN HỌC")
    
    add_heading_2(doc, "2.1. Kiến trúc Transformer và chu trình suy luận hai pha")
    add_body_paragraph(doc, "Kiến trúc Transformer Decoder-only là nền tảng của hầu hết các mô hình ngôn ngữ lớn hiện đại. Một mô hình gồm một tầng nhúng từ (Token Embedding), một chuỗi gồm $N$ khối biến đổi (Transformer Layers) giống nhau nối tiếp nhau, một tầng chuẩn hóa cuối cùng (RMSNorm) và một tầng chiếu xác suất đầu ra (LM Head). Mỗi khối Transformer Layer lại bao gồm hai thành phần chính:")
    add_bullet_item(doc, "Khối cơ chế chú ý (Grouped-Query Attention - GQA): Tính toán sự tương quan giữa các token trong chuỗi thông qua ma trận truy vấn (Query), chìa khóa (Key) và giá trị (Value).", "1. Attention Block: ")
    add_bullet_item(doc, "Khối mạng truyền thẳng nơ-ron (Feed-Forward Network / SwiGLU): Thực hiện các phép biến đổi phi tuyến tính trên không gian vector ẩn để trích xuất tri thức ngôn ngữ sâu.", "2. FFN Block: ")
    add_body_paragraph(doc, "Quá trình suy luận phân kỳ sâu sắc thành hai pha:")
    add_body_paragraph(doc, "1. Pha Prefill: Toàn bộ chuỗi đầu vào $S_{\\text{prompt}}$ được đưa đồng thời qua mạng. Phép nhân ma trận kích thước lớn được kích hoạt, dẫn đến tính toán Compute-Bound với số lượng phép tính dấu phẩy động đạt xấp xỉ:")
    add_callout_box(doc, "FLOPs_prefill ≈ 2 × N_params × S_prompt + 4 × L × n_heads × d_head × S_prompt^2", "Công thức 2.1: Tổng phép tính FLOPs trong pha Prefill")
    add_body_paragraph(doc, "Thành phần bậc hai $S_{\\text{prompt}}^2$ phản ánh sự gia tăng độ phức tạp tính toán khi độ dài ngữ cảnh mở rộng.")
    add_body_paragraph(doc, "2. Pha Decode: Mô hình sinh từng token mới dựa trên token vừa sinh và toàn bộ lịch sử ngữ cảnh. Để tránh việc phải tính toán lại Key và Value của các token cũ, cơ chế KV-Cache được kích hoạt. Tại mỗi bước sinh, kích thước tensor đầu vào chỉ là 1 token, khiến phép toán trở thành Memory-Bandwidth-Bound.")

    add_heading_2(doc, "2.2. Song song đường ống (Pipeline Parallelism) trên cụm biên dị chủng")
    add_body_paragraph(doc, "Song song đường ống là phương pháp phân chia mô hình gồm $N$ layer thành $K$ phân đoạn liên tiếp. Mỗi thiết bị $k \\in [1..K]$ trong cụm sẽ chịu trách nhiệm quản lý và thực thi một dải layer liên tục $[l_k^{\\text{start}} .. l_k^{\\text{end}}]$, thỏa mãn:")
    add_callout_box(doc, "l_1^start = 0,   l_K^end = N - 1,   và   l_{k+1}^start = l_k^end + 1   (với mọi 1 ≤ k < K)", "Công thức 2.2: Ràng buộc phân cắt layer liên tục (Contiguous Partitioning)")
    add_body_paragraph(doc, "Đặc tính 'liên tục' mang ý nghĩa sống còn trong mạng biên: nó đảm bảo giữa hai thiết bị kế cận chỉ phát sinh đúng một lần truyền tensor kích hoạt trung gian (Activation Tensor) tại ranh giới cắt, triệt tiêu hoàn toàn sự phân mảnh mạng và giảm thiểu tối đa phụ tải truyền thông.")

    add_heading_2(doc, "2.3. Mô hình toán học về Bộ nhớ Working-Set toàn diện")
    add_body_paragraph(doc, "Để loại bỏ hoàn toàn nguy cơ sập OOM trên các thiết bị biên vốn có dung lượng bộ nhớ eo hẹp, đề tài chuẩn hóa mô hình Working-Set $M_{\\text{working}}(k, i, j)$ cho một phân đoạn gồm các layer từ $i$ đến $j$ trên nút $k$ như sau:")
    add_callout_box(doc, "M_working(k, i, j) = M_layers(i, j) + M_fixed(k) + M_KV(i, j, S_max, B) + M_transient + M_transport + M_allocator", "Công thức 2.3: Mô hình Bộ nhớ Working-Set toàn diện")
    add_body_paragraph(doc, "Trong đó, từng thành phần được mô hình hóa toán học chặt chẽ:")
    add_bullet_item(doc, "$M_{\\text{layers}}(i, j) = (j - i + 1) \\times \\text{BytesPerLayer}$: Dung lượng lưu trữ trọng số cố định của các layer được giao.", "• ")
    add_bullet_item(doc, "$M_{\\text{fixed}}(k)$: Dung lượng bộ nhớ cho các thành phần cố định. Nút đầu tiên ($k=1$) phải nạp thêm Embedding Layer; nút cuối cùng ($k=K$) phải nạp thêm Final RMSNorm và LM-Head.", "• ")
    add_bullet_item(doc, "$M_{\\text{KV}}(i, j, S_{\\text{max}}, B)$: Dung lượng bộ đệm KV-Cache cho phân đoạn layer, phụ thuộc vào độ dài ngữ cảnh tối đa $S_{\\text{max}}$ và kích thước batch $B$.", "• ")
    add_bullet_item(doc, "$M_{\\text{transient}}$: Bộ đệm tensor kích hoạt cực đại trong quá trình forward pass của các khối MLP/SwiGLU, xấp xỉ $B \\times S_{\\text{prompt}} \\times d_{\\text{model}} \\times 4 \\times \\text{BytesPerElement}$.", "• ")
    add_bullet_item(doc, "$M_{\\text{transport}}$: Bộ nhớ đệm dành cho việc gửi/nhận và staging tensor qua socket mạng.", "• ")
    add_bullet_item(doc, "$M_{\\text{allocator}}$: Phần bù trừ dành cho chi phí phân mảnh bộ nhớ của PyTorch Caching Allocator (thường dự phòng từ 5% đến 10%).", "• ")

    add_heading_2(doc, "2.4. Mô hình Bộ nhớ KV-Cache nhận thức Grouped-Query Attention")
    add_body_paragraph(doc, "Hầu hết các LLM thế hệ mới (như LLaMA-3, Qwen-2.5) đều áp dụng kiến trúc Grouped-Query Attention (GQA) thay vì Multi-Head Attention (MHA) truyền thống để cắt giảm kích thước KV-cache. Thay vì duy trì số đầu Key/Value bằng với số đầu Query ($n_{\\text{heads}}$), GQA gom nhóm các đầu truy vấn dùng chung $n_{\\text{kv\\_heads}}$ đầu Key/Value (ví dụ Qwen2.5-7B có $n_{\\text{heads}} = 28$ nhưng chỉ có $n_{\\text{kv\\_heads}} = 4$, giảm tới 7 lần dung lượng KV).")
    add_body_paragraph(doc, "Dung lượng KV-Cache cho phân đoạn gồm $L_{\\text{seg}} = (j - i + 1)$ layer được tính chính xác bằng công thức:")
    add_callout_box(doc, "M_KV(i, j, S_max, N_seq) = 2 × L_seg × N_seq × n_kv_heads × d_head × S_max × BytesPerKVElement", "Công thức 2.4: Mô hình KV-Cache nhận thức kiến trúc GQA")
    add_body_paragraph(doc, "Hệ số 2 đầu tiên đại diện cho hai trạng thái Key và Value. Việc tích hợp công thức này giúp Orchest Computing dự báo chính xác dung lượng bộ nhớ tăng thêm trong tương lai khi chuỗi hội thoại kéo dài, ngăn chặn OOM triệt để.")

    add_heading_2(doc, "2.5. Ràng buộc khả thi bộ nhớ và Ngưỡng an toàn alpha")
    add_body_paragraph(doc, "Một phương án phân bổ phân đoạn layer $[i..j]$ cho nút $k$ được coi là khả thi về mặt vật lý nếu và chỉ nếu thỏa mãn bất đẳng thức an toàn:")
    add_callout_box(doc, "M_working(k, i, j) ≤ α · M_usable(k)", "Công thức 2.5: Điều kiện khả thi bộ nhớ với ngưỡng an toàn alpha")
    add_body_paragraph(doc, "Trong đó:")
    add_bullet_item(doc, "$M_{\\text{usable}}(k)$: Dung lượng bộ nhớ thực tế có thể sử dụng được trên nút $k$. Đối với máy Windows CUDA, đây là VRAM trống của GPU rời trích xuất qua NVML. Đối với máy Mac Apple Silicon, đây là lượng Unified Memory khả dụng sau khi trừ đi phần bộ nhớ hệ điều hành chiếm dụng.", "• ")
    add_bullet_item(doc, "$\\alpha$: Ngưỡng an toàn trần (Safety Ceiling Parameter), có giá trị mặc định là $\\alpha = 0.90$ (dành 10% bộ nhớ làm vùng đệm chống sốc). Độ nhạy của tham số $\\alpha$ được khảo sát chi tiết trong phần thực nghiệm.", "• ")

    add_heading_2(doc, "2.6. Mô hình Chi phí trễ thời gian (Latency & Cost Model)")
    add_body_paragraph(doc, "Chi phí thực thi của một giai đoạn (stage) phân đoạn layer $[i+1..j]$ trên nút $k$ trong pha Prefill bao gồm hai thành phần:")
    add_callout_box(doc, "C_pre(k, i+1, j) = T_compute_pre(k, i+1, j, S_prefill, B) + T_comm(k-1, k, ActivationSize(i))", "Công thức 2.6: Chi phí tổng hợp của một phân đoạn trong pha Prefill")
    add_body_paragraph(doc, "1. Thời gian tính toán ($T_{\\text{compute\\_pre}}$): Được xác định dựa trên tổng số phép tính FLOPs của các layer được giao chia cho công suất tính toán hiệu dụng thực tế (Effective TFLOPs) của nút được đo đạc qua Telemetry:")
    add_callout_box(doc, "T_compute_pre = ( ∑_{l=i+1}^j FLOPs_layer(l, S_prefill, B) ) / ( Effective_TFLOPs(k) × 10^12 ) × 1000   (ms)", "Công thức 2.7: Thời gian tính toán ma trận pha Prefill")
    add_body_paragraph(doc, "2. Thời gian truyền thông đa thành phần ($T_{\\text{comm}}$): Đề tài xây dựng mô hình truyền thông 6 thành phần bao quát toàn bộ quá trình vận chuyển qua socket:")
    add_callout_box(doc, "T_comm = T_sync + T_D2H + T_encode + T_network + T_decode + T_H2D", "Công thức 2.8: Mô hình chi phí truyền thông đa thành phần")
    add_body_paragraph(doc, "Trong đó, nếu hai layer liên tiếp cùng nằm trên một nút ($k-1 = k$), thì $T_{\\text{comm}} = 0$. Ngược lại, khi truyền qua mạng LAN:")
    add_callout_box(doc, "T_network ≈ RTT / 2 + ( ActivationBytes × 8 ) / ( Effective_Bandwidth_bps ) × 1000   (ms)", "Công thức 2.9: Thời gian vận chuyển gói tin qua mạng LAN")

    add_heading_2(doc, "2.7. Thuật toán Quy hoạch động Phân hoạch Layer (Contiguous DP Partitioner)")
    add_body_paragraph(doc, "Bài toán phân bổ tối ưu $N$ layer trên $K$ nút phần cứng dị chủng được định thức hóa dưới dạng bài toán Quy hoạch động (Dynamic Programming).")
    add_body_paragraph(doc, "Gọi $DP[k][j]$ là tổng thời gian trễ nhỏ nhất để thực thi $j$ layer đầu tiên ($0..j$) trên tiền tố gồm $k$ nút đầu tiên ($1..k$). Công thức truy hồi được thiết lập như sau:")
    add_callout_box(doc, "DP[k][j] = min_{0 ≤ i < j} { DP[k-1][i] + C_pre(k, i+1, j) }", "Công thức 2.10: Hệ thức truy hồi Quy hoạch động")
    add_body_paragraph(doc, "Điều kiện biên khởi tạo:")
    add_bullet_item(doc, "$DP[0][0] = 0$: Không có layer nào trên 0 nút có chi phí bằng 0.", "• ")
    add_bullet_item(doc, "$DP[0][j] = \\infty$ với mọi $j > 0$: Không thể thực thi layer nào nếu không có nút phần cứng.", "• ")
    add_bullet_item(doc, "Cắt tỉa ràng buộc cứng: Bất kỳ trạng thái nào vi phạm điều kiện an toàn bộ nhớ $M_{\\text{working}}(k, i+1, j) > \\alpha \\cdot M_{\\text{usable}}(k)$ đều được gán $C_{\\text{pre}} = \\infty$.", "• ")
    add_body_paragraph(doc, "Sau khi hoàn thành bảng phương án, nghiệm tối ưu toàn cục được xác định bởi:")
    add_callout_box(doc, "P* = argmin_{1 ≤ κ ≤ K} DP[κ][N]", "Công thức 2.11: Nghiệm tối ưu toàn cục của bài toán phân hoạch")
    add_body_paragraph(doc, "Bằng việc lưu vết mảng cha `parent[k][j]`, thuật toán truy vết ngược (Backtracking) để tái lập chính xác ranh giới cắt cho từng nút. Độ phức tạp tính toán của thuật toán là $O(K \\cdot N^2)$, hoàn toàn vượt trội so với duyệt vét cạn tổ hợp $O(\\binom{N-1}{K-1})$, chỉ mất chưa đầy vài mili-giây để giải bài toán phân bổ 64 layer trên 10 nút.")

    add_heading_2(doc, "2.8. Nghiệm tối ưu đo đạc (Measured Oracle) và Phân tích Độ hối tiếc (Regret Analysis)")
    add_body_paragraph(doc, "Một trong những điểm mới then chốt của đề tài là xây dựng Nghiệm tối ưu đo đạc (Measured Oracle) để làm chuẩn mực đối sánh khách quan tuyệt đối. Trên các cụm thực nghiệm có quy mô vừa phải ($K \\le 3$, $N \\le 36$), nhóm nghiên cứu thực hiện duyệt vét cạn toàn bộ không gian tổ hợp các điểm cắt hợp lệ, đo đạc trực tiếp độ trễ thực tế của từng phương án trên phần cứng để tìm ra phương án tối ưu toàn cục tuyệt đối:")
    add_callout_box(doc, "Oracle* = argmin_{P ∈ FeasiblePartitions} Median_TTFT(P)", "Công thức 2.12: Định nghĩa Nghiệm tối ưu đo đạc (Measured Oracle)")
    add_body_paragraph(doc, "Chất lượng của thuật toán Quy hoạch động được định lượng thông qua Độ hối tiếc thực tế (Realized Regret) và Độ hối tiếc chuẩn hóa (Normalized Relative Regret):")
    add_callout_box(doc, "Realized Regret = TTFT_DP - TTFT_Oracle   (ms)\nNormalized Regret (%) = ( TTFT_DP - TTFT_Oracle ) / TTFT_Oracle × 100%", "Công thức 2.13: Định nghĩa Độ hối tiếc phân hoạch (Partition Regret)")
    add_body_paragraph(doc, "Nếu $\\text{Regret} = 0.0$ ms, thuật toán DP đã tìm ra lời giải hoàn hảo trùng khớp 100% với nghiệm tối ưu tuyệt đối của tự nhiên.")
    
    doc.add_page_break()

print("Chapter 1 and 2 builder loaded.")
