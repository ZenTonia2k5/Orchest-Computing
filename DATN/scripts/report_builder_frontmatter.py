# -*- coding: utf-8 -*-
"""
Script to generate the comprehensive, official BKU Midterm Report (Tuần 8-9)
for Orchest Computing.
Outputs: BAO_CAO_TIEN_DO_GIUA_KY_ORCHEST_COMPUTING.docx
"""
import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

from report_builder_core import (
    setup_page_layout, set_cell_background, set_cell_margins, set_table_borders,
    add_heading_1, add_heading_2, add_heading_3, add_body_paragraph, add_bullet_item,
    add_callout_box, format_styled_table,
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TEXT, COLOR_MUTED,
    HEX_PRIMARY, HEX_SECONDARY, HEX_LIGHT_BG, HEX_BORDER, HEX_WHITE
)

def create_cover_page(doc):
    # Top Institution Headers
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(0)
    p_inst.paragraph_format.space_after = Pt(2)
    r1 = p_inst.add_run("ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH\n")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(13)
    r1.font.bold = True
    r1.font.color.rgb = COLOR_TEXT
    
    r2 = p_inst.add_run("TRƯỜNG ĐẠI HỌC BÁCH KHOA\n")
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(14)
    r2.font.bold = True
    r2.font.color.rgb = COLOR_PRIMARY
    
    r3 = p_inst.add_run("KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH\n")
    r3.font.name = "Times New Roman"
    r3.font.size = Pt(13)
    r3.font.bold = True
    r3.font.color.rgb = COLOR_SECONDARY
    
    # Decorative line
    p_line = doc.add_paragraph()
    p_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_line.paragraph_format.space_after = Pt(45)
    r_l = p_line.add_run("-----------------------***-----------------------")
    r_l.font.name = "Times New Roman"
    r_l.font.size = Pt(11)
    r_l.font.color.rgb = COLOR_MUTED
    
    # Title Section
    p_type = doc.add_paragraph()
    p_type.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_type.paragraph_format.space_after = Pt(12)
    r_type = p_type.add_run("BÁO CÁO TIẾN ĐỘ GIỮA KỲ ĐỒ ÁN TỐT NGHIỆP\n(TUẦN 8 - 9, HỌC KỲ I)")
    r_type.font.name = "Times New Roman"
    r_type.font.size = Pt(15)
    r_type.font.bold = True
    r_type.font.color.rgb = COLOR_PRIMARY
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("ORCHEST COMPUTING:\nHỆ THỐNG ĐIỀU PHỐI SUY LUẬN MÔ HÌNH NGÔN NGỮ LỚN\nPHÂN TÁN TRÊN CỤM THIẾT BỊ BIÊN DỊ CHỦNG\nWINDOWS CUDA – MACOS MPS")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(17)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY
    
    p_subtitle = doc.add_paragraph()
    p_subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_subtitle.paragraph_format.space_after = Pt(65)
    r_sub = p_subtitle.add_run("Mô hình chi phí nhận thức Prefill/Decode, Quản lý bộ nhớ động KV-Cache\nvà Đánh giá độ hối tiếc phân hoạch thực tế (Partition Regret Evaluation)")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    r_sub.font.color.rgb = COLOR_MUTED
    
    # Metadata Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w = [Cm(5.5), Cm(9.5)]
    meta_data = [
        ("CÁN BỘ HƯỚNG DẪN:", "TS. [Tên Cán bộ Hướng dẫn]"),
        ("BỘ MÔN:", "Hệ thống và Mạng Máy tính"),
        ("SINH VIÊN THỰC HIỆN:", "1. Trần Thái Tài (tititai13@gmail.com)\n2. Lưu Hải Đăng (Harry Luu - harryluu158@gmail.com)"),
        ("NGÀNH ĐÀO TẠO:", "Kỹ thuật Máy tính / Khoa học Máy tính")
    ]
    for r_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[r_idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.width, cell_v.width = col_w[0], col_w[1]
        cell_k.text, cell_v.text = k, v
        
        pk = cell_k.paragraphs[0]
        pk.paragraph_format.space_after = Pt(3)
        for run in pk.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = COLOR_TEXT
            
        pv = cell_v.paragraphs[0]
        pv.paragraph_format.space_after = Pt(3)
        for run in pv.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(11.5)
            run.font.bold = (r_idx == 0)
            run.font.color.rgb = COLOR_TEXT
            
    # Location and Date
    p_bot = doc.add_paragraph()
    p_bot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_bot.paragraph_format.space_before = Pt(80)
    p_bot.paragraph_format.space_after = Pt(0)
    r_b = p_bot.add_run("THÀNH PHỐ HỒ CHÍ MINH, THÁNG 10 NĂM 2026")
    r_b.font.name = "Times New Roman"
    r_b.font.size = Pt(12)
    r_b.font.bold = True
    r_b.font.color.rgb = COLOR_TEXT
    
    doc.add_page_break()

def create_front_matter(doc):
    # LỜI CAM ĐOAN
    add_heading_1(doc, "LỜI CAM ĐOAN")
    add_body_paragraph(doc, "Chúng tôi xin cam đoan rằng báo cáo tiến độ giữa kỳ đồ án tốt nghiệp với đề tài \"Orchest Computing: Hệ thống điều phối suy luận mô hình ngôn ngữ lớn phân tán trên cụm thiết bị biên dị chủng Windows CUDA – macOS MPS\" là công trình nghiên cứu nghiêm túc và độc lập của nhóm dưới sự định hướng chuyên môn của Cán bộ hướng dẫn.")
    add_body_paragraph(doc, "Toàn bộ kiến trúc hệ thống, mô hình toán học (Cost Model, Working Set Model, GQA-aware KV-Cache Model), thuật toán quy hoạch động (Dynamic Programming), hệ thống giao vận nhị phân (Binary Tensor Protocol), ứng dụng Universal Dashboard và các kết quả đo đạc thực nghiệm (5 Sub-studies, Held-out evaluation, Measured Oracle Regret) được trình bày trong báo cáo này hoàn toàn xuất phát từ mã nguồn và thực nghiệm thực tế được triển khai trên cụm thiết bị thử nghiệm của nhóm.")
    add_body_paragraph(doc, "Các dữ liệu tham khảo, lý thuyết kế thừa từ các công trình nghiên cứu khoa học liên quan của các tác giả khác đều đã được trích dẫn nguồn rõ ràng và đầy đủ theo chuẩn học thuật IEEE. Chúng tôi xin hoàn toàn chịu trách nhiệm trước Hội đồng Đồ án tốt nghiệp và Nhà trường về tính trung thực và liêm chính học thuật của báo cáo này.")
    
    p_sign = doc.add_paragraph()
    p_sign.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_sign.paragraph_format.space_before = Pt(20)
    r_s = p_sign.add_run("TP. Hồ Chí Minh, ngày 13 tháng 10 năm 2026\nSinh viên thực hiện đề tài\n\n\nTrần Thái Tài — Lưu Hải Đăng")
    r_s.font.name = "Times New Roman"
    r_s.font.size = Pt(12)
    r_s.font.italic = True
    r_s.font.bold = True
    
    # LỜI CẢM ƠN
    add_heading_1(doc, "LỜI CẢM ƠN")
    add_body_paragraph(doc, "Để hoàn thành giai đoạn nghiên cứu và hiện thực giữa kỳ (Tuần 1 đến Tuần 9) của Đồ án tốt nghiệp này, nhóm chúng tôi đã nhận được sự quan tâm, chỉ dẫn tận tình và động viên to lớn từ các Thầy, Cô và gia đình.")
    add_body_paragraph(doc, "Trước hết, nhóm xin gửi lời tri ân sâu sắc nhất đến Cán bộ hướng dẫn, người đã trực tiếp định hướng đề tài, gợi mở các bài toán nghiên cứu có tính mới mẻ và hàm lượng học thuật cao, cũng như luôn kiên nhẫn định hướng tư duy phân tích toán học và phương pháp luận thực nghiệm chuẩn mực cho nhóm trong suốt thời gian qua.")
    add_body_paragraph(doc, "Nhóm cũng xin chân thành cảm ơn các Quý Thầy/Cô thuộc Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách Khoa – ĐHQG-HCM, đặc biệt là các Thầy/Cô Bộ môn Hệ thống và Mạng Máy tính, đã truyền đạt cho chúng tôi nền tảng kiến thức vững chắc về hệ điều hành, hệ phân tán, mạng máy tính và trí tuệ nhân tạo, tạo tiền đề để chúng tôi tự tin giải quyết bài toán phức tạp này.")
    
    # TÓM TẮT ĐỀ TÀI (ABSTRACT)
    add_heading_1(doc, "TÓM TẮT ĐỀ TÀI (ABSTRACT)")
    add_heading_2(doc, "Tóm tắt tiếng Việt")
    add_body_paragraph(doc, "Sự phát triển vượt bậc của các Mô hình Ngôn ngữ Lớn (LLMs) đặt ra thách thức nghiêm trọng về năng lực phần cứng khi triển khai tại biên (Edge Computing). Một thiết bị cá nhân đơn lẻ thường bị giới hạn bởi dung lượng VRAM hoặc RAM khả dụng, không thể nạp và suy luận các mô hình hàng tỷ tham số. Nghiên cứu này thiết kế và hiện thực Orchest Computing — một runtime orchestrator phục vụ suy luận LLMs theo cơ chế song song đường ống (Pipeline Parallelism) trên cụm thiết bị cá nhân dị chủng (Windows NVIDIA CUDA và macOS Apple Silicon MPS) kết nối qua mạng nội bộ (LAN).")
    add_body_paragraph(doc, "Khác với các công trình đi trước chủ yếu áp dụng heuristics tĩnh hoặc không mô hình hóa động tài nguyên, đóng góp cốt lõi của đề tài bao gồm: (1) Xây dựng Cost Model phân tách rạch ròi hai pha suy luận: Prefill (Compute-bound) và Decode (Memory-bandwidth-bound), tích hợp mô hình Working-Set toàn diện có nhận thức KV-cache theo kiến trúc Grouped-Query Attention (GQA) và mức tải đồng thời; (2) Thiết kế thuật toán Quy hoạch động (Contiguous Dynamic Programming) với độ phức tạp đa thức O(K * N^2) nhằm tìm kiếm điểm phân cắt layer liên tục tối ưu độ trễ Time-to-First-Token (TTFT) dưới trần an toàn bộ nhớ cứng; (3) Thiết kế giao thức vận chuyển Tensor nhị phân Zero-overhead với checksum CRC32 và mã hóa mTLS/CurveZMQ; (4) Xây dựng ứng dụng Universal Dashboard hoàn chỉnh cho phép tự động khám phá cụm qua UDP Beacon, quản trị fleet và phân bổ trực quan.")
    add_body_paragraph(doc, "Đến mốc giữa kỳ (Tuần 8 - 9), nhóm đã hoàn thiện 100% Core Engine, vượt qua 45/46 ca kiểm thử tự động, và triển khai thực nghiệm toàn diện trên cụm thực tế gồm 3 máy (Windows RTX 4070 Laptop, MacBook Pro M2 Pro, Windows RTX 3050). Kết quả đo đạc trên 4 mô hình thuộc họ Qwen2.5 chứng minh thuật toán DP đạt độ hối tiếc thực tế (Realized Regret) bằng 0.0 ms so với nghiệm vét cạn tối ưu tuyệt đối (Measured Oracle), chứng minh tính khả thi và độ chính xác vượt trội của mô hình.")
    
    add_heading_2(doc, "Abstract (English)")
    add_body_paragraph(doc, "The rapid advancement of Large Language Models (LLMs) poses significant hardware deployment challenges on consumer edge devices due to the fundamental 'Memory Wall'. Individual edge devices lack sufficient VRAM or unified memory to host multi-billion parameter models locally. This graduation thesis presents Orchest Computing, a distributed runtime orchestrator designed for collaborative LLM pipeline-parallel inference across heterogeneous edge clusters comprising Windows NVIDIA CUDA and macOS Apple Silicon MPS machines over standard Local Area Networks (LAN).", italic=True)
    add_body_paragraph(doc, "Distinct from existing distributed frameworks that rely on static heuristics or ignore dynamic runtime state, our novel contributions include: (1) A comprehensive Prefill-oriented and Decode-aware analytical Cost Model tightly coupled with a GQA-aware dynamic KV-cache footprint and strict memory safety ceilings; (2) An optimal contiguous layer Dynamic Programming (DP) partitioner operating in O(K * N^2) polynomial time to minimize Time-to-First-Token (TTFT) without memory violation; (3) A zero-overhead binary tensor transport protocol featuring CRC32 packet integrity and mTLS/CurveZMQ encryption; and (4) A production-grade Universal Dashboard Web Application supporting UDP broadcast discovery, fleet management, and interactive pipeline monitoring.", italic=True)
    add_body_paragraph(doc, "By the midterm milestone (Weeks 8–9), the core engine and its automated test suite (45/46 tests passing) have been fully implemented. Extensive physical cluster benchmarks across heterogeneous nodes (RTX 4070 Laptop, Apple M2 Pro, RTX 3050) across four Qwen2.5 model architectures reveal a Realized Regret of exactly 0.0 ms relative to the exhaustive Measured Oracle, demonstrating the theoretical fidelity and practical efficiency of the proposed orchestration framework.", italic=True)
    
    # DANH MỤC TỪ VIẾT TẮT
    add_heading_1(doc, "DANH MỤC KÝ HIỆU VÀ TỪ VIẾT TẮT")
    abbr_headers = ["Từ viết tắt", "Tên tiếng Anh đầy đủ", "Ý nghĩa / Diễn giải tiếng Việt"]
    abbr_data = [
        ("LLM", "Large Language Model", "Mô hình ngôn ngữ lớn (các mô hình AI tạo sinh nhiều tỷ tham số)"),
        ("TTFT", "Time to First Token", "Thời gian trễ để sinh ra token đầu tiên (đặc trưng cho pha Prefill)"),
        ("ITL", "Inter-Token Latency", "Thời gian trễ trung bình giữa các token liên tiếp (pha Decode)"),
        ("KV-Cache", "Key-Value Cache", "Bộ nhớ đệm lưu trữ các vector Key và Value của các token đã qua xử lý"),
        ("GQA", "Grouped-Query Attention", "Kiến trúc cơ chế chú ý gom nhóm các đầu Key/Value để tiết kiệm bộ nhớ"),
        ("MHA", "Multi-Head Attention", "Cơ chế chú ý đa đầu truyền thống trong kiến trúc Transformer"),
        ("DP", "Dynamic Programming", "Quy hoạch động (thuật toán tối ưu hóa phân hoạch phân rã mô hình)"),
        ("CUDA", "Compute Unified Device Architecture", "Nền tảng tính toán song song trên card đồ họa của NVIDIA"),
        ("MPS", "Metal Performance Shaders", "Framework tăng tốc đồ họa và tính toán tensor trên macOS Apple Silicon"),
        ("NVML", "NVIDIA Management Library", "Thư viện C giám sát trạng thái GPU phần cứng NVIDIA (nhiệt độ, VRAM)"),
        ("WMI", "Windows Management Instrumentation", "Giao diện quản lý và trích xuất thông tin phần cứng của hệ điều hành Windows"),
        ("mTLS", "Mutual Transport Layer Security", "Xác thực bảo mật hai chiều qua chứng chỉ số X.509 giữa Host và Worker"),
        ("RTT", "Round Trip Time", "Thời gian khứ hồi của gói tin truyền qua mạng nội bộ"),
        ("OOM", "Out Of Memory", "Lỗi tràn bộ nhớ phần cứng (xảy ra khi bộ nhớ yêu cầu vượt quá dung lượng)"),
        ("WS", "Working Set", "Tổng tập hợp bộ nhớ cần thiết trong quá trình vận hành suy luận tại một nút"),
        ("GEMM", "General Matrix Multiply", "Phép nhân ma trận tổng quát (phép toán nền tảng của mạng nơ-ron)")
    ]
    abbr_table = doc.add_table(rows=len(abbr_data) + 1, cols=3)
    format_styled_table(abbr_table, [Cm(2.5), Cm(5.5), Cm(7.5)], abbr_headers, abbr_data)
    
    # MỤC LỤC TỔNG QUAN
    add_heading_1(doc, "MỤC LỤC TỔNG QUAN")
    toc_items = [
        ("CHƯƠNG 1: TỔNG QUAN VÀ TÍNH CẤP THIẾT CỦA ĐỀ TÀI", "1"),
        ("  1.1. Bối cảnh nghiên cứu và nhu cầu suy luận LLM tại biên", "1"),
        ("  1.2. Thách thức cốt lõi trong suy luận phân tán tại biên", "2"),
        ("  1.3. Khảo sát công trình liên quan và khoảng trống nghiên cứu", "4"),
        ("  1.4. Mục tiêu nghiên cứu và định vị đóng góp mới", "6"),
        ("  1.5. Các câu hỏi nghiên cứu trọng tâm (Research Questions RQ1 - RQ8)", "8"),
        ("  1.6. Phạm vi và giới hạn của đề tài", "9"),
        ("  1.7. Bố cục của báo cáo tiến độ giữa kỳ", "10"),
        ("CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ MÔ HÌNH TOÁN HỌC", "11"),
        ("  2.1. Kiến trúc Transformer và chu trình suy luận LLM hai pha", "11"),
        ("  2.2. Song song đường ống (Pipeline Parallelism) trên cụm biên dị chủng", "13"),
        ("  2.3. Mô hình Bộ nhớ Working-Set toàn diện (Working Set Memory Model)", "14"),
        ("  2.4. Mô hình Bộ nhớ KV-Cache nhận thức Grouped-Query Attention (GQA)", "16"),
        ("  2.5. Ràng buộc khả thi bộ nhớ và Ngưỡng an toàn alpha", "17"),
        ("  2.6. Mô hình Chi phí trễ thời gian (Latency & Cost Model)", "18"),
        ("  2.7. Thuật toán Quy hoạch động Phân hoạch Layer (Contiguous DP Partitioner)", "20"),
        ("  2.8. Nghiệm tối ưu đo đạc (Measured Oracle) và Phân tích Độ hối tiếc", "22"),
        ("CHƯƠNG 3: THIẾT KẾ KIẾN TRÚC VÀ CÁC GIẢI PHÁP KỸ THUẬT HỆ THỐNG", "24"),
        ("  3.1. Kiến trúc tổng thể Orchest Computing (Mô hình Orchestrator - Worker)", "24"),
        ("  3.2. Phân hệ Tự động Phát hiện và Quản trị Cụm (UDP Beacon & Fleet)", "26"),
        ("  3.3. Phân hệ Đo đạc Động và Đánh giá Phần cứng (Dynamic Profiling)", "28"),
        ("  3.4. Phân hệ Giao thức Mạng và Vận chuyển Tensor Nhị phân Zero-Overhead", "30"),
        ("  3.5. Module Tăng tốc Quy hoạch Động bằng C++ (dp_solver.cpp)", "32"),
        ("  3.6. Module Trích xuất và Phân cắt Lớp Mô hình Linh hoạt (AutoModelSlicer)", "33"),
        ("  3.7. Kiến trúc An toàn và Chịu lỗi Cấp Phiên (Safe Failure)", "34"),
        ("  3.8. Giao diện Người dùng Hợp nhất (Universal Dashboard Web Application)", "35"),
        ("CHƯƠNG 4: KẾT QUẢ THỰC HIỆN ĐẾN GIỮA KỲ (TUẦN 8 - 9)", "37"),
        ("  4.1. Hiện trạng triển khai mã nguồn và kết quả kiểm thử tự động", "37"),
        ("  4.2. Cấu hình cụm thiết bị thử nghiệm vật lý", "38"),
        ("  4.3. Thực nghiệm 1: Chất lượng Phân hoạch DP và Phân tích Độ hối tiếc (Sub-study 1)", "39"),
        ("  4.4. Thực nghiệm 2: Tác động của Băng thông và Độ trễ Mạng (Sub-study 2)", "42"),
        ("  4.5. Thực nghiệm 3: Biến thiên KV-Cache theo Mức độ Đồng thời (Sub-study 3)", "43"),
        ("  4.6. Thực nghiệm 4: Phụ tải Giao vận và An ninh (Sub-study 4)", "44"),
        ("  4.7. Thực nghiệm 5: Tải nhiệt và Ổn định Tài nguyên (Sub-study 5)", "45"),
        ("  4.8. Đánh giá Độ trung thực của Cost Model trên Tập kiểm thử Cô lập (Held-out)", "46"),
        ("  4.9. Khảo sát Độ nhạy của Ngưỡng an toàn Bộ nhớ alpha (RQ8)", "47"),
        ("CHƯƠNG 5: ĐÁNH GIÁ TIẾN ĐỘ VÀ KẾ HOẠCH NỬA SAU HỌC KỲ", "49"),
        ("  5.1. Tổng kết đối chiếu tiến độ hoàn thành so với đề cương đề xuất", "49"),
        ("  5.2. Khó khăn kỹ thuật đã giải quyết và bài học kinh nghiệm", "50"),
        ("  5.3. Kế hoạch nghiên cứu và triển khai chi tiết nửa sau học kỳ (Tuần 10 - 18)", "51"),
        ("TÀI LIỆU THAM KHẢO", "54"),
        ("PHỤ LỤC", "56")
    ]
    p_t = doc.add_paragraph()
    for item, page in toc_items:
        p_row = doc.add_paragraph()
        p_row.paragraph_format.space_after = Pt(2)
        p_row.paragraph_format.line_spacing = 1.15
        is_bold = not item.startswith("  ")
        r_item = p_row.add_run(item)
        r_item.font.name = "Times New Roman"
        r_item.font.size = Pt(11)
        r_item.font.bold = is_bold
        r_item.font.color.rgb = COLOR_PRIMARY if is_bold else COLOR_TEXT
        
        # Dots leader
        dots_count = max(5, 75 - len(item))
        r_dots = p_row.add_run(" " + "." * dots_count + " ")
        r_dots.font.name = "Times New Roman"
        r_dots.font.size = Pt(10)
        r_dots.font.color.rgb = COLOR_MUTED
        
        r_pg = p_row.add_run(page)
        r_pg.font.name = "Times New Roman"
        r_pg.font.size = Pt(11)
        r_pg.font.bold = is_bold
        
    doc.add_page_break()

print("Front matter module loaded.")
