# -*- coding: utf-8 -*-
"""
Full Document Generator:
Converts MIDTERM_PROGRESS_REPORT_WEEKLY_EN.md into official academic Word (.docx) documents.
Generates:
1. d:\\BKU\\Orchest-Computing\\MIDTERM_PROGRESS_REPORT_WEEKLY_EN.docx
2. d:\\BKU\\Orchest-Computing\\MIDTERM_PROGRESS_REPORT_WEEKLY_VN.docx
"""

import os
import sys
import io
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

COLOR_PRIMARY = RGBColor(0x1B, 0x36, 0x5D)     # Deep Navy Blue
COLOR_SECONDARY = RGBColor(0x2B, 0x54, 0x7E)   # Medium Navy
COLOR_TEXT = RGBColor(0x22, 0x22, 0x22)        # Charcoal Text
COLOR_MUTED = RGBColor(0x55, 0x55, 0x55)       # Muted Gray
COLOR_SUCCESS = RGBColor(0x1B, 0x7E, 0x36)     # Deep Green

HEX_PRIMARY = "1B365D"
HEX_SECONDARY = "2B547E"
HEX_LIGHT_BG = "F4F6F9"
HEX_BORDER = "D0D7DE"
HEX_WHITE = "FFFFFF"

def setup_page_layout(doc, header_text, page_prefix="Page "):
    for sec in doc.sections:
        sec.top_margin = Cm(2.0)
        sec.bottom_margin = Cm(2.0)
        sec.left_margin = Cm(3.0)      # Standard 3.0cm binding margin
        sec.right_margin = Cm(2.0)
        
        # Header setup
        header = sec.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run(header_text)
        hrun.font.name = "Times New Roman"
        hrun.font.size = Pt(8.5)
        hrun.font.italic = True
        hrun.font.color.rgb = COLOR_MUTED
        
        # Footer setup
        footer = sec.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        frun_inst = fp.add_run("HCMUT – VNU-HCM  |  ")
        frun_inst.font.name = "Times New Roman"
        frun_inst.font.size = Pt(9.0)
        frun_inst.font.color.rgb = COLOR_MUTED
        
        frun_p = fp.add_run(page_prefix)
        frun_p.font.name = "Times New Roman"
        frun_p.font.size = Pt(9.0)
        frun_p.font.color.rgb = COLOR_MUTED
        
        fldSimple = OxmlElement('w:fldSimple')
        fldSimple.set(qn('w:instr'), 'PAGE')
        fp._p.append(fldSimple)

def set_cell_background(cell, hex_color):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D0D7DE", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))

def add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(14.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return p

def add_h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY
    return p

def add_h3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = COLOR_TEXT
    return p

def add_p(doc, text, bold_prefix="", italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4.5)
    p.paragraph_format.line_spacing = 1.25
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(12)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_TEXT
        
    r_body = p.add_run(text)
    r_body.font.name = "Times New Roman"
    r_body.font.size = Pt(12)
    r_body.font.italic = italic
    r_body.font.color.rgb = COLOR_TEXT
    return p

def add_bullet(doc, text, bold_prefix=""):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.2
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(12)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_TEXT
        
    r_body = p.add_run(text)
    r_body.font.name = "Times New Roman"
    r_body.font.size = Pt(12)
    r_body.font.color.rgb = COLOR_TEXT
    return p

def add_callout(doc, text_or_formula, title="", is_code=False):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, HEX_LIGHT_BG)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=140)
    
    tblPr = tbl._tbl.tblPr
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>
        <w:left w:val="single" w:sz="24" w:space="0" w:color="{HEX_PRIMARY}"/>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))
    
    cp = cell.paragraphs[0]
    cp.paragraph_format.space_before = Pt(2)
    cp.paragraph_format.space_after = Pt(2)
    cp.paragraph_format.line_spacing = 1.2
    
    if title:
        rt = cp.add_run(title + "\n")
        rt.font.name = "Times New Roman"
        rt.font.size = Pt(11)
        rt.font.bold = True
        rt.font.color.rgb = COLOR_PRIMARY
        
    rc = cp.add_run(text_or_formula)
    rc.font.name = "Consolas" if is_code else "Times New Roman"
    rc.font.size = Pt(10.5 if is_code else 11)
    rc.font.italic = not is_code
    rc.font.color.rgb = COLOR_TEXT
    
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)

def format_table(table, col_widths, headers, rows_data):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color=HEX_BORDER, sz="4")
    
    while len(table.rows) < len(rows_data) + 1:
        table.add_row()
        
    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(10.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            
    # Data Rows
    for row_idx, data in enumerate(rows_data):
        row_cells = table.rows[row_idx + 1].cells
        bg_color = HEX_LIGHT_BG if (row_idx % 2 == 1) else HEX_WHITE
        for col_idx, text in enumerate(data):
            row_cells[col_idx].text = str(text)
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=90, bottom=90, left=100, right=100)
            p = row_cells[col_idx].paragraphs[0]
            txt_str = str(text).strip()
            
            if col_idx == 0 or len(txt_str) > 30:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(10)
                if "100%" in txt_str or "COMPLETED" in txt_str or "HOÀN THÀNH" in txt_str:
                    r.font.bold = True
                    r.font.color.rgb = COLOR_SUCCESS
                elif "UPCOMING" in txt_str or "KẾ HOẠCH" in txt_str:
                    r.font.bold = True
                    r.font.color.rgb = COLOR_SECONDARY
                else:
                    r.font.color.rgb = COLOR_TEXT
                    
    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width

def build_academic_header(doc, is_vietnamese=False):
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(0)
    p_inst.paragraph_format.space_after = Pt(2)
    
    if is_vietnamese:
        r1 = p_inst.add_run("ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH\n")
        r2 = p_inst.add_run("TRƯỜNG ĐẠI HỌC BÁCH KHOA\n")
        r3 = p_inst.add_run("KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH — BỘ MÔN HỆ THỐNG VÀ MẠNG MÁY TÍNH\n")
    else:
        r1 = p_inst.add_run("VIETNAM NATIONAL UNIVERSITY HO CHI MINH CITY\n")
        r2 = p_inst.add_run("HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY (HCMUT)\n")
        r3 = p_inst.add_run("FACULTY OF COMPUTER SCIENCE AND ENGINEERING — DEPT. OF COMPUTER SYSTEMS & NETWORKS\n")
        
    r1.font.name = "Times New Roman"; r1.font.size = Pt(11); r1.font.bold = True; r1.font.color.rgb = COLOR_TEXT
    r2.font.name = "Times New Roman"; r2.font.size = Pt(12.5); r2.font.bold = True; r2.font.color.rgb = COLOR_PRIMARY
    r3.font.name = "Times New Roman"; r3.font.size = Pt(10.5); r3.font.bold = True; r3.font.color.rgb = COLOR_SECONDARY
    
    p_line = doc.add_paragraph()
    p_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_line.paragraph_format.space_after = Pt(16)
    r_l = p_line.add_run("-----------------------***-----------------------")
    r_l.font.name = "Times New Roman"; r_l.font.size = Pt(10); r_l.font.color.rgb = COLOR_MUTED
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(4)
    if is_vietnamese:
        r_t = p_title.add_run("BÁO CÁO TIẾN ĐỘ GIỮA KỲ: ORCHEST COMPUTING\n")
        r_sub = p_title.add_run("Hệ thống Điều phối Suy luận Mô hình Ngôn ngữ Lớn Phân tán\ntrên Cụm Thiết bị Biên Dị chủng (Windows CUDA & macOS MPS)")
    else:
        r_t = p_title.add_run("MIDTERM PROGRESS REPORT: ORCHEST COMPUTING\n")
        r_sub = p_title.add_run("Decentralized Large Language Model Inference Orchestration on Heterogeneous Consumer-Grade Edge Clusters (Windows CUDA & macOS MPS)")
        
    r_t.font.name = "Times New Roman"; r_t.font.size = Pt(16); r_t.font.bold = True; r_t.font.color.rgb = COLOR_PRIMARY
    r_sub.font.name = "Times New Roman"; r_sub.font.size = Pt(12); r_sub.font.italic = True; r_sub.font.color.rgb = COLOR_SECONDARY
    
    # Metadata Table
    tbl = doc.add_table(rows=6, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="D0D7DE", sz="4")
    
    if is_vietnamese:
        meta_data = [
            ("Chương trình đào tạo", "Đồ án Tốt nghiệp Kỹ sư / Khóa luận Tốt nghiệp Đại học"),
            ("Cơ quan chủ quản", "Trường Đại học Bách Khoa – ĐHQG-HCM (Khoa KH&KT Máy tính)"),
            ("Cán bộ hướng dẫn", "TS. Quản Thành Thơ (Cán bộ hướng dẫn chính)"),
            ("Sinh viên thực hiện", "1. Trần Thái Tài (tititai13@gmail.com)\n2. Lưu Hải Đăng (Harry Luu) (harryluu158@gmail.com)"),
            ("Mã nguồn dự án", "d:\\BKU\\Orchest-Computing\\DATN (Local Git Repository)"),
            ("Kỳ báo cáo", "Mốc Giữa kỳ (Hoàn thành Tuần 1–9; Đề cương Kế hoạch Tuần 10–18)")
        ]
    else:
        meta_data = [
            ("Academic Program", "Graduation Capstone Project / B.Eng. Thesis"),
            ("Institution", "Faculty of Computer Science and Engineering, HCMUT – VNU-HCM"),
            ("Advisor", "Dr. [Advisor Name]"),
            ("Students", "1. Tran Thai Tai (tititai13@gmail.com)\n2. Luu Hai Dang (Harry Luu) (harryluu158@gmail.com)"),
            ("Project Repository", "d:\\BKU\\Orchest-Computing\\DATN (Local Git Repository)"),
            ("Reporting Period", "Midterm Milestone (Weeks 1–9 Completed; Weeks 10–18 Roadmap)")
        ]
        
    for idx, (label, val) in enumerate(meta_data):
        row = tbl.rows[idx]
        set_cell_background(row.cells[0], HEX_LIGHT_BG)
        set_cell_background(row.cells[1], HEX_WHITE)
        set_cell_margins(row.cells[0], top=80, bottom=80, left=100, right=100)
        set_cell_margins(row.cells[1], top=80, bottom=80, left=100, right=100)
        
        row.cells[0].width = Cm(4.2)
        row.cells[1].width = Cm(11.8)
        
        p0 = row.cells[0].paragraphs[0]
        r0 = p0.add_run(label)
        r0.font.name = "Times New Roman"; r0.font.size = Pt(10.5); r0.font.bold = True; r0.font.color.rgb = COLOR_TEXT
        
        p1 = row.cells[1].paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.name = "Times New Roman"; r1.font.size = Pt(10.5); r1.font.color.rgb = COLOR_TEXT
        
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(4)
    sp.paragraph_format.space_after = Pt(8)

def generate_english_document(output_path):
    doc = docx.Document()
    setup_page_layout(doc, "MIDTERM PROGRESS REPORT: ORCHEST COMPUTING", page_prefix="Page ")
    build_academic_header(doc, is_vietnamese=False)
    
    # EXECUTIVE SUMMARY
    add_h1(doc, "EXECUTIVE SUMMARY")
    add_callout(
        doc,
        "By the Midterm Milestone (end of Week 9), the Orchest Computing research project has successfully completed 100% of the targeted scope planned for Phase P1 (Survey, Design & Modeling) and Phase P2 (Core Engine Implementation & Physical Cluster Empirical Benchmarking).\n\n"
        "The core software modules comprise ~8,400 lines of modular code, accompanied by a comprehensive automated test suite (45/46 test cases passing, 97.8% pass rate). Empirical experiments conducted directly on a physical heterogeneous edge testbed (NVIDIA RTX 4070 Laptop, Apple M2 Pro, and NVIDIA RTX 3050 Laptop) across four Qwen2.5 architectures (0.5B, 1.5B, 3B, and 7B) demonstrated a Realized Regret of exactly 0.0 ms (0.0%) compared to an exhaustive Measured Oracle, validating the mathematical optimality of our Dynamic Programming (DP) partitioner.",
        title="Key Midterm Milestone Highlights"
    )
    add_p(doc, "Below is the exhaustive, week-by-week account of all engineering and theoretical milestones achieved, followed by the concrete objectives and deliverables scheduled for the remaining half of the semester.")
    
    # PART I
    add_h1(doc, "PART I: WEEK-BY-WEEK ACCOMPLISHMENTS (WEEKS 1 – 9)")
    
    add_h2(doc, "Phase P1: Survey, Design, and Theoretical Formulation (Weeks 1 – 4)")
    
    add_h3(doc, "Week 1: Problem Formulation, Systematic Literature Review & Comparative Analysis")
    add_bullet(doc, "Formalized the core challenges of running Large Language Models (LLMs) on consumer-grade LAN edge devices: the physical 'Memory Wall' (insufficient VRAM on edge GPUs), severe hardware heterogeneity (disparate accelerators, operating systems, and memory hierarchies), and dynamic network volatility.", "• Problem Definition: ")
    add_bullet(doc, "Conducted an exhaustive comparison across leading edge/distributed inference paradigms:", "• Related-Work Systematic Survey: ")
    add_bullet(doc, "PipeEdge: Static pipeline partitioning lacking dynamic KV-cache modeling.", "    - ")
    add_bullet(doc, "exo: P2P topology discovery without mathematical execution-regime separation.", "    - ")
    add_bullet(doc, "Petals: Distributed swarm inference oriented toward internet-scale crowd-computing rather than low-latency deterministic local networks.", "    - ")
    add_bullet(doc, "PiPPy: Server-grade pipeline engine unsuited for heterogeneous consumer memory budgets.", "    - ")
    add_bullet(doc, "Constructed the systematic Related-Work Comparison Matrix identifying key research gaps, formalizing the scope boundaries of the Orchest Computing MVP.", "• Deliverable: ")
    
    add_h3(doc, "Week 2: Mathematical Formulation of Working-Set Memory & Dynamic KV-Cache")
    add_bullet(doc, "Formulated the per-node runtime memory requirement equation:", "• Working-Set Decomposition: ")
    add_callout(doc, "M_k^ws = M_k^weight + M_k^kv(S, B) + M_k^act(S, B) + M_k^overhead", title="Equation (2.1): Per-Node Working-Set Memory")
    add_bullet(doc, "Derived an exact memory calculation model dynamically parameterized by model architecture configurations:", "• Grouped-Query Attention (GQA) KV-Cache Formulation: ")
    add_callout(doc, "Mem_KV_per_layer = 2 * n_kv_heads * d_head * S_max * B * bytes_per_elem", title="Equation (2.2): Dynamic KV-Cache Memory Scaling")
    add_bullet(doc, "Established the admissible memory boundary M_k^ws <= alpha * M_k,usable, introducing the safety parameter alpha in [0.80, 0.95] to prevent Out-Of-Memory (OOM) crashes under dynamic batch and sequence expansions.", "• Safety Threshold Formulation: ")
    add_bullet(doc, "Mathematical specification document and Python prototype validating KV-cache scaling against actual Hugging Face model configurations (Qwen/Qwen2.5).", "• Deliverable: ")
    
    add_h3(doc, "Week 3: Cost Model Formulation & Contiguous Dynamic Programming (DP) Solver")
    add_bullet(doc, "Formally separated the inference pipeline into two distinct execution regimes:", "• Execution Regime Separation: ")
    add_bullet(doc, "Prefill Phase: Compute-bound, processing prompt tokens in parallel, optimized for Time-To-First-Token (TTFT).", "    1. ")
    add_bullet(doc, "Decode Phase: Memory-bandwidth/network-bound, generating tokens autoregressively, measured for Inter-Token Latency (ITL).", "    2. ")
    add_bullet(doc, "Modeled computation time T_comp(i, j, k) via measured effective TFLOPs and activation transmission overhead T_comm(k, k+1) via network RTT and bandwidth.", "• Cost Matrix Derivation: ")
    add_bullet(doc, "Designed the optimal layer-partitioning algorithm running in O(K * N^2) time complexity (N layers across K ordered devices) strictly enforced under hard memory feasibility constraints:", "• Contiguous DP Formulation: ")
    add_callout(doc, "DP[k][j] = min_{0 <= i <= j} { DP[k-1][i] + T_comp(i+1, j, k) + T_comm(k-1, k) }", title="Equation (2.3): Dynamic Programming Recurrence Relation")
    add_bullet(doc, "Algorithmic proof of contiguous layer assignment, elimination of self-transition loopholes, and baseline DP implementation.", "• Deliverable: ")
    
    add_h3(doc, "Week 4: Network Protocol Specification, Security Architecture & Empirical Protocol")
    add_bullet(doc, "Specified a custom 18-byte zero-overhead binary framing protocol (!4sHIII header: Magic Bytes 'ORCH', Packet Type, Sequence ID, Payload Length, and CRC32 Checksum) enabling direct PyTorch tensor buffer streaming without serialization bloat.", "• TensorPacket Binary Framing: ")
    add_bullet(doc, "Specified Mutual TLS (mTLS) with per-node X.509 certificate verification (ca.crt, host.crt, worker.crt) and optional AES-GCM 128-bit payload encryption.", "• Edge Security Model: ")
    add_bullet(doc, "Formalized the evaluation methodology defining five core empirical sub-studies, including the protocol for comparing DP solutions against an independently evaluated exhaustive Measured Oracle to determine Realized Regret.", "• Empirical Validation Protocol: ")
    add_bullet(doc, "Comprehensive Network Protocol & Benchmark Protocol Specifications (Chapters 2 and 3 of the midterm documentation).", "• Deliverable: ")
    
    add_h2(doc, "Phase P2: Core Engine Implementation & Physical Cluster Benchmarks (Weeks 5 – 9)")
    
    add_h3(doc, "Week 5: Networking & Zero-Configuration Swarm Discovery Subsystem")
    add_bullet(doc, "Implemented orchest.network.discovery, enabling Worker edge nodes to listen on 0.0.0.0:50005 and auto-register with the Host Master via UDP broadcast queries (255.255.255.255), extracting real LAN IPs directly from socket headers.", "• UDP Beacon Auto-Discovery: ")
    add_bullet(doc, "Developed orchest.network.transport with custom TCP socket handling, achieving peak raw socket throughput exceeding 2,400 MB/s on local links.", "• High-Speed Binary Transport: ")
    add_bullet(doc, "Added abstraction adapters supporting Raw TCP, gRPC (tensor_service.proto), and ZeroMQ (zmq_transport.py).", "• Multi-Transport Layer: ")
    add_bullet(doc, "Built background heartbeat loops (2-second interval, 5-second failure threshold) for proactive node health monitoring.", "• Keep-Alive & Session State: ")
    add_bullet(doc, "Fully functioning network engine (discovery.py, transport.py, protocol.py) with zero-config clustering.", "• Deliverable: ")
    
    add_h3(doc, "Week 6: Cross-Platform Hardware Profiler & Telemetry Engine")
    add_bullet(doc, "Windows Platform: Integrated pynvml to query NVIDIA GPU VRAM, compute capability, temperature, and power status; integrated Windows WMI and psutil for host CPU/RAM metrics.", "• Heterogeneous Profiler (Windows): ")
    add_bullet(doc, "macOS Platform: Developed native query logic for Apple Silicon Unified Memory and Metal Performance Shaders (MPS), accurately computing usable memory without discrete VRAM misconceptions.", "• Heterogeneous Profiler (macOS): ")
    add_bullet(doc, "Built an automated GEMM (General Matrix Multiply) benchmarking harness executing FP16 matrix operations on target accelerators (CUDA / MPS) to measure true effective TFLOPs rather than synthetic manufacturer claims.", "• Effective TFLOPs Benchmark: ")
    add_bullet(doc, "Automated round-trip latency (RTT) socket pinging and link throughput benchmarking between cluster nodes.", "• LAN Network Profiler: ")
    add_bullet(doc, "Profiling and telemetry subsystem outputting structured hardware_profile.json and calibration_profile.json.", "• Deliverable: ")
    
    add_h3(doc, "Week 7: Dynamic Programming Solvers (Python & C++ Engine) & AutoModelSlicer")
    add_bullet(doc, "Integrated the Cost Model calibrator, incorporating real hardware profiles into the cost matrix to generate optimal layer assignment plans.", "• Calibrated DP Solver: ")
    add_bullet(doc, "Reimplemented the DP algorithm in native C++ using contiguous flat 1D memory buffers and OpenMP parallelism. Compiled into dynamic libraries (.dll on Windows, .dylib on macOS), invoked via Python ctypes, slashing solver execution time from 15 ms to under 0.3 ms.", "• C++ Core Engine Acceleration: ")
    add_bullet(doc, "Universal AutoModelSlicer: Leveraged model reflection to extract Transformer layer structures across various model families (.model.layers for Qwen/LLaMA); built ModelSliceModule encapsulating assigned layers along with Embeddings or RMSNorm + LM-Head; implemented Selective Weight Sharded Loading to prevent memory waste.", "• Universal AutoModelSlicer: ")
    add_bullet(doc, "High-performance DP solver and modular model slicer passing standalone unit tests.", "• Deliverable: ")
    
    add_h3(doc, "Week 8: Universal Web Application & Glassmorphism Dashboard")
    add_bullet(doc, "Packaged the entire platform into a single unified application where any machine can toggle between Host Master and Worker Edge modes.", "• Unified Universal Launcher: ")
    add_bullet(doc, "Developed a modern Glassmorphism web interface featuring three distinct tabs:", "• FastAPI Backend & Interactive UI: ")
    add_bullet(doc, "Host Master Tab: Cluster scanning, node telemetry display, LLM model selection, Auto DP Solver trigger, and interactive inference Playground.", "    1. ")
    add_bullet(doc, "Worker Edge Tab: Real-time resource monitors, allocated slice inspection, Accept/Reject authorization banners, and connection history logs.", "    2. ")
    add_bullet(doc, "Manage Fleet Tab: Friendly alias assignment, node-specific latency ping diagnostics, Swarm participation toggle, and inactive node pruning.", "    3. ")
    add_bullet(doc, "Implemented dynamic frontend slider rebalancing, ensuring the sum of layers assigned across all nodes remains strictly 100% when manually adjusted.", "• Auto-Balancing Slicing Bar (Sum = 100%): ")
    add_bullet(doc, "Fully responsive, operational Web UI (server.py, app.js, index.html, styles.css).", "• Deliverable: ")
    
    add_h3(doc, "Week 9: Physical Heterogeneous Cluster Benchmarking & Automated Test Suite")
    add_p(doc, "To rigorously validate the theoretical models, empirical experiments were conducted directly on a 3-node physical heterogeneous edge testbed:")
    
    # Hardware testbed table
    tbl_hw = doc.add_table(rows=4, cols=5)
    hw_headers = ["Node ID / Role", "Operating System", "Processor (CPU)", "Memory (RAM)", "Accelerator & Usable VRAM"]
    hw_data = [
        ("Node 1 (Host Master)", "Windows 11 AMD64", "Intel Core i7-13700H (14C/20T)", "32 GB DDR5", "NVIDIA RTX 4070 Laptop (8 GB, 7,200 MB usable)"),
        ("Node 2 (Worker Edge)", "macOS Sonoma arm64", "Apple M2 Pro (10 cores)", "16 GB Unified Memory", "Apple M2 Pro GPU 16C (MPS, 10,240 MB usable)"),
        ("Node 3 (Worker Edge)", "Windows 11 AMD64", "Intel Core i5-11400H (6C/12T)", "16 GB DDR4", "NVIDIA RTX 3050 Laptop (4 GB, 3,600 MB usable)")
    ]
    format_table(tbl_hw, [Cm(3.5), Cm(2.8), Cm(3.5), Cm(2.4), Cm(3.8)], hw_headers, hw_data)
    add_p(doc, "Table 1.1: Physical heterogeneous testbed specifications.")
    
    add_p(doc, "Empirical Sub-Studies Execution Highlights:")
    
    # Sub-study 1 Table
    tbl_ss1 = doc.add_table(rows=5, cols=5)
    ss1_headers = ["Model Architecture", "Layer Count", "Measured Oracle Latency", "DP Predicted Median Latency", "Realized Regret"]
    ss1_data = [
        ("Qwen2.5-0.5B", "24 layers", "33.35 ms", "33.044 ms", "0.0 ms (0.0%)"),
        ("Qwen2.5-1.5B", "28 layers", "64.91 ms", "64.675 ms", "0.0 ms (0.0%)"),
        ("Qwen2.5-3B", "36 layers", "161.42 ms", "161.215 ms", "0.0 ms (0.0%)"),
        ("Qwen2.5-7B", "28 layers", "576.71 ms", "576.150 ms", "0.0 ms (0.0%)")
    ]
    format_table(tbl_ss1, [Cm(3.2), Cm(2.4), Cm(3.6), Cm(3.8), Cm(3.0)], ss1_headers, ss1_data)
    add_p(doc, "Table 1.2: Sub-study 1 Partition Quality & Realized Regret across Qwen2.5 model architectures.")
    
    add_bullet(doc, "Sub-study 1 (Partition Quality & Realized Regret): Across all 4 models, DP achieved a Realized Regret of 0.0 ms (0.0%) compared to the exhaustive Measured Oracle. Outperformed Heuristic Equal Partitioning by 7.8% and Memory-Proportional by 17.1% on Qwen2.5-7B.", "• ")
    add_bullet(doc, "Sub-study 2 (Network Sensitivity): Evaluated RTT (0.5–20 ms) and throughput (100 Mbps–1 Gbps), proving tensor activation latency on 1 Gbps LAN is negligible (0.56–2.3 ms).", "• ")
    add_bullet(doc, "Sub-study 3 (KV-Cache Scalability): Verified dynamic working-set expansion across prompt lengths (128–2048 tokens) and batch sizes (B in {1, 2, 4}), proving static models risk severe OOM.", "• ")
    add_bullet(doc, "Sub-study 4 (Security Overhead Ablation): CRC32 and encryption overhead accounted for merely 1.64 ms on 4 MB payloads, sustaining >2,400 MB/s binary bandwidth.", "• ")
    add_bullet(doc, "Sub-study 5 (Thermal Stability): Continuous telemetry showed GPU temperatures stabilized under 57°C with delta T <= 1.0°C (no thermal throttling observed).", "• ")
    add_bullet(doc, "Held-Out Generalization Evaluation: Evaluated calibrated Cost Model on unobserved 1024-token prompts, achieving Mean Absolute Percentage Errors (MAPE) of 1.79% (0.5B) and 2.76% (1.5B).", "• ")
    add_bullet(doc, "Memory Safety Sensitivity (alpha): Demonstrated that alpha = 0.90 is the optimal balance between maximizing single-node memory utilization and buffering against OOM.", "• ")
    add_bullet(doc, "Automated Test Suite: Validated system stability across 10 test modules comprising 46 test cases using pytest: 45/46 tests passed (97.8% pass rate).", "• ")
    add_bullet(doc, "Empirical dataset files (benchmark_results.json, held_out_evaluation_results.json), comprehensive test logs, and the official BKU Midterm Progress Report document (BAO_CAO_TIEN_DO_GIUA_KY_ORCHEST_COMPUTING.docx).", "• Deliverable: ")
    
    # PART II
    add_h1(doc, "PART II: ROADMAP & OBJECTIVES FOR THE REMAINING HALF OF THE SEMESTER (WEEKS 10 – 18)")
    add_p(doc, "The remaining 9 weeks of the semester will be dedicated to Phase P3 (Runtime Optimization, Pipeline Parallelism & Fault Tolerance) and Phase P4 (Scaled Benchmarking, Thesis Writing & Committee Defense).")
    
    # Schedule Gantt Table
    tbl_gantt = doc.add_table(rows=10, cols=5)
    gantt_headers = ["Week", "Milestone / Workstream Topic", "Phase", "Target Schedule", "Target Deliverable"]
    gantt_data = [
        ("Week 10", "Swarm Autoregressive Decode Loop & Distributed KV-Cache", "Phase P3", "Oct 14 – Oct 20", "Persistent KV-cache decode engine"),
        ("Week 11", "Asynchronous Micro-Batching & Multi-Session Scheduling", "Phase P3", "Oct 21 – Oct 27", "1F1B micro-batch pipeline scheduler"),
        ("Week 12", "Session-Boundary Safe Failure & Dynamic Self-Healing", "Phase P3", "Oct 28 – Nov 03", "Auto-repartitioning fault tolerance"),
        ("Week 13", "14B/32B Model Architectures & 4-Bit Quantization", "Phase P3", "Nov 04 – Nov 10", "Large model 64-layer support"),
        ("Week 14", "Production Universal Dashboard Polish & Visualizer", "Phase P3", "Nov 11 – Nov 17", "Real-time SVG/Canvas pipeline UI"),
        ("Week 15", "Scaled 4–5 Node Physical Cluster Stress Testing", "Phase P4", "Nov 18 – Nov 24", "24-hour continuous stress test logs"),
        ("Week 16", "Comprehensive Statistical Analysis & Empirical Evaluation", "Phase P4", "Nov 25 – Dec 01", "P95/P99 latency, ANOVA, Spearman tests"),
        ("Week 17", "Official Thesis Manuscript Writing & Code Packaging", "Phase P4", "Dec 02 – Dec 08", "Full BKU Thesis Manuscript & Repo"),
        ("Week 18", "Defense Preparation, Video Demonstrations & Defense", "Phase P4", "Dec 09 – Dec 15", "Oral Defense before Academic Committee")
    ]
    format_table(tbl_gantt, [Cm(2.0), Cm(4.6), Cm(2.2), Cm(2.8), Cm(4.4)], gantt_headers, gantt_data)
    add_p(doc, "Table 2.1: Master Schedule and Workstream Roadmap for Weeks 10–18.")
    
    add_h2(doc, "Phase P3: Distributed Runtime, Swarm Pipeline Execution & Fault Tolerance (Weeks 10 – 14)")
    
    add_h3(doc, "Week 10: Autoregressive Swarm Generation Loop & Distributed KV-Cache State Preservation")
    add_bullet(doc, "Optimize token-by-token autoregressive decoding across distributed edge nodes in SwarmAutoregressiveGenerator.", "• Primary Objective: ")
    add_bullet(doc, "Implement persistent, distributed KV-cache state preservation across successive decode steps (t > 1) to eliminate redundant computation.", "    1. ")
    add_bullet(doc, "Optimize tensor activation forwarding during single-token decoding steps to minimize Inter-Token Latency (ITL).", "    2. ")
    add_bullet(doc, "Validate end-to-end token generation consistency against standalone single-node PyTorch execution (Cosine Similarity >= 0.9999).", "    3. ")
    add_bullet(doc, "Fully operational multi-node streaming text generation loop with stable distributed KV-cache retention.", "• Milestone / Deliverable: ")
    
    add_h3(doc, "Week 11: Asynchronous Micro-Batching & Multi-Session Pipeline Scheduling")
    add_bullet(doc, "Eliminate pipeline idle bubbles during concurrent multi-user request execution.", "• Primary Objective: ")
    add_bullet(doc, "Implement an asynchronous micro-batching scheduler in micro_batcher.py using a 1F1B (One Forward, One Backward/Decode) scheduling strategy.", "    1. ")
    add_bullet(doc, "Enable token activation interleaving, allowing node k to compute the prefill of request B while node k+1 evaluates the decode step of request A.", "    2. ")
    add_bullet(doc, "Implement backpressure flow control to prevent socket buffer saturation during peak concurrency bursts.", "    3. ")
    add_bullet(doc, "Micro-batching scheduler demonstrating sustained token throughput improvements under concurrent request loads.", "• Milestone / Deliverable: ")
    
    add_h3(doc, "Week 12: Session-Boundary Safe Failure & Dynamic Self-Healing Repartitioning")
    add_bullet(doc, "Ensure robust cluster fault tolerance and graceful recovery when edge nodes unexpectedly disconnect.", "• Primary Objective: ")
    add_bullet(doc, "Finalize the session-boundary failure protocol: when a node heartbeat expires (5 seconds timeout), safely abort the active inference session and cleanly release network buffers.", "    1. ")
    add_bullet(doc, "Implement automated dynamic repartitioning: automatically purge the failed node from the active registry, invoke the DP solver on the remaining healthy nodes, and re-slice the model without requiring system restart.", "    2. ")
    add_bullet(doc, "Perform physical fault-injection tests (simulating abrupt Wi-Fi disconnection and power loss on Worker nodes).", "    3. ")
    add_bullet(doc, "Verified zero-crash self-healing capability with automated repartitioning under physical disconnection tests.", "• Milestone / Deliverable: ")
    
    add_h3(doc, "Week 13: Large Model Architecture Scaling & 4-Bit Weight Quantization")
    add_bullet(doc, "Expand platform support to larger LLMs (14B and 32B parameters) within consumer hardware constraints.", "• Primary Objective: ")
    add_bullet(doc, "Incorporate support for 64-layer architectures (Qwen2.5-14B and Qwen2.5-32B) into auto_slicer.py and the DP solver cost matrix.", "    1. ")
    add_bullet(doc, "Integrate 4-bit weight quantization loading (bitsandbytes NF4 / AWQ / GGUF-equivalent quantization formats), reducing model weight memory footprint from 2.0 bytes to ~0.5 bytes per parameter.", "    2. ")
    add_bullet(doc, "Benchmark working-set memory footprint to verify that a 3-node edge cluster can successfully host a 32B parameter model without triggering OOM.", "    3. ")
    add_bullet(doc, "Validated 14B and 32B model partition plans running safely across the heterogeneous testbed.", "• Milestone / Deliverable: ")
    
    add_h3(doc, "Week 14: Production Universal Dashboard Polish & Real-Time Pipeline Visualizer")
    add_bullet(doc, "Elevate user experience, monitoring transparency, and visual presentation quality.", "• Primary Objective: ")
    add_bullet(doc, "Implement a dynamic SVG/Canvas-based pipeline flow visualizer on the Host Master Web UI, displaying animated token packet transitions across active cluster nodes in real time.", "    1. ")
    add_bullet(doc, "Add real-time telemetry graphs for GPU memory, temperature, and token throughput (tokens/sec) using Chart.js.", "    2. ")
    add_bullet(doc, "Refine the interactive chat playground to support Server-Sent Events (SSE) for smooth, low-latency token streaming.", "    3. ")
    add_bullet(doc, "Polished, production-ready Universal Web UI ready for live defense demonstrations.", "• Milestone / Deliverable: ")
    
    add_h2(doc, "Phase P4: Scaled Benchmarking, Thesis Writing & Committee Defense (Weeks 15 – 18)")
    
    add_h3(doc, "Week 15: Scaled Physical Cluster Stress Testing & Reliability Verification")
    add_bullet(doc, "Validate system performance and stability under extended multi-node operational conditions.", "• Primary Objective: ")
    add_bullet(doc, "Expand the physical testbed from 3 nodes to 4–5 consumer devices in the university laboratory.", "    1. ")
    add_bullet(doc, "Execute a continuous 24-hour automated stress-test suite simulating randomized user prompts, variable prompt lengths, and network jitter.", "    2. ")
    add_bullet(doc, "Monitor long-term system health: confirm zero memory leaks in PyTorch/Metal runtimes and confirm the absence of thermal throttling over extended execution.", "    3. ")
    add_bullet(doc, "Complete 24-hour reliability and stress-test dataset verifying enterprise-grade edge stability.", "• Milestone / Deliverable: ")
    
    add_h3(doc, "Week 16: Comprehensive Statistical Analysis & Empirical Evaluation")
    add_bullet(doc, "Conduct rigorous mathematical and statistical analysis of all experimental datasets.", "• Primary Objective: ")
    add_bullet(doc, "Compute comprehensive statistical metrics: P50, P90, P95, and P99 latency percentiles; token generation throughput (tokens/second); and Time-To-First-Token (TTFT) distributions.", "    1. ")
    add_bullet(doc, "Conduct Analysis of Variance (ANOVA) and Spearman rank correlation between predicted cost model latencies and empirical measurements.", "    2. ")
    add_bullet(doc, "Generate high-resolution, publication-grade vector plots and comparative performance charts.", "    3. ")
    add_bullet(doc, "Comprehensive experimental data chapter with complete statistical validation.", "• Milestone / Deliverable: ")
    
    add_h3(doc, "Week 17: Official Thesis Manuscript Writing & Open-Source Code Packaging")
    add_bullet(doc, "Finalize the complete graduation thesis manuscript and prepare the public open-source release.", "• Primary Objective: ")
    add_bullet(doc, "Author the final Graduation Thesis manuscript in strict compliance with the formatting guidelines of Ho Chi Minh City University of Technology (HCMUT).", "    1. ")
    add_bullet(doc, "Review and refine all chapters: Introduction, Theoretical Foundations, Architectural Design, Empirical Evaluation, and Conclusion.", "    2. ")
    add_bullet(doc, "Clean, document, and package the codebase with automated setup scripts, unit tests, reproducible benchmark scripts, and an English README.md.", "    3. ")
    add_bullet(doc, "Complete Graduation Thesis manuscript and public-ready open-source repository.", "• Milestone / Deliverable: ")
    
    add_h3(doc, "Week 18: Defense Preparation, Video Demonstrations & Final Committee Defense")
    add_bullet(doc, "Successfully defend the capstone project before the Academic Evaluation Committee.", "• Primary Objective: ")
    add_bullet(doc, "Design professional presentation slides summarizing problem motivation, mathematical models, system architecture, empirical discoveries, and future prospects.", "    1. ")
    add_bullet(doc, "Produce a high-definition demonstration video showcasing cluster auto-discovery, dynamic DP repartitioning, and real-time distributed inference.", "    2. ")
    add_bullet(doc, "Conduct full dry-run rehearsals and deliver the final oral defense before the University Evaluation Committee.", "    3. ")
    add_bullet(doc, "Successful Capstone Defense before the Academic Committee.", "• Milestone / Deliverable: ")
    
    # PART III
    add_h1(doc, "PART III: SUMMARY TIMELINE & DELIVERABLES MATRIX")
    add_p(doc, "The comprehensive execution progress and forward deliverables across the 18-week capstone lifecycle are systematically summarized in Table 3.1 below:")
    
    tbl_matrix = doc.add_table(rows=5, cols=4)
    matrix_headers = ["Phase / Period", "Focus & Major Milestones", "Core Tangible Deliverables", "Current Status"]
    matrix_data = [
        ("Phase P1\n(Weeks 1 – 4)",
         "Literature survey; related-work matrix; mathematical modeling of Working-Set & GQA KV-Cache; Cost Model formulation; O(K*N^2) DP algorithm; 18-byte TensorPacket protocol design.",
         "Formal proposal; mathematical model specifications; network framing design; benchmark protocol.",
         "COMPLETED (100%)\nTheoretical models formalized"),
        ("Phase P2\n(Weeks 5 – 9)",
         "Zero-config UDP discovery; raw binary TCP sockets; cross-platform hardware profiler (NVML/MPS); C++ DP solver; AutoModelSlicer; Universal Web Dashboard; 5 physical sub-studies.",
         "Core engine; profiler suite; C++ solver; Glassmorphism dashboard; empirical dataset; 45/46 tests passed.",
         "COMPLETED (100%)\nEmpirically verified on physical testbed"),
        ("Phase P3\n(Weeks 10 – 14)",
         "Autoregressive swarm decode loop; persistent distributed KV-cache; 1F1B micro-batching; session-boundary safe failure; Qwen2.5-14B/32B models; 4-bit quantization; SVG visualizer.",
         "Distributed streaming runtime; micro-batch scheduler; self-healing repartitioner; large-model support; UI visualizer.",
         "UPCOMING (W10 – W14)\nRuntime implementation in progress"),
        ("Phase P4\n(Weeks 15 – 18)",
         "Scaled 4–5 node testbed; 24-hour continuous stress testing; statistical analysis (P95/P99, ANOVA); official HCMUT thesis manuscript; open-source release; committee defense.",
         "24-hour stress dataset; statistical evaluation chapter; final graduation thesis; presentation slides; committee defense.",
         "UPCOMING (W15 – W18)\nScheduled for final submission")
    ]
    format_table(tbl_matrix, [Cm(2.6), Cm(5.2), Cm(5.0), Cm(3.2)], matrix_headers, matrix_data)
    add_p(doc, "Table 3.1: Systematic Project Timeline and Deliverables Matrix.")
    
    # Sign-off box
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(14)
    
    tbl_sign = doc.add_table(rows=2, cols=2)
    tbl_sign.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_sign, val="none")
    
    for row in tbl_sign.rows:
        row.cells[0].width = Cm(8.0)
        row.cells[1].width = Cm(8.0)
        
    p_adv = tbl_sign.rows[0].cells[0].paragraphs[0]
    p_adv.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_adv_t = p_adv.add_run("ACADEMIC ADVISOR\n(Signature and Full Name)\n\n\n\n\n")
    r_adv_t.font.name = "Times New Roman"; r_adv_t.font.size = Pt(11); r_adv_t.font.bold = True
    r_adv_name = p_adv.add_run("Dr. [Advisor Name]")
    r_adv_name.font.name = "Times New Roman"; r_adv_name.font.size = Pt(11); r_adv_name.font.bold = True
    
    p_stu = tbl_sign.rows[0].cells[1].paragraphs[0]
    p_stu.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_stu_t = p_stu.add_run("STUDENTS PERFORMING PROJECT\n(Signatures and Full Names)\n\n\n\n\n")
    r_stu_t.font.name = "Times New Roman"; r_stu_t.font.size = Pt(11); r_stu_t.font.bold = True
    r_stu_names = p_stu.add_run("Tran Thai Tai        Luu Hai Dang")
    r_stu_names.font.name = "Times New Roman"; r_stu_names.font.size = Pt(11); r_stu_names.font.bold = True
    
    p_note = tbl_sign.rows[1].cells[0].paragraphs[0]
    p_note.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_n = p_note.add_run("Report prepared and compiled for the Midterm Evaluation Committee, Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT).")
    r_n.font.name = "Times New Roman"; r_n.font.size = Pt(9.5); r_n.font.italic = True; r_n.font.color.rgb = COLOR_MUTED
    
    print(f"Saving English Word document to: {output_path} ...")
    doc.save(output_path)
    print(f"-> Successfully created English docx: {output_path} ({os.path.getsize(output_path):,} bytes)")

def generate_vietnamese_document(output_path):
    doc = docx.Document()
    setup_page_layout(doc, "BÁO CÁO TIẾN ĐỘ GIỮA KỲ — ORCHEST COMPUTING", page_prefix="Trang ")
    build_academic_header(doc, is_vietnamese=True)
    
    # TỔNG QUAN ĐỀ TÀI
    add_h1(doc, "TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)")
    add_callout(
        doc,
        "Tính đến mốc Giữa kỳ (kết thúc Tuần 9), đề tài nghiên cứu Orchest Computing đã hoàn thành xuất sắc 100% khối lượng công việc được phê duyệt cho Giai đoạn P1 (Khảo sát, Thiết kế & Mô hình hóa toán học) và Giai đoạn P2 (Hiện thực hóa Core Engine & Thực nghiệm đo đạc trên Cụm vật lý dị chủng).\n\n"
        "Toàn bộ mã nguồn cốt lõi gồm ~8,400 dòng lệnh module hóa chuyên nghiệp, đi kèm bộ kiểm thử tự động toàn diện (45/46 ca kiểm thử đạt chuẩn, Pass rate 97.8%). Các chuỗi thực nghiệm đo đạc trực tiếp trên cụm thiết bị biên vật lý dị chủng (NVIDIA RTX 4070 Laptop, Apple M2 Pro và NVIDIA RTX 3050 Laptop) trên 4 kiến trúc mô hình họ Qwen2.5 (0.5B, 1.5B, 3B và 7B) đã chứng minh độ hối tiếc thực tế đạt mức tuyệt đối Realized Regret = 0.0 ms (0.0%) so với nghiệm tối ưu tuyệt đối Measured Oracle, khẳng định tính tối ưu toán học của thuật toán Quy hoạch động (DP Partitioner).",
        title="Các mốc thành tựu nổi bật đến Giữa kỳ"
    )
    add_p(doc, "Dưới đây là phần trình bày chi tiết từng tuần về toàn bộ các cột mốc kỹ thuật và cơ sở lý thuyết đã đạt được trong nửa đầu học kỳ, cùng đề cương mục tiêu và các sản phẩm bàn giao cụ thể cho nửa sau học kỳ.")
    
    # PHẦN I
    add_h1(doc, "PHẦN I: TIẾN ĐỘ VÀ KẾT QUẢ THỰC HIỆN CHI TIẾT TỪNG TUẦN (TUẦN 1 – 9)")
    
    add_h2(doc, "Giai đoạn P1: Khảo sát, Thiết kế và Định thức hóa Mô hình toán học (Tuần 1 – 4)")
    
    add_h3(doc, "Tuần 1: Xác định bài toán, Tổng quan tài liệu có hệ thống & Ma trận so sánh đối chứng")
    add_bullet(doc, "Định thức hóa các thách thức cốt lõi khi thực thi Mô hình Ngôn ngữ Lớn (LLM) trên mạng biên LAN thiết bị cá nhân: 'Bức tường bộ nhớ' vật lý (VRAM hạn chế trên GPU biên), tính dị chủng phần cứng sâu sắc (khác biệt chip tăng tốc, hệ điều hành, cấu trúc bộ nhớ), và biến động đường truyền mạng cục bộ.", "• Định nghĩa bài toán: ")
    add_bullet(doc, "Thực hiện khảo sát đối chiếu có hệ thống trên các kiến trúc suy luận phân tán tiêu biểu:", "• Khảo sát tài liệu đối chứng: ")
    add_bullet(doc, "PipeEdge: Áp dụng phân hoạch tĩnh, thiếu cơ chế mô hình hóa động KV-cache.", "    - ")
    add_bullet(doc, "exo: Tự động khám phá P2P nhưng thiếu sự phân tách rạch ròi giữa hai pha Prefill và Decode trong mô hình chi phí.", "    - ")
    add_bullet(doc, "Petals: Phân tán dạng bầy đàn hướng tới Internet công cộng thay vì tối ưu độ trễ thấp xác định trong mạng nội bộ LAN.", "    - ")
    add_bullet(doc, "PiPPy: Kiến trúc pipeline cấp máy chủ đám mây, không tương thích với ngân sách bộ nhớ bất đối xứng của thiết bị cá nhân.", "    - ")
    add_bullet(doc, "Xây dựng Ma trận So sánh Đối chứng Hệ thống nhận diện khoảng trống nghiên cứu, xác lập ranh giới phạm vi MVP cho Orchest Computing.", "• Sản phẩm bàn giao: ")
    
    add_h3(doc, "Tuần 2: Mô hình hóa Toán học Bộ nhớ Working-Set & Co giãn Động KV-Cache")
    add_bullet(doc, "Xác lập phương trình tính toán tổng nhu cầu bộ nhớ thời gian chạy trên từng nút phần cứng:", "• Phân rã Working-Set bộ nhớ: ")
    add_callout(doc, "M_k^ws = M_k^weight + M_k^kv(S, B) + M_k^act(S, B) + M_k^overhead", title="Công thức (2.1): Bộ nhớ Working-Set thời gian chạy trên nút k")
    add_bullet(doc, "Xây dựng mô hình tính toán bộ nhớ động phụ thuộc chặt chẽ vào cấu trúc cơ chế chú ý nhóm đa truy vấn (Grouped-Query Attention):", "• Mô hình toán học KV-Cache GQA: ")
    add_callout(doc, "Mem_KV_per_layer = 2 * n_kv_heads * d_head * S_max * B * bytes_per_elem", title="Công thức (2.2): Sự biến thiên bộ nhớ KV-Cache theo Layer")
    add_bullet(doc, "Thiết lập ngưỡng an toàn bộ nhớ M_k^ws <= alpha * M_k,usable với hệ số an toàn alpha in [0.80, 0.95] nhằm triệt tiêu hoàn toàn sự cố tràn bộ nhớ (Out-Of-Memory - OOM) khi độ dài ngữ cảnh mở rộng.", "• Thiết lập ngưỡng trần an toàn: ")
    add_bullet(doc, "Tài liệu đặc tả mô hình toán học và bản mẫu Python kiểm chứng độ chính xác tính toán KV-cache trên các mô hình Hugging Face thực tế (Qwen/Qwen2.5).", "• Sản phẩm bàn giao: ")
    
    add_h3(doc, "Tuần 3: Mô hình Chi phí Hai pha & Thuật toán Quy hoạch động (DP) Phân hoạch Liên tục")
    add_bullet(doc, "Tách biệt rõ ràng chu trình suy luận thành hai chế độ thực thi chuyên biệt:", "• Phân tách chế độ thực thi: ")
    add_bullet(doc, "Pha Prefill: Bị nghẽn bởi năng lực tính toán (Compute-bound), xử lý song song các token ngữ cảnh đầu vào, đo bằng Thời gian đến Token đầu tiên (TTFT).", "    1. ")
    add_bullet(doc, "Pha Decode: Bị nghẽn bởi băng thông bộ nhớ và truyền thông mạng (Memory-bandwidth/network-bound), sinh token tự hồi quy từng bước, đo bằng Độ trễ giữa các Token (ITL).", "    2. ")
    add_bullet(doc, "Xây dựng ma trận chi phí tính toán T_comp(i, j, k) qua hiệu năng TFLOPs thực đo và chi phí truyền tensor kích hoạt T_comm(k, k+1) qua RTT và băng thông mạng.", "• Xây dựng ma trận chi phí: ")
    add_bullet(doc, "Thiết kế thuật toán Quy hoạch động tìm phân rã tối ưu với độ phức tạp O(K * N^2) trên N layer và K máy tính, kèm ràng buộc phân hoạch liên tục và khả thi bộ nhớ:", "• Thuật toán Quy hoạch động DP: ")
    add_callout(doc, "DP[k][j] = min_{0 <= i <= j} { DP[k-1][i] + T_comp(i+1, j, k) + T_comm(k-1, k) }", title="Công thức (2.3): Phương trình Truy hồi Quy hoạch động")
    add_bullet(doc, "Chứng minh tính liên tục của phân đoạn layer, loại bỏ triệt để hiện tượng vòng lặp tự chuyển tiếp (self-transition loophole) và mã nguồn DP cơ sở.", "• Sản phẩm bàn giao: ")
    
    add_h3(doc, "Tuần 4: Đặc tả Giao thức Mạng Nhị phân, Kiến trúc An ninh & Đề cương Đo đạc Thực nghiệm")
    add_bullet(doc, "Thiết kế giao thức đóng khung nhị phân tiêu đề cố định 18 bytes (!4sHIII: Magic Bytes 'ORCH', Packet Type, Sequence ID, Payload Length, Checksum CRC32) cho phép truyền trực tiếp bộ đệm Tensor PyTorch không qua tuần tự hóa trung gian.", "• Đóng khung nhị phân TensorPacket: ")
    add_bullet(doc, "Đặc tả xác thực Mutual TLS (mTLS) với chứng chỉ số X.509 riêng biệt cho từng nút (ca.crt, host.crt, worker.crt) và tùy chọn mã hóa luồng dữ liệu AES-GCM 128-bit.", "• Kiến trúc an ninh biên: ")
    add_bullet(doc, "Xây dựng phương pháp luận đo đạc gồm 5 chuỗi thực nghiệm chuyên sâu, xác lập giao thức đối chiếu giữa thuật toán DP và nghiệm vét cạn Measured Oracle để tính toán Độ hối tiếc (Realized Regret).", "• Đề cương thực nghiệm khoa học: ")
    add_bullet(doc, "Hồ sơ đặc tả giao thức truyền nhận và đề cương benchmark chuẩn hóa (tương ứng Chương 2 và Chương 3 trong hồ sơ báo cáo giữa kỳ).", "• Sản phẩm bàn giao: ")
    
    add_h2(doc, "Giai đoạn P2: Hiện thực hóa Core Engine & Thực nghiệm Cụm vật lý (Tuần 5 – 9)")
    
    add_h3(doc, "Tuần 5: Phân hệ Mạng và Khám phá Tự động Bầy đàn Không Cấu hình (Zero-Config Swarm)")
    add_bullet(doc, "Hiện thực module orchest.network.discovery cho phép các Worker lắng nghe cổng 0.0.0.0:50005 và tự động đăng ký với Host Master qua cơ chế UDP broadcast (255.255.255.255), trích xuất IP LAN thực tế từ header socket.", "• Khám phá tự động qua UDP Beacon: ")
    add_bullet(doc, "Xây dựng orchest.network.transport với socket TCP tùy biến cao, đạt băng thông socket thô vượt mốc 2,400 MB/s trên đường truyền nội bộ.", "• Giao vận nhị phân tốc độ cao: ")
    add_bullet(doc, "Bổ sung các bộ tương thích trừu tượng hỗ trợ linh hoạt Raw TCP, gRPC (tensor_service.proto) và ZeroMQ (zmq_transport.py).", "• Đa tầng giao vận: ")
    add_bullet(doc, "Xây dựng luồng Heartbeat nền (chu kỳ 2 giây, ngưỡng timeout 5 giây) để chủ động giám sát trạng thái sức khỏe từng nút trong cụm.", "• Giám sát trạng thái cụm: ")
    add_bullet(doc, "Phân hệ mạng hoàn chỉnh (discovery.py, transport.py, protocol.py) đáp ứng khả năng gom cụm tự động không cấu hình.", "• Sản phẩm bàn giao: ")
    
    add_h3(doc, "Tuần 6: Đo kiểm Phần cứng Đa nền tảng & Phân hệ Đo lường từ xa (Telemetry)")
    add_bullet(doc, "Nền tảng Windows: Tích hợp pynvml để truy vấn VRAM, compute capability, nhiệt độ và công suất GPU NVIDIA; tích hợp Windows WMI và psutil cho CPU/RAM.", "• Đo kiểm phần cứng (Windows): ")
    add_bullet(doc, "Nền tảng macOS: Xây dựng logic truy vấn bản địa cấu trúc Unified Memory trên Apple Silicon và Metal Performance Shaders (MPS), tính toán chuẩn xác bộ nhớ khả dụng mà không bị nhầm lẫn với VRAM rời.", "• Đo kiểm phần cứng (macOS): ")
    add_bullet(doc, "Xây dựng bộ đo kiểm GEMM (General Matrix Multiply) chạy phép nhân ma trận FP16 trực tiếp trên bộ tăng tốc (CUDA / MPS) để đo TFLOPs thực dụng thay vì dựa vào thông số quảng cáo lý thuyết.", "• Đo kiểm TFLOPs thực dụng: ")
    add_bullet(doc, "Đo tự động độ trễ khứ hồi RTT và băng thông truyền tải giữa các cặp nút trong mạng nội bộ.", "• Đo lường mạng LAN: ")
    add_bullet(doc, "Phân hệ đo kiểm và viễn thám xuất dữ liệu có cấu trúc ra hardware_profile.json và calibration_profile.json.", "• Sản phẩm bàn giao: ")
    
    add_h3(doc, "Tuần 7: Bộ giải Quy hoạch động (Python & C++ Engine) & AutoModelSlicer")
    add_bullet(doc, "Tích hợp bộ cân chỉnh mô hình chi phí, đưa các tham số phần cứng thực tế vào ma trận để sinh kế hoạch phân rã layer tối ưu.", "• Bộ giải DP cân chỉnh: ")
    add_bullet(doc, "Tái hiện thực thuật toán DP bằng ngôn ngữ C++ bản địa sử dụng mảng phẳng 1 chiều liên tục và tính toán song song OpenMP. Biên dịch thành thư viện động (.dll trên Windows, .dylib trên macOS), gọi qua Python ctypes, giảm thời gian giải thuật từ 15 ms xuống dưới 0.3 ms.", "• Tăng tốc C++ Core Engine: ")
    add_bullet(doc, "Universal AutoModelSlicer: Áp dụng cơ chế reflection bóc tách cấu trúc layer của Transformer (.model.layers cho Qwen/LLaMA); xây dựng ModelSliceModule gói gọn layer được gán kèm Embeddings hoặc RMSNorm + LM-Head; nạp trọng số sharded chọn lọc lên VRAM chống lãng phí bộ nhớ.", "• Bộ cắt mô hình tự động: ")
    add_bullet(doc, "Bộ giải DP hiệu năng cao và module cắt mô hình vượt qua toàn bộ unit test độc lập.", "• Sản phẩm bàn giao: ")
    
    add_h3(doc, "Tuần 8: Ứng dụng Hợp nhất Web Application & Bảng điều khiển Glassmorphism")
    add_bullet(doc, "Đóng gói toàn bộ nền tảng thành một ứng dụng thống nhất cho phép bất kỳ máy tính nào cũng có thể linh hoạt chuyển đổi giữa vai trò Host Master và Worker Edge.", "• Bộ khởi động hợp nhất: ")
    add_bullet(doc, "Xây dựng giao diện web phong cách Glassmorphism hiện đại với 3 tab chức năng rạch ròi:", "• Backend FastAPI & Giao diện Trực quan: ")
    add_bullet(doc, "Tab Host Master: Quét cụm, hiển thị telemetry các nút, chọn mô hình LLM, kích hoạt giải thuật DP tự động, và Playground tương tác thử nghiệm.", "    1. ")
    add_bullet(doc, "Tab Worker Edge: Giám sát tài nguyên thời gian thực, kiểm tra lát cắt được phân bổ, banner chấp thuận/từ chối, và nhật ký kết nối.", "    2. ")
    add_bullet(doc, "Tab Manage Fleet: Đặt tên gợi nhớ cho máy, đo ping độ trễ từng nút, bật/tắt tham gia Swarm, và dọn dẹp nút ngắt kết nối.", "    3. ")
    add_bullet(doc, "Hiện thực thanh trượt tự động cân bằng phân bổ layer trên giao diện, bảo đảm tổng số layer gán trên các nút luôn bằng 100% khi người dùng điều chỉnh thủ công.", "• Thanh trượt tự cân bằng (Tổng = 100%): ")
    add_bullet(doc, "Giao diện web hoàn chỉnh, tương thích mọi thiết bị (server.py, app.js, index.html, styles.css).", "• Sản phẩm bàn giao: ")
    
    add_h3(doc, "Tuần 9: Thực nghiệm Đo đạc trên Cụm Vật lý Dị chủng & Bộ Kiểm thử Tự động")
    add_p(doc, "Để kiểm chứng chặt chẽ các mô hình toán học và thuật toán, nhóm đã tiến hành đo đạc trực tiếp trên cụm thực nghiệm vật lý gồm 3 máy tính cá nhân dị chủng:")
    
    # Bảng phần cứng VN
    tbl_hw_vn = doc.add_table(rows=4, cols=5)
    hw_headers_vn = ["Nút / Vai trò", "Hệ điều hành", "Vi xử lý (CPU)", "Bộ nhớ RAM", "Bộ tăng tốc & VRAM Khả dụng"]
    hw_data_vn = [
        ("Nút 1 (Host Master)", "Windows 11 AMD64", "Intel Core i7-13700H (14C/20T)", "32 GB DDR5", "NVIDIA RTX 4070 Laptop (8 GB, 7,200 MB khả dụng)"),
        ("Nút 2 (Worker Edge)", "macOS Sonoma arm64", "Apple M2 Pro (10 nhân)", "16 GB Unified Memory", "Apple M2 Pro GPU 16C (MPS, 10,240 MB khả dụng)"),
        ("Nút 3 (Worker Edge)", "Windows 11 AMD64", "Intel Core i5-11400H (6C/12T)", "16 GB DDR4", "NVIDIA RTX 3050 Laptop (4 GB, 3,600 MB khả dụng)")
    ]
    format_table(tbl_hw_vn, [Cm(3.5), Cm(2.8), Cm(3.5), Cm(2.4), Cm(3.8)], hw_headers_vn, hw_data_vn)
    add_p(doc, "Bảng 1.1: Thông số kỹ thuật của 3 nút phần cứng dị chủng trong cụm thực nghiệm vật lý.")
    
    add_p(doc, "Kết quả thực nghiệm nổi bật:")
    
    # Bảng Sub-study 1 VN
    tbl_ss1_vn = doc.add_table(rows=5, cols=5)
    ss1_headers_vn = ["Kiến trúc Mô hình", "Số Layer", "Độ trễ Oracle Đo đạc", "Độ trễ DP Dự đoán (Median)", "Độ hối tiếc (Regret)"]
    ss1_data_vn = [
        ("Qwen2.5-0.5B", "24 layers", "33.35 ms", "33.044 ms", "0.0 ms (0.0%)"),
        ("Qwen2.5-1.5B", "28 layers", "64.91 ms", "64.675 ms", "0.0 ms (0.0%)"),
        ("Qwen2.5-3B", "36 layers", "161.42 ms", "161.215 ms", "0.0 ms (0.0%)"),
        ("Qwen2.5-7B", "28 layers", "576.71 ms", "576.150 ms", "0.0 ms (0.0%)")
    ]
    format_table(tbl_ss1_vn, [Cm(3.2), Cm(2.4), Cm(3.6), Cm(3.8), Cm(3.0)], ss1_headers_vn, ss1_data_vn)
    add_p(doc, "Bảng 1.2: Kết quả đánh giá chất lượng phân hoạch DP và độ hối tiếc thực tế đối chiếu Measured Oracle.")
    
    add_bullet(doc, "Thực nghiệm 1 (Chất lượng Phân hoạch & Độ hối tiếc): Trên toàn bộ 4 mô hình, DP đạt mức Realized Regret = 0.0 ms (0.0%) so với nghiệm tối ưu Measured Oracle. Vượt trội hơn Heuristic Chia đều 7.8% và Heuristic Tỷ lệ bộ nhớ 17.1% trên mô hình Qwen2.5-7B.", "• ")
    add_bullet(doc, "Thực nghiệm 2 (Độ nhạy Mạng): Khảo sát RTT (0.5–20 ms) và băng thông (100 Mbps–1 Gbps), chứng minh độ trễ truyền tensor kích hoạt trên mạng LAN 1 Gbps là không đáng kể (0.56–2.3 ms).", "• ")
    add_bullet(doc, "Thực nghiệm 3 (Co giãn KV-Cache): Đo đạc sự phình to của Working-Set khi mở rộng độ dài prompt (128–2048 tokens) và kích thước batch (B in {1, 2, 4}), chứng minh mô hình tĩnh sẽ gây tràn VRAM sập hệ thống.", "• ")
    add_bullet(doc, "Thực nghiệm 4 (Phụ tải An ninh & Giao vận): Phụ tải checksum CRC32 và mã hóa chỉ chiếm 1.64 ms trên gói tin 4 MB, duy trì băng thông socket thực tế trên 2,400 MB/s.", "• ")
    add_bullet(doc, "Thực nghiệm 5 (Độ ổn định Nhiệt độ): Giám sát telemetry liên tục xác nhận nhiệt độ GPU ổn định dưới 57°C với delta T <= 1.0°C (không xuất hiện hiện tượng quá nhiệt giảm xung).", "• ")
    add_bullet(doc, "Đánh giá Khả năng Khái quát hóa: Đánh giá mô hình chi phí trên tập dữ liệu chưa từng quan sát (prompt 1024 tokens), sai số phần trăm tuyệt đối trung bình (MAPE) chỉ đạt 1.79% (0.5B) và 2.76% (1.5B).", "• ")
    add_bullet(doc, "Độ nhạy Hệ số An toàn Bộ nhớ: Xác định alpha = 0.90 là điểm cân bằng hoàn hảo giữa việc tối đa hóa tận dụng VRAM đơn máy và bảo vệ chống tràn bộ nhớ.", "• ")
    add_bullet(doc, "Bộ Kiểm thử Tự động: Đánh giá độ tin cậy hệ thống qua 10 file test với 46 ca kiểm thử bằng pytest: 45/46 ca kiểm thử đạt chuẩn (Pass rate: 97.8%).", "• ")
    add_bullet(doc, "Tập dữ liệu thực nghiệm (benchmark_results.json, held_out_evaluation_results.json), log kiểm thử và tài liệu báo cáo Word chính thức (BAO_CAO_TIEN_DO_GIUA_KY_ORCHEST_COMPUTING.docx).", "• Sản phẩm bàn giao: ")
    
    # PHẦN II
    add_h1(doc, "PHẦN II: KẾ HOẠCH VÀ MỤC TIÊU CHO NỬA SAU HỌC KỲ (TUẦN 10 – 18)")
    add_p(doc, "9 tuần tiếp theo của học kỳ được phân bổ trọng tâm cho Giai đoạn P3 (Tối ưu hóa Runtime, Song song đường ống & Chịu lỗi) và Giai đoạn P4 (Thực nghiệm mở rộng, Viết Luận văn & Bảo vệ trước Hội đồng).")
    
    # Bảng Gantt VN
    tbl_gantt_vn = doc.add_table(rows=10, cols=5)
    gantt_headers_vn = ["Tuần", "Hạng mục Công việc Trọng tâm", "Giai đoạn", "Thời gian Dự kiến", "Sản phẩm Giao nộp Kỳ vọng"]
    gantt_data_vn = [
        ("Tuần 10", "Chu trình Giải mã Tự hồi quy & KV-Cache Phân tán", "Giai đoạn P3", "14/10 – 20/10", "Engine giải mã token kèm lưu vết KV-cache"),
        ("Tuần 11", "Gom cụm Bất đồng bộ Micro-Batching & Lập lịch 1F1B", "Giai đoạn P3", "21/10 – 27/10", "Bộ lập lịch pipeline micro-batching"),
        ("Tuần 12", "Chịu lỗi Cấp phiên & Tự động Tái phân rã (Self-Healing)", "Giai đoạn P3", "28/10 – 03/11", "Cơ chế chịu lỗi tự phân hoạch lại"),
        ("Tuần 13", "Mở rộng Mô hình 14B/32B & Lượng tử hóa 4-Bit", "Giai đoạn P3", "04/11 – 10/11", "Hỗ trợ mô hình 64 layers trên cụm biên"),
        ("Tuần 14", "Tinh chỉnh Universal Dashboard & Trực quan hóa Pipeline", "Giai đoạn P3", "11/11 – 17/11", "Giao diện SVG/Canvas luồng token thời gian thực"),
        ("Tuần 15", "Mở rộng Thử nghiệm Cụm 4–5 Nút & Stress Test 24 Giờ", "Giai đoạn P4", "18/11 – 24/11", "Tập dữ liệu kiểm thử tải áp lực 24h"),
        ("Tuần 16", "Phân tích Thống kê Chuyên sâu & Đánh giá Hiệu năng", "Giai đoạn P4", "25/11 – 01/12", "Chỉ số P95/P99, kiểm định ANOVA, Spearman"),
        ("Tuần 17", "Hoàn thiện Luận văn Tốt nghiệp & Đóng gói Mã nguồn mở", "Giai đoạn P4", "02/12 – 08/12", "Bản thảo Luận văn BKU & Repo hoàn chỉnh"),
        ("Tuần 18", "Tập dượt Báo cáo, Video Demo & Bảo vệ Hội đồng", "Giai đoạn P4", "09/12 – 15/12", "Bảo vệ thành công trước Hội đồng Chấm ĐATN")
    ]
    format_table(tbl_gantt_vn, [Cm(2.0), Cm(4.6), Cm(2.2), Cm(2.8), Cm(4.4)], gantt_headers_vn, gantt_data_vn)
    add_p(doc, "Bảng 2.1: Sơ đồ Tiến độ và Kế hoạch Triển khai Master Roadmap cho Tuần 10–18.")
    
    add_h2(doc, "Giai đoạn P3: Runtime Phân tán, Thực thi Swarm Pipeline & Cơ chế Chịu lỗi (Tuần 10 – 14)")
    
    add_h3(doc, "Tuần 10: Tối ưu Chu trình Sinh Token Tự hồi quy & Duy trì Trạng thái KV-Cache Phân tán")
    add_bullet(doc, "Tối ưu hóa chu trình giải mã tự hồi quy từng token qua các nút biên phân tán trong SwarmAutoregressiveGenerator.", "• Mục tiêu chính: ")
    add_bullet(doc, "Hiện thực hóa cơ chế duy trì trạng thái bộ nhớ đệm KV-cache phân tán bền vững qua các bước decode kế tiếp (t > 1) để triệt tiêu việc tính toán lặp lại.", "    1. ")
    add_bullet(doc, "Tối ưu hóa việc chuyển tiếp tensor kích hoạt trong từng bước giải mã đơn lẻ để giảm thiểu tối đa Độ trễ giữa các token (ITL).", "    2. ")
    add_bullet(doc, "Kiểm chứng tính nhất quán về mặt số học của chuỗi token sinh ra so với việc chạy trên một máy PyTorch đơn lẻ (Cosine Similarity >= 0.9999).", "    3. ")
    add_bullet(doc, "Chu trình sinh văn bản streaming đa nút hoạt động ổn định và chính xác kèm cơ chế giữ KV-cache phân tán.", "• Cột mốc / Sản phẩm: ")
    
    add_h3(doc, "Tuần 11: Gom cụm Kích hoạt Vi mô Bất đồng bộ (Micro-Batching) & Lập lịch Đa phiên")
    add_bullet(doc, "Triệt tiêu các 'bong bóng nhàn rỗi' (pipeline idle bubbles) trong đường ống khi phục vụ nhiều yêu cầu người dùng đồng thời.", "• Mục tiêu chính: ")
    add_bullet(doc, "Xây dựng bộ điều phối micro-batching bất đồng bộ trong micro_batcher.py áp dụng chiến lược lập lịch 1F1B (Một bước Forward, Một bước Decode).", "    1. ")
    add_bullet(doc, "Cho phép đan xen kích hoạt token, giúp nút k có thể xử lý prefill của yêu cầu B trong khi nút k+1 đang tính decode của yêu cầu A.", "    2. ")
    add_bullet(doc, "Cài đặt cơ chế kiểm soát dòng dữ liệu ngược (backpressure) chống tràn bộ đệm socket khi lưu lượng yêu cầu tăng đột biến.", "    3. ")
    add_bullet(doc, "Bộ lập lịch micro-batching cải thiện rõ rệt thông lượng token tổng thể dưới tải đồng thời.", "• Cột mốc / Sản phẩm: ")
    
    add_h3(doc, "Tuần 12: Chịu lỗi An toàn Cấp phiên (Session-Boundary) & Tự động Tái phân rã Tự phục hồi")
    add_bullet(doc, "Bảo đảm khả năng chịu lỗi bền bỉ và phục hồi mượt mà khi các nút biên đột ngột ngắt kết nối mạng hoặc tắt nguồn.", "• Mục tiêu chính: ")
    add_bullet(doc, "Hoàn thiện giao thức xử lý sự cố cấp phiên: khi heartbeat của một nút hết hạn (5 giây timeout), hủy phiên an toàn và giải phóng bộ đệm mạng sạch sẽ.", "    1. ")
    add_bullet(doc, "Hiện thực cơ chế tự động tái phân rã động: tự động loại bỏ nút hỏng khỏi danh bạ hoạt động, kích hoạt thuật toán DP trên các nút lành lặn còn lại và cắt lại mô hình mà không cần khởi động lại toàn hệ thống.", "    2. ")
    add_bullet(doc, "Tiến hành các ca thử nghiệm bơm lỗi vật lý (ngắt đột ngột cáp Wi-Fi và tắt nguồn máy Worker).", "    3. ")
    add_bullet(doc, "Xác nhận khả năng tự phục hồi không gây crash ứng dụng và tự động phân rã lại thành công dưới kiểm thử ngắt kết nối vật lý.", "• Cột mốc / Sản phẩm: ")
    
    add_h3(doc, "Tuần 13: Mở rộng Hỗ trợ Kiến trúc Mô hình Lớn & Lượng tử hóa Trọng số 4-Bit")
    add_bullet(doc, "Mở rộng khả năng hỗ trợ các mô hình LLM quy mô lớn hơn (14B và 32B tham số) trong giới hạn tài nguyên phần cứng cá nhân.", "• Mục tiêu chính: ")
    add_bullet(doc, "Bổ sung cấu hình cho các kiến trúc 64 layers (Qwen2.5-14B và Qwen2.5-32B) vào auto_slicer.py và ma trận chi phí của bộ giải DP.", "    1. ")
    add_bullet(doc, "Tích hợp cơ chế nạp lượng tử hóa trọng số 4-bit (NF4 / AWQ / tương đương GGUF), giảm dung lượng bộ nhớ trọng số từ 2.0 bytes xuống ~0.5 byte cho mỗi tham số.", "    2. ")
    add_bullet(doc, "Đo đạc Working-Set bộ nhớ để kiểm chứng cụm 3 máy biên có thể chứa và thực thi an toàn mô hình 32B mà không sập OOM.", "    3. ")
    add_bullet(doc, "Kế hoạch phân hoạch mô hình 14B và 32B được kiểm chứng vận hành an toàn trên cụm dị chủng.", "• Cột mốc / Sản phẩm: ")
    
    add_h3(doc, "Tuần 14: Tinh chỉnh Toàn diện Universal Dashboard & Bộ Trực quan hóa Đường ống Thời gian thực")
    add_bullet(doc, "Nâng cao trải nghiệm người dùng, tính minh bạch trong giám sát và thẩm mỹ giao diện báo cáo demo.", "• Mục tiêu chính: ")
    add_bullet(doc, "Xây dựng bộ visualizer luồng pipeline động bằng SVG/Canvas trên Web UI của Host Master, mô phỏng chuyển động của các gói tin token qua các nút cụm theo thời gian thực.", "    1. ")
    add_bullet(doc, "Bổ sung biểu đồ viễn thám thời gian thực cho bộ nhớ GPU, nhiệt độ và thông lượng token (tokens/giây) bằng Chart.js.", "    2. ")
    add_bullet(doc, "Hoàn thiện khu vực chat playground hỗ trợ Server-Sent Events (SSE) để dòng chữ streaming mượt mà với độ trễ tối thiểu.", "    3. ")
    add_bullet(doc, "Giao diện Web UI chuyên nghiệp, sẵn sàng cho các buổi báo cáo và trình diễn trực tiếp trước hội đồng.", "• Cột mốc / Sản phẩm: ")
    
    add_h2(doc, "Giai đoạn P4: Thực nghiệm Mở rộng, Viết Luận văn & Bảo vệ Hội đồng (Tuần 15 – 18)")
    
    add_h3(doc, "Tuần 15: Kiểm thử Áp lực (Stress-Testing) trên Cụm Mở rộng & Kiểm chứng Độ tin cậy")
    add_bullet(doc, "Đánh giá hiệu năng và độ ổn định lâu dài của hệ thống trong điều kiện vận hành đa nút mở rộng liên tục.", "• Mục tiêu chính: ")
    add_bullet(doc, "Mở rộng cụm thử nghiệm vật lý từ 3 nút lên 4–5 thiết bị cá nhân tại phòng thí nghiệm của trường.", "    1. ")
    add_bullet(doc, "Chạy bộ kiểm thử áp lực tự động liên tục trong 24 giờ với các câu hỏi ngẫu nhiên, độ dài prompt biến thiên và tạo nhiễu mạng.", "    2. ")
    add_bullet(doc, "Giám sát sức khỏe hệ thống: xác nhận không có hiện tượng rò rỉ bộ nhớ (memory leaks) trong runtime PyTorch/Metal và không xảy ra hiện tượng giảm xung do nhiệt.", "    3. ")
    add_bullet(doc, "Tập dữ liệu kiểm thử áp lực và độ tin cậy 24 giờ hoàn chỉnh, chứng minh độ ổn định cấp độ thực tế.", "• Cột mốc / Sản phẩm: ")
    
    add_h3(doc, "Tuần 16: Phân tích Thống kê Chuyên sâu & Đánh giá Thực nghiệm Khoa học")
    add_bullet(doc, "Tiến hành phân tích toán học và thống kê học nghiêm ngặt trên toàn bộ các tập dữ liệu thực nghiệm đã thu thập.", "• Mục tiêu chính: ")
    add_bullet(doc, "Tính toán các chỉ số thống kê phân phối: phân vị độ trễ P50, P90, P95, P99; thông lượng sinh token (tokens/giây); và phân phối thời gian TTFT.", "    1. ")
    add_bullet(doc, "Thực hiện Phân tích Phương sai (ANOVA) và hệ số tương quan thứ hạng Spearman giữa độ trễ dự đoán bởi mô hình chi phí và độ trễ thực đo.", "    2. ")
    add_bullet(doc, "Xuất các biểu đồ vector độ phân giải cao phục vụ cho ấn phẩm khoa học và luận văn tốt nghiệp.", "    3. ")
    add_bullet(doc, "Chương báo cáo kết quả thực nghiệm với đầy đủ phân tích thống kê toán học.", "• Cột mốc / Sản phẩm: ")
    
    add_h3(doc, "Tuần 17: Soạn thảo Bản thảo Luận văn Chính thức & Đóng gói Mã nguồn Mở")
    add_bullet(doc, "Hoàn thiện trọn vẹn bản thảo Luận văn Tốt nghiệp và chuẩn bị phát hành kho mã nguồn mở ra cộng đồng.", "• Mục tiêu chính: ")
    add_bullet(doc, "Biên soạn bản thảo Luận văn Tốt nghiệp chính thức tuân thủ tuyệt đối quy chuẩn định dạng của Trường Đại học Bách Khoa – ĐHQG-HCM.", "    1. ")
    add_bullet(doc, "Rà soát và tinh chỉnh toàn diện các chương: Mở đầu, Cơ sở lý thuyết, Thiết kế kiến trúc, Đánh giá thực nghiệm và Kết luận kiến nghị.", "    2. ")
    add_bullet(doc, "Làm sạch, chú thích mã nguồn, đóng gói repo mã nguồn mở kèm script cài đặt tự động, bộ unit test, script tái hiện kết quả và tài liệu hướng dẫn README.md.", "    3. ")
    add_bullet(doc, "Bản thảo Luận văn Tốt nghiệp hoàn chỉnh và kho mã nguồn công khai sẵn sàng chuyển giao.", "• Cột mốc / Sản phẩm: ")
    
    add_h3(doc, "Tuần 18: Chuẩn bị Báo cáo, Video Demo & Bảo vệ Đồ án Tốt nghiệp trước Hội đồng")
    add_bullet(doc, "Bảo vệ thành công xuất sắc đề tài đồ án tốt nghiệp trước Hội đồng Đánh giá Chuyên môn.", "• Mục tiêu chính: ")
    add_bullet(doc, "Thiết kế slide báo cáo tổng kết chuyên nghiệp làm nổi bật tính cấp thiết, mô hình toán, kiến trúc hệ thống, phát hiện thực nghiệm và định hướng tương lai.", "    1. ")
    add_bullet(doc, "Sản xuất video demo độ phân giải cao minh họa các tính năng tự khám phá cụm, tự động phân rã DP và suy luận phân tán thời gian thực.", "    2. ")
    add_bullet(doc, "Tập dượt phản biện toàn diện và thực hiện buổi bảo vệ chính thức trước Hội đồng Đánh giá của Khoa/Bộ môn.", "    3. ")
    add_bullet(doc, "Bảo vệ thành công Đồ án Tốt nghiệp trước Hội đồng Đánh giá.", "• Cột mốc / Sản phẩm: ")
    
    # PHẦN III
    add_h1(doc, "PHẦN III: TỔNG HỢP TIẾN ĐỘ VÀ MA TRẬN SẢN PHẨM BÀN GIAO")
    add_p(doc, "Bảng ma trận 3.1 dưới đây tổng hợp bức tranh toàn cảnh về tiến độ triển khai và các sản phẩm giao nộp tương ứng qua toàn bộ 18 tuần của Đồ án Tốt nghiệp:")
    
    tbl_matrix_vn = doc.add_table(rows=5, cols=4)
    matrix_headers_vn = ["Giai đoạn / Thời gian", "Mục tiêu và Nhiệm vụ Trọng tâm", "Sản phẩm Giao nộp Cụ thể", "Tình trạng Thực tế"]
    matrix_data_vn = [
        ("Giai đoạn P1\n(Tuần 1 – 4)",
         "Khảo sát tài liệu; lập ma trận related work; định thức hóa mô hình Working-Set và KV-Cache GQA; thiết kế thuật toán DP O(K*N^2); đặc tả giao thức mạng nhị phân.",
         "Bản đề cương chi tiết; đặc tả Cost Model; đặc tả giao thức truyền nhận; kế hoạch benchmark.",
         "ĐÃ HOÀN THÀNH 100%\n(Hồ sơ thiết kế và mô hình toán hoàn thiện)"),
        ("Giai đoạn P2\n(Tuần 5 – 9)",
         "Xây dựng Auto Discovery UDP, Node Registry, Heartbeat; viết module Profiler phần cứng (NVML, WMI, MPS); xây dựng Cost Model Calibrator; hiện thực DP Solver (Python & C++); sinh chứng chỉ mTLS, mã hóa CRC32; xây dựng Dashboard Web UI; chạy 5 Sub-studies thực nghiệm.",
         "Core Engine hoàn chỉnh; module Profiler; Cost Model đã cân chỉnh; module C++ DP Solver; Dashboard Web 3 Tab; bộ dữ liệu thực nghiệm 5 Sub-studies; 45/46 tests passed.",
         "ĐÃ HOÀN THÀNH 100%\n(Mã nguồn hoạt động ổn định, đo đạc cụm thực tế đầy đủ)"),
        ("Giai đoạn P3\n(Tuần 10 – 14)",
         "Xây dựng luồng thực thi song song đường ống (Swarm Pipeline Executor); AutoModelSlicer nạp trọng số chọn lọc; chu trình sinh token tự hồi quy đa nút; cơ chế Micro-batching; hoàn thiện cơ chế chịu lỗi Safe Failure cấp phiên; hỗ trợ mô hình 14B/32B và lượng tử hóa 4-bit.",
         "Hệ thống pipeline phân tán hoàn chỉnh; chu trình sinh văn bản tự hồi quy; cơ chế chịu lỗi tự động; kiểm thử tích hợp đa nút; visualizer luồng token.",
         "KẾ HOẠCH NỬA SAU (T10 – T14)\n(Đang tiến hành triển khai runtime)"),
        ("Giai đoạn P4\n(Tuần 15 – 18)",
         "Thực hiện chuỗi thực nghiệm quy mô lớn; phân tích thống kê chuyên sâu (P95/P99, ANOVA); đóng gói mã nguồn và tài liệu kỹ thuật hoàn chỉnh; hoàn thiện bản thảo Luận văn Tốt nghiệp và Slide bảo vệ.",
         "Bộ dữ liệu thực nghiệm áp lực 24h; phân tích thống kê hoàn chỉnh; Luận văn Tốt nghiệp BKU chính thức; slide báo cáo; bảo vệ hội đồng.",
         "KẾ HOẠCH NỬA SAU (T15 – T18)\n(Dự kiến hoàn thành đúng hạn bảo vệ)")
    ]
    format_table(tbl_matrix_vn, [Cm(2.6), Cm(5.2), Cm(5.0), Cm(3.2)], matrix_headers_vn, matrix_data_vn)
    add_p(doc, "Bảng 3.1: Ma trận tổng hợp tiến độ và các mốc sản phẩm bàn giao của Đồ án Tốt nghiệp.")
    
    # Khối chữ ký phê duyệt
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(14)
    
    tbl_sign_vn = doc.add_table(rows=2, cols=2)
    tbl_sign_vn.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_sign_vn, val="none")
    
    for row in tbl_sign_vn.rows:
        row.cells[0].width = Cm(8.0)
        row.cells[1].width = Cm(8.0)
        
    p_adv_vn = tbl_sign_vn.rows[0].cells[0].paragraphs[0]
    p_adv_vn.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_adv_t_vn = p_adv_vn.add_run("CÁN BỘ HƯỚNG DẪN\n(Ký và ghi rõ họ tên)\n\n\n\n\n")
    r_adv_t_vn.font.name = "Times New Roman"; r_adv_t_vn.font.size = Pt(11); r_adv_t_vn.font.bold = True
    r_adv_name_vn = p_adv_vn.add_run("TS. Quản Thành Thơ")
    r_adv_name_vn.font.name = "Times New Roman"; r_adv_name_vn.font.size = Pt(11); r_adv_name_vn.font.bold = True
    
    p_stu_vn = tbl_sign_vn.rows[0].cells[1].paragraphs[0]
    p_stu_vn.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_stu_t_vn = p_stu_vn.add_run("SINH VIÊN THỰC HIỆN ĐỀ TÀI\n(Ký và ghi rõ họ tên)\n\n\n\n\n")
    r_stu_t_vn.font.name = "Times New Roman"; r_stu_t_vn.font.size = Pt(11); r_stu_t_vn.font.bold = True
    r_stu_names_vn = p_stu_vn.add_run("Trần Thái Tài          Lưu Hải Đăng")
    r_stu_names_vn.font.name = "Times New Roman"; r_stu_names_vn.font.size = Pt(11); r_stu_names_vn.font.bold = True
    
    p_note_vn = tbl_sign_vn.rows[1].cells[0].paragraphs[0]
    p_note_vn.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_n_vn = p_note_vn.add_run("Báo cáo được lập và trình nộp cho Hội đồng Đánh giá Giữa kỳ, Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách Khoa – ĐHQG-HCM.")
    r_n_vn.font.name = "Times New Roman"; r_n_vn.font.size = Pt(9.5); r_n_vn.font.italic = True; r_n_vn.font.color.rgb = COLOR_MUTED
    
    print(f"Saving Vietnamese Word document to: {output_path} ...")
    doc.save(output_path)
    print(f"-> Successfully created Vietnamese docx: {output_path} ({os.path.getsize(output_path):,} bytes)")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    output_en = os.path.join(project_root, "MIDTERM_PROGRESS_REPORT_WEEKLY_EN.docx")
    output_vn = os.path.join(project_root, "MIDTERM_PROGRESS_REPORT_WEEKLY_VN.docx")
    
    print("=" * 70)
    print("Starting conversion from MIDTERM_PROGRESS_REPORT_WEEKLY_EN.md to DOCX...")
    print("=" * 70)
    
    generate_english_document(output_en)
    generate_vietnamese_document(output_vn)
    
    print("\n" + "=" * 70)
    print("ALL DOCUMENTS GENERATED SUCCESSFULLY!")
    print(f"1. English version:    {output_en}")
    print(f"2. Vietnamese version: {output_vn}")
    print("=" * 70)

if __name__ == "__main__":
    main()
