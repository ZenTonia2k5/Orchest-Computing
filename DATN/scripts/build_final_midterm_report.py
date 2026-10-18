# -*- coding: utf-8 -*-
"""
Master Execution Script to generate:
d:\\BKU\\Orchest-Computing\\BAO_CAO_TIEN_DO_GIUA_KY_ORCHEST_COMPUTING.docx
"""
import os
import sys
import io
import docx

# Set UTF-8 encoding for stdout/stderr to avoid Windows cp1252 errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from report_builder_core import setup_page_layout
from report_builder_frontmatter import create_cover_page, create_front_matter
from report_builder_ch1_ch2 import build_chapter_1, build_chapter_2
from report_builder_ch3 import build_chapter_3
from report_builder_ch4 import build_chapter_4
from report_builder_ch5_back import build_chapter_5_and_backmatter

def main():
    print("=== Khởi tạo quá trình tạo Báo cáo Giữa kỳ Orchest Computing ===")
    doc = docx.Document()
    
    print("[1/7] Thiết lập bố cục trang, lề chuẩn BKU (Left 3.0cm, Right 2.0cm, Top/Bottom 2.0cm)...")
    setup_page_layout(doc)
    
    print("[2/7] Tạo Trang bìa chuẩn ĐH Bách Khoa TP.HCM...")
    create_cover_page(doc)
    
    print("[3/7] Tạo Lời cam đoan, Lời cảm ơn, Tóm tắt Abstract, Mục lục, Danh mục từ viết tắt...")
    create_front_matter(doc)
    
    print("[4/7] Tạo Chương 1: Tổng quan và Tính cấp thiết của đề tài...")
    build_chapter_1(doc)
    
    print("[5/7] Tạo Chương 2: Cơ sở lý thuyết và Mô hình toán học...")
    build_chapter_2(doc)
    
    print("[6/7] Tạo Chương 3: Thiết kế kiến trúc và Các giải pháp kỹ thuật hệ thống...")
    build_chapter_3(doc)
    
    print("[7/7] Tạo Chương 4: Kết quả thực hiện đến giữa kỳ (Tuần 8 - 9) và Chương 5, Tài liệu tham khảo, Phụ lục...")
    build_chapter_4(doc)
    build_chapter_5_and_backmatter(doc)
    
    # Target output file path
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    output_docx_path = os.path.join(project_root, "BAO_CAO_TIEN_DO_GIUA_KY_ORCHEST_COMPUTING.docx")
    
    print(f"Đang lưu file Word tại: {output_docx_path} ...")
    doc.save(output_docx_path)
    
    file_size_bytes = os.path.getsize(output_docx_path)
    file_size_kb = file_size_bytes / 1024
    
    print("\n" + "="*70)
    print("HOÀN TẤT XUẤT SẮC BÁO CÁO GIỮA KỲ ĐỒ ÁN TỐT NGHIỆP DẠNG WORD (.DOCX)!")
    print(f"- Đường dẫn tập tin: {output_docx_path}")
    print(f"- Dung lượng: {file_size_kb:.1f} KB ({file_size_bytes:,} bytes)")
    print(f"- Tổng số đoạn văn (paragraphs): {len(doc.paragraphs)}")
    print(f"- Tổng số bảng biểu (tables): {len(doc.tables)}")
    print("="*70)

if __name__ == "__main__":
    main()
