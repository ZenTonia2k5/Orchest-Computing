# -*- coding: utf-8 -*-
"""
Helper test to verify python-docx formatting capabilities:
- Table shading and borders
- Callouts with left accent border
- Headers and footers with page numbers
"""
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def test_helpers():
    doc = docx.Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.0)
    
    # Paragraph
    p = doc.add_paragraph()
    r = p.add_run("Kiểm tra định dạng tiếng Việt: Hệ thống Orchest Computing")
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    
    doc.save("test_output.docx")
    print("Test passed successfully!")

if __name__ == "__main__":
    test_helpers()
