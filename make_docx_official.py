import re
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_inline_runs(paragraph, text, default_font="Times New Roman", default_size=12, default_color=None):
    pattern = re.compile(r'(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|([^*`]+))')
    for match in pattern.finditer(text):
        full = match.group(0)
        bold = False
        italic = False
        code = False
        run_text = full

        if full.startswith('**') and full.endswith('**'):
            bold = True
            run_text = full[2:-2]
        elif full.startswith('*') and full.endswith('*'):
            italic = True
            run_text = full[1:-1]
        elif full.startswith('`') and full.endswith('`'):
            code = True
            run_text = full[1:-1]

        run = paragraph.add_run(run_text)
        run.bold = bold
        run.italic = italic
        run.font.name = "Consolas" if code else default_font
        run.font.size = Pt(10 if code else default_size)
        if default_color:
            run.font.color.rgb = default_color
        elif code:
            run.font.color.rgb = RGBColor(30, 41, 59)

def convert_md_to_docx(md_path: Path, docx_path: Path):
    doc = Document()

    # Page setup: Standard 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)

    # Base Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(6)

    lines = md_path.read_text(encoding="utf-8").split("\n")
    i = 0
    in_table = False
    table_rows = []

    while i < len(lines):
        line = lines[i].rstrip("\r\n")

        # Table handling
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip()[1:-1].split("|")]
            # Check separator
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                i += 1
                continue
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(cells)
            i += 1
            continue
        else:
            if in_table:
                # Flush table
                if table_rows:
                    num_cols = max(len(r) for r in table_rows)
                    tbl = doc.add_table(rows=len(table_rows), cols=num_cols)
                    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                    tbl.autofit = False

                    for r_idx, row in enumerate(table_rows):
                        for c_idx, cell_value in enumerate(row):
                            if c_idx < num_cols:
                                cell = tbl.cell(r_idx, c_idx)
                                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                                p = cell.paragraphs[0]
                                p.paragraph_format.space_after = Pt(2)
                                p.paragraph_format.line_spacing = 1.15
                                if r_idx == 0:
                                    set_cell_background(cell, "E2E8F0")
                                    add_inline_runs(p, cell_value, default_font="Arial", default_size=10, default_color=RGBColor(15, 23, 42))
                                    for run in p.runs:
                                        run.bold = True
                                else:
                                    if r_idx % 2 == 1:
                                        set_cell_background(cell, "F8FAFC")
                                    add_inline_runs(p, cell_value, default_font="Times New Roman", default_size=10)
                                set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
                    doc.add_paragraph() # Spacing
                in_table = False
                table_rows = []

        if not line.strip():
            i += 1
            continue

        if line.startswith("# "):
            title_text = line[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(12)
            run = p.add_run(title_text)
            run.font.name = 'Arial'
            run.font.size = Pt(18)
            run.bold = True
            run.font.color.rgb = RGBColor(15, 23, 42)

        elif line.startswith("## "):
            h_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = 'Arial'
            run.font.size = Pt(14)
            run.bold = True
            run.font.color.rgb = RGBColor(30, 58, 138)

        elif line.startswith("### "):
            h_text = line[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = 'Arial'
            run.font.size = Pt(12)
            run.bold = True
            run.font.color.rgb = RGBColor(30, 64, 175)

        elif line.startswith("#### "):
            h_text = line[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = 'Arial'
            run.font.size = Pt(11)
            run.bold = True
            run.font.color.rgb = RGBColor(55, 65, 81)

        elif line.startswith("> "):
            quote_text = line[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            p.paragraph_format.right_indent = Inches(0.2)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            add_inline_runs(p, quote_text, default_font="Times New Roman", default_size=11, default_color=RGBColor(55, 65, 81))
            for run in p.runs:
                run.italic = True

        elif line.startswith("* ") or line.startswith("- "):
            item_text = line[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.25
            add_inline_runs(p, item_text)

        elif re.match(r"^\d+\.\s+", line):
            num_match = re.match(r"^\d+\.\s+", line)
            content = line[num_match.end():].strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.25
            add_inline_runs(p, content)

        elif line.startswith("---"):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            # Add subtle horizontal divider
            pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="CBD5E1"/></w:pBdr>')
            p._p.get_or_add_pPr().append(pBdr)

        else:
            p = doc.add_paragraph()
            add_inline_runs(p, line)

        i += 1

    if in_table and table_rows:
        num_cols = max(len(r) for r in table_rows)
        tbl = doc.add_table(rows=len(table_rows), cols=num_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for r_idx, row in enumerate(table_rows):
            for c_idx, cell_value in enumerate(row):
                if c_idx < num_cols:
                    cell = tbl.cell(r_idx, c_idx)
                    p = cell.paragraphs[0]
                    add_inline_runs(p, cell_value, default_font="Times New Roman", default_size=10)

    doc.save(docx_path)
    print(f"Verified DOCX created: {docx_path}")

if __name__ == "__main__":
    src = Path("PATENT_DISCLOSURE_REPORT.md")
    dst = Path("AccessAI_Patent_Disclosure_Report.docx")
    convert_md_to_docx(src, dst)
