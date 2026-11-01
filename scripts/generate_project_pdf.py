"""
scripts/generate_project_pdf.py
=============================================================================
Master PDF Generator for MRI Brain Tumor Detection Student Project Documentation
Outputs:
1. MRI_Brain_Tumor_Detection_Complete_Project_Documentation.pdf (Workspace Root)
2. docs/project_report/MRI_Brain_Tumor_Detection_Complete_Project_Documentation.pdf
=============================================================================
"""

import sys
import shutil
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

from scripts.pdf_chapters_1_to_6 import (
    add_cover_and_toc, add_chapter_1, add_chapter_2, add_chapter_3,
    add_chapter_4, add_chapter_5, add_chapter_6
)
from scripts.pdf_chapters_7_to_12 import (
    add_chapter_7, add_chapter_8, add_chapter_9,
    add_chapter_10, add_chapter_11, add_chapter_12
)
from scripts.pdf_chapters_13_to_18 import (
    add_chapter_13, add_chapter_14, add_chapter_15,
    add_chapter_16, add_chapter_17, add_chapter_18
)
from scripts.pdf_chapters_19_to_22 import (
    add_chapter_19, add_chapter_20, add_chapter_21, add_chapter_22_appendix
)


# ---------------------------------------------------------------------------
# Numbered Canvas for Running Headers and Footers ("Page X of Y")
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        # Do not draw headers or footers on the cover page (Page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Header
        self.drawString(40, 755, "MRI Brain Tumor Detection — Complete Student Project Technical Documentation")
        self.drawRightString(572, 755, "Custom CNN & Grad-CAM XAI")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(40, 748, 572, 748)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(40, 42, 572, 42)
        self.drawString(40, 31, "Student Project Technical Report · Python 3.11 · TensorFlow 2.15 · Streamlit")
        self.drawRightString(572, 31, f"Page {self._pageNumber} of {page_count}")

        self.restoreState()


# ---------------------------------------------------------------------------
# Style Factory
# ---------------------------------------------------------------------------
def setup_custom_styles():
    styles = getSampleStyleSheet()

    PRIMARY_COLOR = colors.HexColor("#1A2B4C")    # Deep Navy
    SECONDARY_COLOR = colors.HexColor("#2563EB")  # Royal Blue
    TEXT_COLOR = colors.HexColor("#1E293B")       # Dark Slate

    styles.add(ParagraphStyle(
        name="DocTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PRIMARY_COLOR,
        alignment=1, # Center
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="DocSubtitle",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=SECONDARY_COLOR,
        alignment=1,
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name="MetaBlock",
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=10,
    ))

    styles.add(ParagraphStyle(
        name="ChapterHeading",
        fontName="Helvetica-Bold",
        fontSize=13.0,
        leading=16.5,
        textColor=PRIMARY_COLOR,
        spaceBefore=14,
        spaceAfter=5,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=13.5,
        textColor=SECONDARY_COLOR,
        spaceBefore=9,
        spaceAfter=3,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="SubsectionHeading",
        fontName="Helvetica-Bold",
        fontSize=9.0,
        leading=12.0,
        textColor=colors.HexColor("#334155"),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="ReportBody",
        fontName="Helvetica",
        fontSize=8.4,
        leading=11.6,
        textColor=TEXT_COLOR,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name="ReportBullet",
        fontName="Helvetica",
        fontSize=8.4,
        leading=11.4,
        textColor=TEXT_COLOR,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=2.5,
    ))

    styles.add(ParagraphStyle(
        name="CalloutText",
        fontName="Helvetica-Oblique",
        fontSize=8.0,
        leading=11.0,
        textColor=colors.HexColor("#1E3A8A"),
    ))

    styles.add(ParagraphStyle(
        name="CodeBlockStyle",
        fontName="Courier",
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor("#0F172A"),
    ))

    styles.add(ParagraphStyle(
        name="FigureCaption",
        fontName="Helvetica-Oblique",
        fontSize=7.4,
        leading=9.8,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceBefore=2.5,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="TableHeaderStyle",
        fontName="Helvetica-Bold",
        fontSize=7.6,
        leading=9.8,
        textColor=colors.white,
        alignment=0,
    ))

    styles.add(ParagraphStyle(
        name="TableCellStyle",
        fontName="Helvetica",
        fontSize=7.4,
        leading=9.5,
        textColor=TEXT_COLOR,
        alignment=0,
    ))

    styles.add(ParagraphStyle(
        name="TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=7.4,
        leading=9.5,
        textColor=PRIMARY_COLOR,
        alignment=0,
    ))

    styles.add(ParagraphStyle(
        name="TOCItem",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.8,
        textColor=PRIMARY_COLOR,
    ))

    styles.add(ParagraphStyle(
        name="TOCDesc",
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#64748B"),
    ))

    return styles


# ---------------------------------------------------------------------------
# UI Element Helpers
# ---------------------------------------------------------------------------
def make_callout(text, styles, title="KEY CONCEPT & MEDICAL RATIONALE"):
    callout_data = [
        [
            Paragraph(f"<b>{title}:</b> {text}", styles["CalloutText"])
        ]
    ]
    t = Table(callout_data, colWidths=[532])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0F9FF")),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LINELEFT', (0,0), (0,0), 3.0, colors.HexColor("#0284C7")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#BAE6FD")),
    ]))
    return t


def make_code_box(code_text, styles):
    code_data = [[Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["CodeBlockStyle"])]]
    t = Table(code_data, colWidths=[532])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    return t


def make_table(headers, rows, col_widths, styles):
    table_data = []
    header_row = [Paragraph(h, styles["TableHeaderStyle"]) for h in headers]
    table_data.append(header_row)

    for row_idx, row in enumerate(rows):
        formatted_row = []
        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                p = Paragraph(str(cell), styles["TableCellBold"])
            else:
                p = Paragraph(str(cell), styles["TableCellStyle"])
            formatted_row.append(p)
        table_data.append(formatted_row)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    t_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A2B4C")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3.0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.0),
        ('LEFTPADDING', (0,0), (-1,-1), 4.5),
        ('RIGHTPADDING', (0,0), (-1,-1), 4.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor("#CBD5E1")),
    ]
    for r_idx in range(1, len(table_data)):
        bg = colors.HexColor("#FFFFFF") if r_idx % 2 != 0 else colors.HexColor("#F8FAFC")
        t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), bg))

    t.setStyle(TableStyle(t_style))
    return t


def make_image_flowable(image_path, width, height, caption, styles):
    p = Path(image_path)
    if not p.exists():
        return Paragraph(f"<i>[Image file not found: {p}]</i>", styles["FigureCaption"])
    img = Image(str(p), width=width, height=height)
    cap = Paragraph(caption, styles["FigureCaption"])
    return KeepTogether([Spacer(1, 2.5), img, cap, Spacer(1, 2.5)])


# ---------------------------------------------------------------------------
# Main PDF Compilation
# ---------------------------------------------------------------------------
def build_master_pdf():
    pdf_filename = "MRI_Brain_Tumor_Detection_Complete_Project_Documentation.pdf"
    output_pdf_path = PROJECT_ROOT / pdf_filename

    print(f"Initializing PDF generation: {output_pdf_path}")
    
    # 54pt top/bottom margins, 40pt left/right margins on letter (612x792 pt)
    # Printable width: 612 - 80 = 532 pt
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=54,
        bottomMargin=54,
    )

    styles = setup_custom_styles()
    story = []

    print("Adding Cover & Table of Contents...")
    add_cover_and_toc(story, styles, make_callout, make_table, make_image_flowable)

    print("Adding Chapters 1 to 6...")
    add_chapter_1(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_2(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_3(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_4(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_5(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_6(story, styles, make_callout, make_table, make_image_flowable)

    print("Adding Chapters 7 to 12...")
    add_chapter_7(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_8(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_9(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_10(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_11(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_12(story, styles, make_callout, make_table, make_image_flowable)

    print("Adding Chapters 13 to 18...")
    add_chapter_13(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_14(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_15(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_16(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_17(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_18(story, styles, make_callout, make_table, make_image_flowable)

    print("Adding Chapters 19 to 22 (Appendix)...")
    add_chapter_19(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_20(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_21(story, styles, make_callout, make_table, make_image_flowable)
    add_chapter_22_appendix(story, styles, make_callout, make_table, make_image_flowable)

    print(f"Building PDF with {len(story)} flowable elements using NumberedCanvas...")
    doc.build(story, canvasmaker=NumberedCanvas)

    print(f"SUCCESS: Master PDF compiled successfully at: {output_pdf_path}")

    # Also place a copy in docs/project_report/
    docs_report_dir = PROJECT_ROOT / "docs" / "project_report"
    docs_report_dir.mkdir(parents=True, exist_ok=True)
    report_copy_path = docs_report_dir / pdf_filename
    shutil.copy2(output_pdf_path, report_copy_path)
    print(f"Saved duplicate copy to: {report_copy_path}")

    # Print file statistics
    size_mb = output_pdf_path.stat().st_size / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    build_master_pdf()
