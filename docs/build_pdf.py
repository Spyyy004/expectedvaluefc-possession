"""Render docs/technical_document.md to ExpectedValueFC.pdf with reportlab (simple Markdown subset:
#/##/### headings, paragraphs, '- ' and '1. ' lists, | tables |, ``` code blocks, **bold**, `code`).

    python docs/build_pdf.py [output.pdf]
"""
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "technical_document.md")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "submissions", "ExpectedValueFC.pdf")

GREEN = colors.HexColor("#1f6b45")
LINE = colors.HexColor("#c9d3cc")
SOFT = colors.HexColor("#eef4f0")

base = getSampleStyleSheet()
S = {
    "h1": ParagraphStyle("h1", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24,
                         alignment=TA_LEFT, textColor=GREEN, spaceAfter=2),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16,
                         textColor=GREEN, spaceBefore=9, spaceAfter=4, keepWithNext=1),
    "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=11, leading=14,
                         spaceBefore=6, spaceAfter=3, keepWithNext=1),
    "p": ParagraphStyle("p", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.6,
                        spaceAfter=5),
    "cell": ParagraphStyle("cell", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=10),
    "cellh": ParagraphStyle("cellh", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8,
                            leading=10, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.8, leading=10, backColor=SOFT,
                           borderPadding=5, spaceBefore=3, spaceAfter=7),
}


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def table(rows, width):
    head, body = rows[0], rows[1:]
    n = len(head)
    data = [[Paragraph(inline(c), S["cellh"]) for c in head]] + \
           [[Paragraph(inline(c), S["cell"]) for c in r + [""] * (n - len(r))] for r in body]
    # column widths proportional to the longest text in each column (bounded)
    lengths = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(n)]
    weights = [min(max(l, 6), 60) for l in lengths]
    widths = [width * w / sum(weights) for w in weights]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build():
    lines = open(SRC, encoding="utf-8").read().splitlines()
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title="Expected Value FC - Technical Document", author="Expected Value FC")
    width = A4[0] - 34 * mm
    story, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            story.append(Preformatted("\n".join(lines[i + 1:j]), S["code"]))
            i = j + 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story += [table(rows, width), Spacer(1, 6)]
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), S["h3"])); i += 1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), S["h2"])); i += 1
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), S["h1"])); i += 1
        elif re.match(r"^(- |\d+\. )", line):
            ordered = bool(re.match(r"^\d+\. ", line))
            items = []
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
                items.append(ListItem(Paragraph(inline(re.sub(r"^(- |\d+\. )", "", lines[i])), S["p"]),
                                      leftIndent=12))
                i += 1
            story.append(ListFlowable(items, bulletType="1" if ordered else "bullet", start="1" if ordered else None,
                                      leftIndent=12, bulletFontSize=8))
        elif line.strip():
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|- |\d+\. )", lines[i]):
                para.append(lines[i]); i += 1
            story.append(Paragraph(inline(" ".join(para)), S["p"]))
        else:
            i += 1

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(17 * mm, 9 * mm, "Expected Value FC - More than a Hack 2027, Phase 1: Ball Possession")
        canvas.drawRightString(A4[0] - 17 * mm, 9 * mm, "page {}".format(d.page))
        canvas.restoreState()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
