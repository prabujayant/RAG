"""Rendering shims shared by the corpus generator.

These adapt the ``scripts.corpus_shims`` markdown helpers so they can also
emit PDF (reportlab) and DOCX (python-docx) documents without duplicating
content logic in the generator.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
FENCE_RE = re.compile(r"^```")
TABLE_ROW_RE = re.compile(r"^\|.*\|$")
SEP_ROW_RE = re.compile(r"^\|[\s:|-]+\|$")


def _strip_md(text: str) -> str:
    """Strip markdown link/image syntax and inline code for text documents."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text


def _tables_to_html(md: str) -> str:
    """Convert markdown tables into simple HTML tables (reportlab platypus)."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if TABLE_ROW_RE.match(line) and i + 1 < len(lines) and SEP_ROW_RE.match(lines[i + 1]):
            headers = [c.strip() for c in line.strip("|").split("|")]
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            html = ["<table>", "<tr>" + "".join(f"<td><b>{h}</b></td>" for h in headers) + "</tr>"]
            for row in rows:
                html.append("<tr>" + "".join(f"<td>{_strip_md(c)}</td>" for c in row) + "</tr>")
            html.append("</table>")
            out.append("\n".join(html))
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def get_title(md: str) -> str:
    """Extract the first level-1 heading from a markdown document."""
    for line in md.splitlines():
        if HEADING_RE.match(line) and line.startswith("# "):
            return _strip_md(HEADING_RE.match(line).group(2).strip())
    return "Untitled"


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------


def render_pdf(md: str, path: Path, *, output_dir: Path) -> Path:
    """Render a markdown document to PDF using reportlab platypus."""
    from reportlab.platypus import Frame, NextPageTemplate, PageTemplate  # noqa: F401

    out_path = output_dir / (path.stem + ".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = get_title(md)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=20, spaceAfter=10 * mm, textColor=colors.HexColor("#0f2a43"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=15, spaceBefore=7 * mm, spaceAfter=4 * mm, textColor=colors.HexColor("#155e94"))
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=12, spaceBefore=5 * mm, spaceAfter=3 * mm, textColor=colors.HexColor("#1f7a9e"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=3 * mm)
    code = ParagraphStyle("code", parent=styles["Code"], fontSize=9, leading=12, backColor=colors.HexColor("#f4f4f4"))
    note = ParagraphStyle("note", parent=body, textColor=colors.HexColor("#555555"), leftIndent=6 * mm, borderColor=colors.HexColor("#d0d0d0"))

    story: list[Any] = []
    in_code = False
    for line in _tables_to_html(md).splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
            continue
        if in_code:
            story.append(Paragraph(_strip_md(line).replace(" ", "&nbsp;"), code))
            continue
        m = HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            txt = _strip_md(m.group(2).strip())
            style = {2: h2, 3: h3}.get(level, body if level > 3 else h1)
            if level == 1:
                story.append(KeepTogether([Paragraph(txt, h1)]))
            else:
                story.append(KeepTogether([Paragraph(txt, style)]))
        elif line.startswith("> "):
            story.append(Paragraph(_strip_md(line[2:]), note))
        elif line.startswith("|"):
            pass  # handled by _tables_to_html
        elif "<table>" in line:
            # table html block already emitted
            story.append(Spacer(1, 2 * mm))
        elif "<tr>" in line:
            continue
        elif "</table>" in line:
            story.append(Spacer(1, 4 * mm))
        elif line.startswith("```"):
            continue
        elif line.strip():
            story.append(Paragraph(_strip_md(line), body))
        else:
            story.append(Spacer(1, 3 * mm))

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm, title=title)
    doc.build(story)
    return out_path


# ---------------------------------------------------------------------------
# DOCX rendering
# ---------------------------------------------------------------------------


def render_docx(md: str, path: Path, *, output_dir: Path) -> Path:
    """Render a markdown document to DOCX using python-docx."""
    from docx import Document
    from docx.shared import Pt

    out_path = output_dir / (path.stem + ".docx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    document.add_heading(get_title(md), 0)

    in_code = False
    for line in _tables_to_html(md).splitlines():
        import html as html_lib

        line = html_lib.unescape(line)
        if FENCE_RE.match(line):
            in_code = not in_code
            continue
        if in_code:
            p = document.add_paragraph()
            run = p.add_run(_strip_md(line))
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            continue
        m = HEADING_RE.match(line)
        if m:
            document.add_heading(_strip_md(m.group(2).strip()), level=min(len(m.group(1)), 4))
        elif line.startswith("> "):
            p = document.add_paragraph()
            run = p.add_run(_strip_md(line[2:]))
            run.italic = True
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if SEP_ROW_RE.match(line):
                continue
            p = document.add_paragraph()
            run = p.add_run(" | ".join(_strip_md(c) for c in cells))
            run.font.size = Pt(9)
        elif "<table>" in line or "</table>" in line or "<tr>" in line:
            continue
        else:
            sentence = _strip_md(line)
            if "- " in sentence:
                document.add_paragraph(sentence, style="List Bullet")
            elif sentence.strip():
                document.add_paragraph(sentence)

    document.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def render_html(md: str, path: Path, *, output_dir: Path) -> Path:
    """Render a markdown document to a standalone HTML file."""

    out_path = output_dir / (path.stem + ".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = get_title(md)
    body_lines: list[str] = []
    in_code = False
    for line in _tables_to_html(md).splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
            if not in_code:
                body_lines.append("</pre>")
            else:
                body_lines.append("<pre><code>")
            continue
        if in_code:
            body_lines.append(_html_escape(line))
            continue
        m = HEADING_RE.match(line)
        if m:
            level = min(len(m.group(1)), 6)
            txt = _strip_md(m.group(2).strip())
            body_lines.append(f"<h{level}>{txt}</h{level}>")
        elif line.startswith("> "):
            body_lines.append(f"<blockquote>{_strip_md(line[2:])}</blockquote>")
        elif line.startswith("|"):
            pass
        elif "<table>" in line or "</table>" in line or "<tr>" in line:
            body_lines.append(line)
        elif line.strip():
            body_lines.append(f"<p>{_strip_md(line)}</p>")
        else:
            body_lines.append("<br>")

    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n</head>\n<body>\n"
        + "\n".join(body_lines)
        + "\n</body>\n</html>\n"
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _html_escape(text: str) -> str:
    import html

    return html.escape(text, quote=False)