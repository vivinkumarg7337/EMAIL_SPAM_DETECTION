"""Convert GOOGLE_DOCS_REPORT.md into an editable Google Docs .docx upload."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "GOOGLE_DOCS_REPORT.md"
OUT = ROOT / "reports" / "EMAIL_SPAM_DETECTION_REPORT.docx"


def inline(paragraph, text: str) -> None:
    for part in re.split(r"(\*\*.*?\*\*|`.*?`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        else:
            paragraph.add_run(part)


def main() -> None:
    doc = Document()
    lines = SRC.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            headers = [c.strip() for c in line.strip("|").split("|")]
            table = doc.add_table(rows=1, cols=len(headers))
            table.style = "Table Grid"
            for j, value in enumerate(headers):
                table.rows[0].cells[j].text = value.replace("**", "")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                values = [c.strip() for c in lines[i].strip("|").split("|")]
                if len(values) == len(headers):
                    row = table.add_row()
                    for j, value in enumerate(values):
                        row.cells[j].text = value.replace("**", "")
                i += 1
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:], 0)
        elif line.startswith("## "):
            doc.add_heading(line[3:], 1)
        else:
            paragraph = doc.add_paragraph()
            inline(paragraph, line)
        i += 1
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
