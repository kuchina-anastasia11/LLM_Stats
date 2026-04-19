import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import fitz  
import pdfplumber

@dataclass
class ExtractedDocument:
    text: str
    n_pages: int
    n_tables: int

_NUMBER_IN_CELL = re.compile(r"-?\d+\.?\d*")
_STAT_KEYWORDS = re.compile(
    r"\b(p[-\s]?value|mean|[sS][dD]|[sS][eE]|CI|df|[nN]\b|"
    r"t[-\s]?test|F[-\s]?test|chi|odds|hazard|"
    r"coefficient|estimate|\u03b2|OR|HR|RR|AOR|"
    r"median|range|IQR)\b",
    re.I,
)


def is_data_table(table: list) -> bool:
    if not table or len(table) < 2:
        return False
    if max(len(r) for r in table) < 2:
        return False

    header_text = " ".join((c or "") for c in table[0])
    if _STAT_KEYWORDS.search(header_text):
        return True

    total_cells = 0
    numeric_cells = 0
    for row in table[1:]:
        for cell in row:
            cell_text = (cell or "").strip()
            if not cell_text:
                continue
            total_cells += 1
            if _NUMBER_IN_CELL.search(cell_text):
                numeric_cells += 1

    if total_cells == 0:
        return False
    return numeric_cells / total_cells >= 0.3


def table_to_markdown(table: list) -> str:
    if not is_data_table(table):
        return ""

    cleaned = []
    for row in table:
        cleaned.append([(cell or "").replace("\n", " ").strip() for cell in row])

    n_cols = max(len(r) for r in cleaned)
    for row in cleaned:
        while len(row) < n_cols:
            row.append("")

    lines = []
    lines.append("| " + " | ".join(cleaned[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in cleaned[0]) + " |")
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_tables_pdfplumber(pdf_path):
    tables_by_page = {}
    bboxes_by_page = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                found = page.find_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                })
                if not found:
                    found = page.find_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    })

                md_tables = []
                page_bboxes = []
                for tbl in (found or []):
                    md = table_to_markdown(tbl.extract())
                    if md:
                        md_tables.append(md)
                        page_bboxes.append(tbl.bbox)

                if md_tables:
                    tables_by_page[i] = md_tables
                    bboxes_by_page[i] = page_bboxes
    except Exception:
        pass

    return tables_by_page, bboxes_by_page

def raw_pages(pdf_path, bboxes_by_page=None):
    bboxes_by_page = bboxes_by_page or {}
    doc = fitz.open(pdf_path)
    pages = []
    for page_idx, page in enumerate(doc, start=1):
        table_bboxes = bboxes_by_page.get(page_idx, [])

        if not table_bboxes:
            text = page.get_text("text") or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        else:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            lines = []
            for block in blocks:
                if block["type"] != 0:
                    continue
                by = (block["bbox"][1] + block["bbox"][3]) / 2
                bx = (block["bbox"][0] + block["bbox"][2]) / 2

                in_table = any(
                    x0 - 5 <= bx <= x1 + 5 and y0 - 5 <= by <= y1 + 5
                    for (x0, y0, x1, y1) in table_bboxes
                )
                if in_table:
                    continue

                for line_info in block.get("lines", []):
                    text = "".join(span["text"] for span in line_info["spans"]).strip()
                    if text:
                        lines.append(text)

        pages.append(lines)
    doc.close()
    return pages


def detect_boilerplate(pages):
    if not pages:
        return set()
    n = len(pages)
    counts = Counter()
    for lines in pages:
        for ln in set(lines):
            if len(ln.split()) <= 35:
                counts[ln] += 1
    return {ln for ln, c in counts.items() if c / n >= 0.4}


def is_page_number(line):
    s = line.strip().strip(".")
    return bool(re.fullmatch(r"\d{1,4}", s)) or \
           bool(re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", s, re.I))


def build_text(pages, boilerplate, tables_by_page):
    parts = []
    buf = []
    tables_inserted = set()

    def flush():
        if buf:
            text = " ".join(buf)
            text = re.sub(r"-\s+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                parts.append(text)
            buf.clear()

    def insert_tables(page_num):
        if page_num in tables_inserted:
            return
        tables_inserted.add(page_num)
        for md in tables_by_page.get(page_num, []):
            parts.append(md)

    prev_page = 1
    for page_idx, lines in enumerate(pages, start=1):
        if page_idx != prev_page:
            flush()
            insert_tables(prev_page)
            prev_page = page_idx

        for ln in lines:
            if ln in boilerplate or is_page_number(ln):
                continue
            buf.append(ln)
            if ln.endswith((".", "!", "?")) and len(" ".join(buf)) > 200:
                flush()

    flush()
    insert_tables(prev_page)
    for pg in sorted(tables_by_page.keys()):
        insert_tables(pg)

    return "\n\n".join(parts)

def extract(pdf_path):
    """PDF -> ExtractedDocument (текст + markdown-таблицы, без дубликатов)."""
    pdf_path = Path(pdf_path)
    tables_by_page, bboxes_by_page = extract_tables_pdfplumber(pdf_path)
    pages = raw_pages(pdf_path, bboxes_by_page)
    boilerplate = detect_boilerplate(pages)
    text = build_text(pages, boilerplate, tables_by_page)
    n_tables = sum(len(v) for v in tables_by_page.values())
    return ExtractedDocument(text=text, n_pages=len(pages), n_tables=n_tables)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m pipeline.pdf_extractor <path.pdf> [output.md]")
        raise SystemExit(1)

    doc = extract(sys.argv[1])

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = str(Path(sys.argv[1]).with_suffix(".md"))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc.text)

