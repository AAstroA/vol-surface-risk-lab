from __future__ import annotations

from pathlib import Path

import fitz


SOURCE = Path("reports/latex/SPX_WRDS_Thesis_Report_2005_2021.tex")
PDF = Path("reports/SPX_WRDS_Thesis_Report_2005_2021.pdf")

SOURCE_REPLACEMENTS = {
    "pdfauthor={Volatility Surface Risk Lab}": "pdfauthor={Alireza Moslemi Haghighi}",
    "The requested endpoint is 31 December 2021": "The analysis endpoint is 31 December 2021",
    "The initial caption was therefore too confident.": "That interpretation would be too strong.",
    "Thus the user's expected geometry does emerge after scaling": "Thus the canonical geometry emerges after scaling",
    "The new full-revaluation experiment validates the user's intuition:": "The full-revaluation experiment tests the core modeling implication:",
}

PDF_REPLACEMENTS = {
    "requested endpoint": "analysis endpoint",
    "The initial caption was therefore too confident.": "That interpretation was too strong.",
    "the user's expected geometry": "the canonical geometry",
    "validates the user's intuition": "tests the core implication",
}


def update_source() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for old, new in SOURCE_REPLACEMENTS.items():
        if old not in text:
            raise RuntimeError(f"Expected source phrase not found: {old!r}")
        text = text.replace(old, new)
    SOURCE.write_text(text, encoding="utf-8")


def update_pdf() -> None:
    document = fitz.open(PDF)
    pending: list[tuple[fitz.Page, fitz.Rect, str, float]] = []
    found = {phrase: 0 for phrase in PDF_REPLACEMENTS}

    for page in document:
        for old, new in PDF_REPLACEMENTS.items():
            for rect in page.search_for(old):
                found[old] += 1
                page.add_redact_annot(rect, fill=(1, 1, 1))
                font_size = max(6.0, min(10.0, rect.height * 0.72))
                pending.append((page, rect, new, font_size))

    missing = [phrase for phrase, count in found.items() if count == 0]
    if missing:
        raise RuntimeError(f"Expected PDF phrases not found: {missing}")

    for page in document:
        page.apply_redactions()

    for page, rect, replacement, font_size in pending:
        target = fitz.Rect(rect.x0, rect.y0 - 0.5, rect.x1, rect.y1 + 1.0)
        result = page.insert_textbox(
            target,
            replacement,
            fontname="Times-Roman",
            fontsize=font_size,
            color=(0.15, 0.18, 0.22),
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
        )
        if result < -1:
            raise RuntimeError(f"Replacement did not fit: {replacement!r}")

    metadata = document.metadata or {}
    metadata["author"] = "Alireza Moslemi Haghighi"
    metadata["title"] = "MSc Thesis - SPX Volatility-Surface Risk on WRDS, 2005-2021"
    document.set_metadata(metadata)

    temporary = PDF.with_suffix(".cleaned.pdf")
    document.save(temporary, garbage=4, deflate=True, clean=True)
    document.close()
    temporary.replace(PDF)

    check = fitz.open(PDF)
    text = "\n".join(page.get_text() for page in check)
    check.close()
    for old in PDF_REPLACEMENTS:
        if old in text:
            raise RuntimeError(f"Old phrase remains in PDF: {old!r}")


def main() -> None:
    update_source()
    update_pdf()
    print("Report language and metadata updated.")


if __name__ == "__main__":
    main()
