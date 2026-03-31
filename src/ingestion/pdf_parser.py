"""src/ingestion/pdf_parser.py -- Multimodal PDF parser.
Extracts text (PyMuPDF), tables (pdfplumber), and images (PyMuPDF + VLM).
"""
from __future__ import annotations
import io, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import fitz
import pdfplumber
from PIL import Image
from src.models.vision import get_vision_model

MIN_IMAGE_AREA = 80 * 80


@dataclass
class ParsedChunk:
    chunk_id: str
    source: str
    page: int
    chunk_type: str  # "text" | "table" | "image"
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ParseResult:
    filename: str
    total_pages: int
    text_chunks: list[ParsedChunk]
    table_chunks: list[ParsedChunk]
    image_chunks: list[ParsedChunk]
    errors: list[str] = field(default_factory=list)

    @property
    def all_chunks(self):
        return self.text_chunks + self.table_chunks + self.image_chunks

    @property
    def counts(self):
        return {"text": len(self.text_chunks), "table": len(self.table_chunks),
                "image": len(self.image_chunks),
                "total": len(self.all_chunks)}


class PDFParser:
    def __init__(self, use_vision: bool = True):
        self._use_vision = use_vision
        self._vision = get_vision_model() if use_vision else None

    def parse(self, pdf_path: Path) -> ParseResult:
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        tc, tbc, ic, errors = [], [], [], []
        with fitz.open(str(pdf_path)) as fd, \
             pdfplumber.open(str(pdf_path)) as pd:
            total = len(fd)
            for i in range(total):
                pn = i + 1
                try:
                    tc.extend(self._text(fd[i], filename, pn, len(tc)))
                except Exception as e:
                    errors.append(f"Page {pn} text: {e}")
                try:
                    tbc.extend(self._tables(pd.pages[i], filename, pn, len(tbc)))
                except Exception as e:
                    errors.append(f"Page {pn} tables: {e}")
                try:
                    ic.extend(self._images(fd, fd[i], filename, pn, len(ic)))
                except Exception as e:
                    errors.append(f"Page {pn} images: {e}")
        return ParseResult(filename=filename, total_pages=total,
                           text_chunks=tc, table_chunks=tbc,
                           image_chunks=ic, errors=errors)

    def _text(self, page, src, pn, off):
        raw = page.get_text("text")
        cleaned = re.sub(r"\s+", " ",
            re.sub(r"[^\x09\x0A\x0D\x20-\x7E\xA0-\uFFFF]", " ", raw)).strip()
        if len(cleaned) < 30:
            return []
        return [ParsedChunk(
            chunk_id=f"text_{src}_{pn:04d}_{off:04d}",
            source=src, page=pn, chunk_type="text", text=cleaned,
            metadata={"source": src, "page": pn, "chunk_type": "text"})]

    def _tables(self, page, src, pn, off):
        chunks = []
        for ti, raw in enumerate(page.extract_tables()):
            if not raw or len(raw) < 2 or not raw[0] or len(raw[0]) < 2:
                continue
            if sum(1 for r in raw for c in r
                   if c and str(c).strip()) < 4:
                continue
            nc = max(len(r) for r in raw)
            cl = [[re.sub(r"\s+", " ", str(c)).strip()
                   if c else "" for c in r] for r in raw]
            for r in cl:
                while len(r) < nc:
                    r.append("")
            lines = [
                "| " + " | ".join(cl[0]) + " |",
                "| " + " | ".join(["---"] * nc) + " |"]
            for r in cl[1:]:
                lines.append("| " + " | ".join(r) + " |")
            chunks.append(ParsedChunk(
                chunk_id=f"table_{src}_{pn:04d}_{off+ti:04d}",
                source=src, page=pn, chunk_type="table",
                text="\n".join(lines),
                metadata={"source": src, "page": pn,
                          "chunk_type": "table", "rows": len(raw), "cols": nc}))
        return chunks

    def _images(self, doc, page, src, pn, off):
        chunks, idx = [], 0
        for xi in page.get_images(full=True):
            try:
                base = doc.extract_image(xi[0])
                img = Image.open(io.BytesIO(base["image"]))
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")
            except Exception:
                continue
            w, h = img.size
            if w * h < MIN_IMAGE_AREA:
                continue
            hint = f"Page {pn} of {src}. Dimensions: {w}x{h}px."
            if self._use_vision and self._vision:
                summary = self._vision.summarise(img, context=hint)
            else:
                summary = f"[Image on page {pn} of {src}, {w}x{h}px]"
            chunks.append(ParsedChunk(
                chunk_id=f"image_{src}_{pn:04d}_{off+idx:04d}",
                source=src, page=pn, chunk_type="image",
                text=summary,
                metadata={"source": src, "page": pn,
                          "chunk_type": "image", "width": w, "height": h}))
            idx += 1
        return chunks