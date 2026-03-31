"""src/ingestion/chunker.py
Splits long text chunks into overlapping windows.
Table and image chunks pass through unchanged.
"""
from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_settings
from src.ingestion.pdf_parser import ParsedChunk


class TextChunker:
    def __init__(self, chunk_size=None, chunk_overlap=None):
        cfg = get_settings()
        self._sp = RecursiveCharacterTextSplitter(
            chunk_size=(chunk_size or cfg.chunk_size) * 4,
            chunk_overlap=(chunk_overlap or cfg.chunk_overlap) * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True)

    def split(self, chunks: list[ParsedChunk]) -> list[ParsedChunk]:
        result = []
        for c in chunks:
            if c.chunk_type != "text":
                result.append(c)
                continue
            subs = self._sp.split_text(c.text)
            if len(subs) <= 1:
                result.append(c)
                continue
            for i, t in enumerate(subs):
                t = t.strip()
                if not t:
                    continue
                m = dict(c.metadata)
                m["sub_index"] = i
                m["parent_chunk_id"] = c.chunk_id
                result.append(ParsedChunk(
                    chunk_id=f"{c.chunk_id}_sub{i:03d}",
                    source=c.source, page=c.page,
                    chunk_type="text", text=t, metadata=m))
        return result