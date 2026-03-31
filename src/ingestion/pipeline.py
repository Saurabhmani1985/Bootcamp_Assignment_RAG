"""src/ingestion/pipeline.py -- End-to-end ingestion orchestrator."""
from __future__ import annotations
import asyncio, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import get_embedder
from src.ingestion.pdf_parser import PDFParser
from src.retrieval.vector_store import get_vector_store

_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class IngestionResult:
    filename: str
    collection: str
    total_pages: int
    chunk_counts: dict
    processing_time_s: float
    errors: list[str]


class IngestionPipeline:
    def __init__(self, collection=None, use_vision=True):
        from src.config import get_settings
        cfg = get_settings()
        self.collection = collection or cfg.default_collection
        self.parser = PDFParser(use_vision=use_vision)
        self.chunker = TextChunker()
        self.embedder = get_embedder()
        self.store = get_vector_store()

    def run(self, pdf_path: Path) -> IngestionResult:
        t0 = time.perf_counter()
        pr = self.parser.parse(Path(pdf_path))
        chunked = self.chunker.split(pr.all_chunks)
        embeddings = self.embedder.embed_chunks(
            chunked, show_progress=True)
        self.store.upsert(
            chunks=chunked, embeddings=embeddings,
            collection=self.collection)
        counts = {"text": 0, "table": 0, "image": 0}
        for c in chunked:
            counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1
        counts["total"] = sum(counts.values())
        name = pdf_path.name if hasattr(pdf_path, "name") else str(pdf_path)
        return IngestionResult(
            filename=name, collection=self.collection,
            total_pages=pr.total_pages, chunk_counts=counts,
            processing_time_s=round(time.perf_counter() - t0, 1),
            errors=pr.errors)

    async def run_async(self, pdf_path: Path) -> IngestionResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.run, pdf_path)