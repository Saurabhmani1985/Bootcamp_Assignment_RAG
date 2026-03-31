"""src/retrieval/retriever.py -- Full RAG pipeline: embed -> retrieve -> generate."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
from src.config import get_settings
from src.ingestion.embedder import get_embedder
from src.models.llm import get_llm
from src.retrieval.vector_store import get_vector_store


@dataclass
class SourceReference:
    filename: str
    page: int
    chunk_type: str
    score: float
    excerpt: str


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[SourceReference]
    retrieval_stats: dict
    latency_ms: float
    model: str


class Retriever:
    def __init__(self, collection=None):
        cfg = get_settings()
        self.collection = collection or cfg.default_collection
        self.embedder = get_embedder()
        self.store = get_vector_store()
        self.llm = get_llm()
        self.cfg = cfg

    def query(self, question: str, top_k=None,
              collection=None,
              chunk_type_filter=None) -> QueryResult:
        t0 = time.perf_counter()
        top_k = top_k or self.cfg.default_top_k
        col = collection or self.collection
        results = self.store.query(
            self.embedder.embed_query(question),
            top_k=top_k, collection=col,
            chunk_type_filter=chunk_type_filter)
        llm_r = self.llm.generate(
            question=question,
            context=self.llm.format_context(results))
        tc = {}
        for r in results:
            tc[r["chunk_type"]] = tc.get(r["chunk_type"], 0) + 1
        sources = [
            SourceReference(
                filename=r["source"], page=r["page"],
                chunk_type=r["chunk_type"], score=r["score"],
                excerpt=r["text"][:300].replace("\n", " "))
            for r in results]
        return QueryResult(
            question=question, answer=llm_r.answer,
            sources=sources,
            retrieval_stats={
                "total_retrieved": len(results),
                "by_type": tc, "collection": col,
                "top_k_requested": top_k,
                "llm_input_tokens": llm_r.input_tokens,
                "llm_output_tokens": llm_r.output_tokens,
                "llm_latency_ms": llm_r.latency_ms},
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            model=llm_r.model)


_ret = None


def get_retriever() -> Retriever:
    global _ret
    if _ret is None:
        _ret = Retriever()
    return _ret