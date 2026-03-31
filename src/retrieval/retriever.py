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
        used_filter = chunk_type_filter
        if chunk_type_filter and not results:
            # If no chunks exist for the selected type, retry without filter.
            results = self.store.query(
                self.embedder.embed_query(question),
                top_k=top_k, collection=col,
                chunk_type_filter=None)
            used_filter = None
        llm_error = None
        try:
            llm_r = self.llm.generate(
                question=question,
                context=self.llm.format_context(results))
            answer = llm_r.answer
            model = llm_r.model
            llm_input_tokens = llm_r.input_tokens
            llm_output_tokens = llm_r.output_tokens
            llm_latency_ms = llm_r.latency_ms
        except Exception as e:
            llm_error = str(e)
            answer = self._build_retrieval_only_answer(
                results=results,
                err=llm_error,
                requested_filter=chunk_type_filter,
                used_filter=used_filter)
            model = "retrieval-only-fallback"
            llm_input_tokens = 0
            llm_output_tokens = 0
            llm_latency_ms = 0.0
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
            question=question, answer=answer,
            sources=sources,
            retrieval_stats={
                "total_retrieved": len(results),
                "by_type": tc, "collection": col,
                "top_k_requested": top_k,
                "llm_input_tokens": llm_input_tokens,
                "llm_output_tokens": llm_output_tokens,
                "llm_latency_ms": llm_latency_ms,
                "fallback_reason": llm_error},
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            model=model)

    def _build_retrieval_only_answer(
            self,
            results: list[dict],
            err: str,
            requested_filter: Optional[str] = None,
            used_filter: Optional[str] = None) -> str:
        if not results:
            return (
                "The LLM is currently unavailable and no relevant context "
                "was retrieved to answer the question. "
                f"Provider error: {err}")
        lines = [
            "LLM generation is temporarily unavailable. "
            "Showing top retrieved evidence instead:"]
        if requested_filter and used_filter != requested_filter:
            lines.append(
                f"Requested chunk_type_filter='{requested_filter}' returned no "
                "matches, so retrieval was retried without a chunk type filter.")
        for i, r in enumerate(results[:5], 1):
            excerpt = r["text"].replace("\n", " ").strip()
            lines.append(
                f"{i}. {r['source']} (page {r['page']}, "
                f"{r['chunk_type']}, score {r['score']:.2f}): "
                f"{excerpt[:260]}")
        lines.append("Use this evidence to proceed until LLM credits are restored.")
        lines.append(f"Provider error: {err}")
        return "\n".join(lines)


_ret = None


def get_retriever() -> Retriever:
    global _ret
    if _ret is None:
        _ret = Retriever()
    return _ret