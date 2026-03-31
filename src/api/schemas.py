"""src/api/schemas.py -- All Pydantic v2 request and response schemas."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChunkCounts(BaseModel):
    text: int; table: int; image: int; total: int


class IngestResponse(BaseModel):
    status: Literal["success", "partial", "error"]
    filename: str; collection: str; total_pages: int
    chunk_counts: ChunkCounts; processing_time_s: float
    warnings: list[str] = Field(default_factory=list)
    message: str = "Document successfully ingested."


class ComponentStatus(BaseModel):
    status: Literal["ok", "unavailable"]; detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]; version: str
    uptime_seconds: float; indexed_documents: int
    total_chunks: int; index_size_mb: float
    components: dict[str, ComponentStatus]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    collection: str = Field("diagnostic_rag")
    top_k: int = Field(6, ge=1, le=20)
    chunk_type_filter: Optional[
        Literal["text", "table", "image"]] = Field(None)

    @field_validator("question")
    @classmethod
    def strip_q(cls, v): return v.strip()


class SourceReference(BaseModel):
    filename: str; page: int
    chunk_type: Literal["text", "table", "image"]
    score: float; excerpt: str


class RetrievalStats(BaseModel):
    total_retrieved: int; by_type: dict[str, int]
    collection: str; top_k_requested: int
    llm_input_tokens: int; llm_output_tokens: int
    llm_latency_ms: float


class QueryResponse(BaseModel):
    question: str; answer: str
    sources: list[SourceReference]
    retrieval_stats: RetrievalStats
    latency_ms: float; model: str


class DocumentInfo(BaseModel):
    filename: str; chunks: dict[str, int]; total_chunks: int


class DocumentsResponse(BaseModel):
    collection: str; document_count: int
    documents: list[DocumentInfo]


class DeleteRequest(BaseModel):
    filename: str; collection: str = Field("diagnostic_rag")


class DeleteResponse(BaseModel):
    status: Literal["deleted", "not_found"]
    filename: str; chunks_removed: int; message: str