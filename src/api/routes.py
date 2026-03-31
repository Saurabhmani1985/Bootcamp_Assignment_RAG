"""src/api/routes.py -- All FastAPI route handlers."""
from __future__ import annotations
import tempfile, time
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from src.api.schemas import (
    ChunkCounts, ComponentStatus, DeleteRequest, DeleteResponse,
    DocumentInfo, DocumentsResponse, HealthResponse, IngestResponse,
    QueryRequest, QueryResponse, RetrievalStats, SourceReference)

router = APIRouter()
_START = time.time()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    from src.config import get_settings
    from src.retrieval.vector_store import get_vector_store
    cfg = get_settings()
    comps = {}
    try:
        store = get_vector_store()
        stats = store.get_stats()
        total_chunks = stats["total_chunks"]
        indexed_docs = len(store.list_indexed_documents())
        comps["vector_store"] = ComponentStatus(
            status="ok",
            detail=f"ChromaDB -- {total_chunks} chunks in '{cfg.default_collection}'")
    except Exception as e:
        total_chunks = indexed_docs = 0
        comps["vector_store"] = ComponentStatus(
            status="unavailable", detail=str(e))
    try:
        from src.ingestion.embedder import get_embedder
        emb = get_embedder()
        comps["embedding_model"] = ComponentStatus(
            status="ok",
            detail=f"{cfg.embedding_model} (dim={emb.dim})")
    except Exception as e:
        comps["embedding_model"] = ComponentStatus(
            status="unavailable", detail=str(e))
    ok = bool(cfg.anthropic_api_key
              and cfg.anthropic_api_key.startswith("sk-"))
    comps["llm"] = ComponentStatus(
        status="ok" if ok else "unavailable",
        detail=cfg.llm_model if ok else "API key not set")
    comps["vision_model"] = ComponentStatus(
        status="ok" if ok else "unavailable",
        detail=cfg.vision_model if ok else "API key not set")
    try:
        sz = sum(f.stat().st_size
                 for f in cfg.chroma_persist_dir.rglob("*")
                 if f.is_file())
        index_size_mb = round(sz / 1048576, 2)
    except Exception:
        index_size_mb = 0.0
    overall = ("ok" if all(c.status == "ok"
                           for c in comps.values()) else "degraded")
    return HealthResponse(
        status=overall, version="1.0.0",
        uptime_seconds=round(time.time() - _START, 1),
        indexed_documents=indexed_docs,
        total_chunks=total_chunks,
        index_size_mb=index_size_mb,
        components=comps)


@router.post("/ingest", response_model=IngestResponse,
             status_code=status.HTTP_201_CREATED, tags=["Ingestion"])
async def ingest_document(
        file: UploadFile = File(...),
        collection: str = Form("diagnostic_rag"),
        use_vision: bool = Form(True)):
    if not file.filename or \
            not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file.")
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 100 MB limit.")
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf") as f:
            f.write(content)
            tmp = Path(f.name)
        from src.ingestion.pipeline import IngestionPipeline
        r = await IngestionPipeline(
            collection=collection,
            use_vision=use_vision).run_async(tmp)
        st = ("success" if not r.errors
              else ("partial"
                    if r.chunk_counts["total"] > 0 else "error"))
        return IngestResponse(
            status=st, filename=file.filename,
            collection=r.collection,
            total_pages=r.total_pages,
            chunk_counts=ChunkCounts(**r.chunk_counts),
            processing_time_s=r.processing_time_s,
            warnings=r.errors[:10],
            message=(
                f"Indexed {r.chunk_counts['total']} chunks "
                f"from {r.total_pages} pages."))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")
    finally:
        if tmp and tmp.exists():
            tmp.unlink(missing_ok=True)


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(req: QueryRequest):
    from src.retrieval.retriever import get_retriever
    from src.retrieval.vector_store import get_vector_store
    if get_vector_store().collection_count(req.collection) == 0:
        raise HTTPException(
            404,
            f"Collection '{req.collection}' is empty. "
            "Please ingest a document first.")
    try:
        r = get_retriever().query(
            question=req.question, top_k=req.top_k,
            collection=req.collection,
            chunk_type_filter=req.chunk_type_filter)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")
    rs = r.retrieval_stats
    return QueryResponse(
        question=r.question, answer=r.answer,
        sources=[SourceReference(
            filename=s.filename, page=s.page,
            chunk_type=s.chunk_type,
            score=s.score, excerpt=s.excerpt)
                 for s in r.sources],
        retrieval_stats=RetrievalStats(
            total_retrieved=rs["total_retrieved"],
            by_type=rs["by_type"],
            collection=rs["collection"],
            top_k_requested=rs["top_k_requested"],
            llm_input_tokens=rs["llm_input_tokens"],
            llm_output_tokens=rs["llm_output_tokens"],
            llm_latency_ms=rs["llm_latency_ms"]),
        latency_ms=r.latency_ms,
        model=r.model)


@router.get("/documents",
           response_model=DocumentsResponse, tags=["Management"])
def list_documents(collection: str = "diagnostic_rag"):
    from src.retrieval.vector_store import get_vector_store
    docs = get_vector_store().list_indexed_documents(collection)
    return DocumentsResponse(
        collection=collection, document_count=len(docs),
        documents=[DocumentInfo(
            filename=d["filename"],
            chunks=d["chunks"],
            total_chunks=d["total_chunks"]) for d in docs])


@router.delete("/delete",
               response_model=DeleteResponse, tags=["Management"])
def delete_document(req: DeleteRequest):
    from src.retrieval.vector_store import get_vector_store
    n = get_vector_store().delete_document(
        source=req.filename, collection=req.collection)
    if n == 0:
        return DeleteResponse(
            status="not_found", filename=req.filename,
            chunks_removed=0,
            message=f"No chunks found for '{req.filename}'")
    return DeleteResponse(
        status="deleted", filename=req.filename,
        chunks_removed=n,
        message=f"Removed {n} chunks for '{req.filename}'")