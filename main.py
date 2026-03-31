"""main.py -- FastAPI application entry point.
"""
from __future__ import annotations
import logging, time
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("multimodal_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("  Bootcamp -- Diagnostic Multimodal RAG -- starting up")
    log.info("=" * 60)
    try:
        from src.ingestion.embedder import get_embedder
        emb = get_embedder()
        log.info(f"  OK Embedding model ready (dim={emb.dim})")
    except Exception as e:
        log.warning(f"  FAIL Embedding: {e}")
    try:
        from src.retrieval.vector_store import get_vector_store
        store = get_vector_store()
        stats = store.get_stats()
        log.info(
            f"  OK ChromaDB ready "
            f"({stats['total_chunks']} chunks in "
            f"'{stats['collection']}' )")
    except Exception as e:
        log.warning(f"  FAIL ChromaDB: {e}")
    log.info("  OK API ready -- open /docs for Swagger UI")
    log.info("=" * 60)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Bootcamp Assignment -- Diagnostic Multimodal RAG",
    description=(
        "Multimodal RAG for diesel ECU diagnostic PDF.\n\n"
        "**Quick start:** POST /ingest PDF, then POST /query"),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(ms)
    log.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} [{ms}ms]")
    return response


@app.exception_handler(Exception)
async def exc_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc),
                 "error_type": type(exc).__name__})


app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "message": "Bootcamp -- Diagnostic Multimodal RAG",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"})


if __name__ == "__main__":
    from src.config import get_settings
    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level=cfg.log_level.lower(),
        access_log=False)