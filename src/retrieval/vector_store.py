"""src/retrieval/vector_store.py -- ChromaDB persistent vector store."""
from __future__ import annotations
import threading
from typing import Optional
import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from src.config import get_settings
from src.ingestion.pdf_parser import ParsedChunk

_lock = threading.Lock()
_client = None


def _get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = chromadb.PersistentClient(
                    path=str(get_settings().chroma_persist_dir),
                    settings=ChromaSettings(
                        anonymized_telemetry=False))
    return _client


class VectorStore:
    def __init__(self, collection=None):
        self._default = collection or get_settings().default_collection
        self._client = _get_client()

    def _col(self, name):
        return self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks, embeddings, collection=None, batch_size=256):
        col = self._col(collection or self._default)
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [{k: v for k, v in
                  {**c.metadata, "chunk_id": c.chunk_id}.items()
                  if isinstance(v, (str, int, float, bool))}
                 for c in chunks]
        emb = embeddings.tolist()
        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            col.upsert(ids=ids[sl], embeddings=emb[sl],
                       documents=docs[sl], metadatas=metas[sl])

    def query(self, query_embedding, top_k=6, collection=None,
              chunk_type_filter=None, source_filter=None):
        col = self._col(collection or self._default)
        if col.count() == 0:
            return []
        conds = []
        if chunk_type_filter:
            conds.append({"chunk_type": {"$eq": chunk_type_filter}})
        if source_filter:
            conds.append({"source": {"$eq": source_filter}})
        where = (conds[0] if len(conds) == 1
                 else ({"$and": conds} if len(conds) > 1 else None))
        kw = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"])
        if where:
            kw["where"] = where
        r = col.query(**kw)
        return [
            {"chunk_id": cid, "text": doc,
             "chunk_type": meta.get("chunk_type", "text"),
             "source": meta.get("source", ""),
             "page": meta.get("page", 0),
             "score": round(1.0 - dist, 4),
             "metadata": meta}
            for cid, doc, meta, dist in zip(
                r["ids"][0], r["documents"][0],
                r["metadatas"][0], r["distances"][0])]

    def collection_count(self, collection=None):
        return self._col(collection or self._default).count()

    def list_indexed_documents(self, collection=None):
        col = self._col(collection or self._default)
        if col.count() == 0:
            return []
        data = col.get(include=["metadatas"])
        stats = {}
        for m in data["metadatas"]:
            s = m.get("source", "unknown")
            ct = m.get("chunk_type", "text")
            if s not in stats:
                stats[s] = {"text": 0, "table": 0, "image": 0}
            stats[s][ct] = stats[s].get(ct, 0) + 1
        return [{"filename": s, "chunks": v,
                 "total_chunks": sum(v.values())}
                for s, v in stats.items()]

    def list_collections(self):
        return [c.name for c in self._client.list_collections()]

    def delete_document(self, source, collection=None):
        col = self._col(collection or self._default)
        r = col.get(where={"source": {"$eq": source}},
                    include=["metadatas"])
        if r["ids"]:
            col.delete(ids=r["ids"])
        return len(r["ids"])

    def delete_collection(self, name):
        self._client.delete_collection(name)

    def get_stats(self, collection=None):
        cn = collection or self._default
        try:
            count = self._col(cn).count()
        except Exception:
            count = 0
        return {"collection": cn, "total_chunks": count,
                "all_collections": self.list_collections()}


_store = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store