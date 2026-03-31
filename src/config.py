"""src/config.py -- All settings via pydantic-settings."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore")
    anthropic_api_key: str = Field(...)
    vision_model: str = Field("claude-opus-4-5")
    llm_model: str = Field("claude-sonnet-4-6")
    max_tokens_response: int = Field(2048)
    temperature: float = Field(0.1)
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(384)
    chroma_persist_dir: Path = Field(Path("./data/chroma"))
    default_collection: str = Field("diagnostic_rag")
    chunk_size: int = Field(800)
    chunk_overlap: int = Field(100)
    default_top_k: int = Field(6)
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    log_level: Literal["DEBUG","INFO","WARNING","ERROR"] = Field("INFO")

    @field_validator("chroma_persist_dir", mode="before")
    @classmethod
    def make_dir(cls, v):
        p = Path(v); p.mkdir(parents=True, exist_ok=True); return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()