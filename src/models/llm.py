"""src/models/llm.py -- Claude LLM with custom RAG prompt template."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import get_settings

SYSTEM_PROMPT = """You are an expert automotive diagnostic engineer
specialising in common-rail diesel engine ECU fault diagnosis.
Rules:
1. Base your answer ONLY on the provided context.
2. State the fault code, lamp status, and recovery mode when relevant.
3. Present diagnostic checks as a numbered list.
4. If the answer is not in the context, say so clearly.
5. Format fault codes in backticks, e.g. `P0087`."""

RAG_TEMPLATE = (
    "## Retrieved Context\n{context}\n\n---\n\n"
    "## Question\n{question}\n\n## Answer:")


@dataclass
class LLMResponse:
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class LLMClient:
    def __init__(self):
        cfg = get_settings()
        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.llm_model
        self._max_tokens = cfg.max_tokens_response
        self._temperature = cfg.temperature

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def generate(self, question: str, context: str) -> LLMResponse:
        t0 = time.perf_counter()
        r = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": RAG_TEMPLATE.format(
                           context=context, question=question)}])
        return LLMResponse(
            answer=r.content[0].text.strip(),
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            model=r.model)

    def format_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant context retrieved."
        parts = []
        for ct, label in [("text", "Text Excerpts"),
                           ("table", "Table Data"),
                           ("image", "Diagram Descriptions (VLM)")]:
            group = [c for c in chunks if c["chunk_type"] == ct]
            if group:
                parts.append(f"### {label}")
                for i, c in enumerate(group, 1):
                    parts.append(
                        f"[{i} | {c['source']} | Page {c['page']} | "
                        f"Score {c['score']:.2f}]\n{c['text']}")
        return "\n\n".join(parts)


_llm: Optional[LLMClient] = None

def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm