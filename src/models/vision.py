"""src/models/vision.py
Claude Vision VLM: converts PDF images to searchable text summaries.
Images MUST go through a VLM before embedding -- never embed raw pixels.
"""
from __future__ import annotations
import base64, io, threading
from typing import Optional
import anthropic
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import get_settings

IMAGE_SUMMARY_PROMPT = """You are analysing a page from a diesel engine ECU
diagnostic manual. Describe this image in detail:
1. Type of diagram (wiring, circuit, table, etc.)
2. ECU pin numbers, sensor names, connector labels
3. Connections and relationships shown
4. All labels and annotations visible
5. Diagnostic significance
Be precise and technical."""


class VisionModel:
    def __init__(self):
        cfg = get_settings()
        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.vision_model

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=15), reraise=True)
    def summarise(self, image: Image.Image, context: str = "") -> str:
        b64 = self._to_b64(image)
        prompt = f"Context: {context}\n\n{IMAGE_SUMMARY_PROMPT}" if context else IMAGE_SUMMARY_PROMPT
        response = self._client.messages.create(
            model=self._model, max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt}]}])
        return response.content[0].text.strip()

    def summarise_batch(self, images, contexts=None):
        contexts = contexts or [""] * len(images)
        results = []
        for img, ctx in zip(images, contexts):
            try: results.append(self.summarise(img, ctx))
            except Exception as e: results.append(f"[VLM error: {e}]")
        return results

    @staticmethod
    def _to_b64(img: Image.Image) -> str:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > 1568:
            s = 1568 / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()


_vm: Optional[VisionModel] = None

def get_vision_model() -> VisionModel:
    global _vm
    if _vm is None:
        _vm = VisionModel()
    return _vm