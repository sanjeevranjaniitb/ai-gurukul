"""LLM service wrapping the Ollama HTTP API for generation and streaming."""

from __future__ import annotations

import time
from typing import Iterator

import httpx

from backend.app.config import AppConfig
from backend.app.logging_utils import get_logger
from backend.app.models import GenerationResult, RetrievalResult

logger = get_logger("llm_service")

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain enough information, "
    "say so clearly. Do not make up facts."
)


def _build_prompt(question: str, context: list[RetrievalResult]) -> str:
    """Build a full prompt string from context chunks and the user question."""
    parts: list[str] = [_SYSTEM_PROMPT, ""]
    if context:
        parts.append("Context:")
        for i, r in enumerate(context, 1):
            parts.append(f"[{i}] {r.chunk.text}")
        parts.append("")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n".join(parts)


class LLMService:
    """Thin wrapper around the Ollama ``/api/generate`` endpoint."""

    def __init__(self, config: AppConfig | None = None) -> None:
        cfg = config or AppConfig()
        self.model_name: str = cfg.llm_model
        self.base_url: str = cfg.llm_base_url.rstrip("/")
        self.temperature: float = cfg.llm_temperature
        self.max_tokens: int = cfg.llm_max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context: list[RetrievalResult],
    ) -> GenerationResult:
        """Send a non-streaming request to Ollama and return the full result."""
        prompt = _build_prompt(question, context)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        start = time.monotonic()
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            return _error_result(self.model_name, start)
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return _error_result(self.model_name, start)
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama returned HTTP %s", exc.response.status_code)
            return _error_result(self.model_name, start)

        data = resp.json()
        duration_ms = (time.monotonic() - start) * 1000
        return GenerationResult(
            answer=data.get("response", ""),
            model=data.get("model", self.model_name),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            duration_ms=duration_ms,
        )

    def generate_stream(
        self,
        question: str,
        context: list[RetrievalResult],
    ) -> Iterator[str]:
        """Stream tokens from Ollama, yielding each token as it arrives."""
        prompt = _build_prompt(question, context)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = _parse_stream_line(line)
                        if chunk is not None:
                            yield chunk
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
        except httpx.TimeoutException:
            logger.error("Ollama streaming request timed out")
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Ollama streaming returned HTTP %s", exc.response.status_code
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_stream_line(line: str) -> str | None:
    """Extract the token text from a single streaming JSON line."""
    import json

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if data.get("done"):
        return None
    return data.get("response")


def _error_result(model: str, start: float) -> GenerationResult:
    """Return a placeholder result when the LLM call fails."""
    return GenerationResult(
        answer="I'm sorry, I was unable to generate a response. Please try again later.",
        model=model,
        prompt_tokens=0,
        completion_tokens=0,
        duration_ms=(time.monotonic() - start) * 1000,
    )
