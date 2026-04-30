from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.llm import IAnswerSynthesizer
from copilot_iitb.domain.models import RAGAnswer, SourceCitation

_STREAM_FORMAT_ADDON = (
    "\n\nSTREAMING FORMAT (this turn only):\n"
    "- Ignore any requirement to output JSON.\n"
    "- Reply in plain language or markdown only. Ground claims only in the provided evidence.\n"
    "- If evidence is insufficient or conflicting, say so clearly in prose.\n"
)


class OpenAIStructuredSynthesizer(IAnswerSynthesizer):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.llm_provider == "azure_openai":
            self._client = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                api_key=settings.azure_openai_api_key,
            )
            self._chat_model = settings.azure_openai_chat_deployment
        else:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required when LLM_PROVIDER=openai.")
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            self._chat_model = settings.openai_model

    def _system_prompt(self) -> str:
        base = self._settings.rag_system_prompt
        addon = (self._settings.rag_instruction_priority_addon or "").strip()
        if not addon:
            return base
        return f"{base.rstrip()}\n\n{addon}"

    def _system_prompt_stream(self) -> str:
        return self._system_prompt().rstrip() + _STREAM_FORMAT_ADDON

    def _chat_completion_kwargs(self) -> dict[str, Any]:
        return {
            "max_tokens": self._settings.rag_synthesis_max_output_tokens,
            "timeout": self._settings.rag_synthesis_timeout_seconds,
        }

    async def asynthesize(
        self,
        *,
        user_query: str,
        evidence_blocks: list[str],
        evidence_labels: list[str],
        recent_dialogue: str,
        memory_hints: str,
    ) -> RAGAnswer:
        system = self._system_prompt()
        user = json.dumps(
            {
                "user_query": user_query,
                "evidence": [{"label": lab, "text": txt} for lab, txt in zip(evidence_labels, evidence_blocks)],
                "recent_dialogue": recent_dialogue,
                "memory_hints": memory_hints,
            },
            ensure_ascii=False,
        )

        resp = await self._client.chat.completions.create(
            model=self._chat_model,
            temperature=self._settings.rag_synthesis_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            **self._chat_completion_kwargs(),
        )
        raw = resp.choices[0].message.content or "{}"
        data: dict[str, Any] = json.loads(raw)
        return RAGAnswer.model_validate(data)

    async def asynthesize_stream(
        self,
        *,
        user_query: str,
        evidence_blocks: list[str],
        evidence_labels: list[str],
        recent_dialogue: str,
        memory_hints: str,
    ) -> AsyncIterator[str]:
        system = self._system_prompt_stream()
        user = json.dumps(
            {
                "user_query": user_query,
                "evidence": [{"label": lab, "text": txt} for lab, txt in zip(evidence_labels, evidence_blocks)],
                "recent_dialogue": recent_dialogue,
                "memory_hints": memory_hints,
            },
            ensure_ascii=False,
        )
        stream = await self._client.chat.completions.create(
            model=self._chat_model,
            temperature=self._settings.rag_synthesis_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
            **self._chat_completion_kwargs(),
        )
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            piece = choice.delta.content if choice.delta else None
            if piece:
                yield piece


class HeuristicOfflineSynthesizer(IAnswerSynthesizer):
    """Deterministic fallback when no chat model API key is configured."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def asynthesize(
        self,
        *,
        user_query: str,
        evidence_blocks: list[str],
        evidence_labels: list[str],
        recent_dialogue: str,
        memory_hints: str,
    ) -> RAGAnswer:
        _ = recent_dialogue, memory_hints
        if not evidence_blocks:
            return RAGAnswer(
                answer=self._settings.no_context_answer,
                citations=[],
                confidence=0.0,
                insufficient_evidence=True,
                follow_up_question="Can you upload or point me to the relevant internal document?",
            )
        best_label = evidence_labels[0]
        best_text = evidence_blocks[0][:400]
        return RAGAnswer(
            answer=(
                "Offline mode (no LLM API key): here is the top retrieved excerpt.\n\n"
                f"[{best_label}] {best_text}"
            ),
            citations=[
                SourceCitation(
                    chunk_id=best_label,
                    document_id=None,
                    title=None,
                    snippet=best_text[:240],
                    score=None,
                )
            ],
            confidence=0.35,
            insufficient_evidence=True,
            follow_up_question=(
                "Set LLM_PROVIDER=openai with OPENAI_API_KEY, or LLM_PROVIDER=azure_openai with Azure OpenAI "
                "env vars, to enable full grounded answers with citations."
            ),
        )

    async def asynthesize_stream(
        self,
        *,
        user_query: str,
        evidence_blocks: list[str],
        evidence_labels: list[str],
        recent_dialogue: str,
        memory_hints: str,
    ) -> AsyncIterator[str]:
        full = (
            await self.asynthesize(
                user_query=user_query,
                evidence_blocks=evidence_blocks,
                evidence_labels=evidence_labels,
                recent_dialogue=recent_dialogue,
                memory_hints=memory_hints,
            )
        ).answer
        if full:
            yield full


def build_synthesizer(settings: Settings) -> IAnswerSynthesizer:
    if settings.llm_provider == "azure_openai":
        return OpenAIStructuredSynthesizer(settings)
    if settings.openai_api_key:
        return OpenAIStructuredSynthesizer(settings)
    return HeuristicOfflineSynthesizer(settings)
