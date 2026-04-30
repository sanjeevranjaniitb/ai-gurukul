from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from copilot_iitb.domain.models import RAGAnswer


class IAnswerSynthesizer(ABC):
    """LLM boundary: produce grounded answers with citations."""

    @abstractmethod
    async def asynthesize(
        self,
        *,
        user_query: str,
        evidence_blocks: list[str],
        evidence_labels: list[str],
        recent_dialogue: str,
        memory_hints: str,
    ) -> RAGAnswer:
        raise NotImplementedError

    @abstractmethod
    def asynthesize_stream(
        self,
        *,
        user_query: str,
        evidence_blocks: list[str],
        evidence_labels: list[str],
        recent_dialogue: str,
        memory_hints: str,
    ) -> AsyncIterator[str]:
        """Yield answer text fragments (UTF-8 strings); citations come from retrieval, not the stream."""
        raise NotImplementedError
