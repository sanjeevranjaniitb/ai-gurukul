from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float | None
    metadata: dict[str, Any]


class IRetriever(ABC):
    @abstractmethod
    async def aretrieve(self, query: str, *, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        raise NotImplementedError


class IReranker(ABC):
    @abstractmethod
    async def arerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        *,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank ``chunks`` for ``query``.

        When ``query_embedding`` is supplied (e.g. same vector used for vector search),
        implementers should avoid re-embedding the query to save latency and API cost.
        """
        raise NotImplementedError
