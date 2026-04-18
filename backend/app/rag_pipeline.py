"""RAG pipeline wiring retrieval and generation.

Calls ``EmbeddingStore.search()`` to find relevant chunks, then
``LLMService.generate()`` (or ``generate_stream()``) to produce an answer.
When no chunk exceeds the relevance threshold the pipeline returns a
"no relevant information" message without invoking the LLM.
"""

from __future__ import annotations

from typing import Iterator

from backend.app.config import AppConfig
from backend.app.embedding_store import EmbeddingStore
from backend.app.llm_service import LLMService
from backend.app.logging_utils import get_logger
from backend.app.models import GenerationResult, RetrievalResult

logger = get_logger("rag_pipeline")

_NO_CONTEXT_MESSAGE = (
    "I could not find relevant information in the uploaded documents "
    "to answer your question."
)


class RAGPipeline:
    """Orchestrates retrieval then generation for a user question.

    Parameters
    ----------
    embedding_store:
        Pre-initialised ``EmbeddingStore`` used for semantic search.
    llm_service:
        Pre-initialised ``LLMService`` used for answer generation.
    config:
        Application configuration.  Only ``retrieval_top_k`` is read.
    relevance_threshold:
        Minimum similarity score (0–1) a chunk must reach to be
        considered relevant.  If every retrieved chunk falls below
        this value the pipeline returns a *no relevant information*
        message without calling the LLM.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        llm_service: LLMService,
        config: AppConfig | None = None,
        relevance_threshold: float = 0.3,
    ) -> None:
        self._store = embedding_store
        self._llm = llm_service
        cfg = config or AppConfig()
        self._top_k: int = cfg.retrieval_top_k
        self.relevance_threshold: float = relevance_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> GenerationResult:
        """Retrieve context and generate a complete answer.

        Returns a ``GenerationResult`` whose ``answer`` field is either
        the LLM response or the *no relevant information* fallback.
        """
        results = self._retrieve(question)
        relevant = self._filter_relevant(results)

        if not relevant:
            logger.info("No relevant context for question: %s", question)
            return GenerationResult(
                answer=_NO_CONTEXT_MESSAGE,
                model=self._llm.model_name,
                prompt_tokens=0,
                completion_tokens=0,
                duration_ms=0.0,
            )

        logger.info(
            "Generating answer from %d relevant chunks (top score=%.3f)",
            len(relevant),
            relevant[0].score,
        )
        return self._llm.generate(question, relevant)

    def query_stream(self, question: str) -> Iterator[str]:
        """Retrieve context and stream tokens from the LLM.

        If no relevant context is found the iterator yields the
        *no relevant information* message as a single token and stops.
        """
        results = self._retrieve(question)
        relevant = self._filter_relevant(results)

        if not relevant:
            logger.info("No relevant context for question: %s", question)
            yield _NO_CONTEXT_MESSAGE
            return

        logger.info(
            "Streaming answer from %d relevant chunks (top score=%.3f)",
            len(relevant),
            relevant[0].score,
        )
        yield from self._llm.generate_stream(question, relevant)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, question: str) -> list[RetrievalResult]:
        """Run semantic search against the embedding store."""
        return self._store.search(question, top_k=self._top_k)

    def _filter_relevant(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Keep only chunks whose score meets the relevance threshold."""
        return [r for r in results if r.score >= self.relevance_threshold]
