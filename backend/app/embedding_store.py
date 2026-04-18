"""Embedding store using sentence-transformers and ChromaDB.

Provides semantic embedding generation and persistent vector storage
for document chunks, with top-k similarity search retrieval.
"""

from __future__ import annotations

import os
import ssl

# Force offline mode and disable SSL verification for HuggingFace Hub.
# This works around macOS Python SSL certificate issues when the model
# is already cached locally.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

# Monkey-patch SSL to skip verification as a last resort for cached models
_original_create_default_context = ssl.create_default_context

def _permissive_ssl_context(*args, **kwargs):
    ctx = _original_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl.create_default_context = _permissive_ssl_context

from sentence_transformers import SentenceTransformer
import chromadb

from backend.app.config import AppConfig
from backend.app.logging_utils import get_logger
from backend.app.models import DocumentChunk, RetrievalResult

logger = get_logger("embedding_store")


class EmbeddingStore:
    """Embeds and stores document chunks in ChromaDB for semantic retrieval.

    Parameters
    ----------
    collection_name:
        Name of the ChromaDB collection to use.
    persist_directory:
        Path to the ChromaDB persistent storage directory.
        Defaults to ``AppConfig.chroma_persist_dir``.
    embedding_model:
        Name of the sentence-transformers model to load.
        Defaults to ``AppConfig.embedding_model`` (``"all-MiniLM-L6-v2"``).
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        config = AppConfig()
        self._persist_directory = persist_directory or config.chroma_persist_dir
        self._model_name = embedding_model or config.embedding_model

        logger.info(
            "Loading sentence-transformers model '%s'", self._model_name
        )
        self._model = SentenceTransformer(self._model_name)

        logger.info(
            "Connecting to ChromaDB at '%s'", self._persist_directory
        )
        self._client = chromadb.PersistentClient(
            path=self._persist_directory
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready (%d existing items)",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and upsert document chunks into ChromaDB.

        Parameters
        ----------
        chunks:
            List of ``DocumentChunk`` objects to store.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "page_number": c.page_number,
                "token_count": c.token_count,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]

        embeddings = self._model.encode(texts, show_progress_bar=False).tolist()

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks into collection", len(chunks))

    def search(
        self, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Retrieve the most semantically similar chunks for *query*.

        Parameters
        ----------
        query:
            The user question or search string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Results sorted by relevance (highest score first).
        """
        if self._collection.count() == 0:
            return []

        query_embedding = self._model.encode(
            [query], show_progress_bar=False
        ).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results: list[RetrievalResult] = []
        for idx in range(len(results["ids"][0])):
            meta = results["metadatas"][0][idx]
            distance = results["distances"][0][idx]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to a similarity score in [0, 1].
            score = 1.0 - (distance / 2.0)

            chunk = DocumentChunk(
                chunk_id=results["ids"][0][idx],
                text=results["documents"][0][idx],
                document_id=meta["document_id"],
                page_number=meta["page_number"],
                token_count=meta["token_count"],
                start_char=meta["start_char"],
                end_char=meta["end_char"],
            )
            retrieval_results.append(
                RetrievalResult(chunk=chunk, score=score, distance=distance)
            )

        # Sort by score descending (most relevant first).
        retrieval_results.sort(key=lambda r: r.score, reverse=True)
        return retrieval_results
