"""Unit tests for EmbeddingStore.

All heavy dependencies (SentenceTransformer, ChromaDB) are mocked at the
sys.modules level so that tests run in milliseconds without downloading
models or touching disk.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.models import DocumentChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "doc-1",
    page: int = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=doc_id,
        page_number=page,
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
    )


def _fake_encode(texts, show_progress_bar=False):
    """Return deterministic 384-dim unit embeddings seeded by text hash."""
    embeddings = []
    for t in texts:
        rng = np.random.RandomState(abs(hash(t)) % (2**31))
        vec = rng.randn(384).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        embeddings.append(vec)
    return np.array(embeddings)


class _InMemoryCollection:
    """Minimal in-memory ChromaDB collection stand-in."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def count(self) -> int:
        return len(self._store)

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, cid in enumerate(ids):
            self._store[cid] = {
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i],
            }

    def query(self, query_embeddings, n_results, include):
        if not self._store:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        query_vec = np.array(query_embeddings[0])
        scored = []
        for cid, data in self._store.items():
            stored_vec = np.array(data["embedding"])
            cos_sim = float(
                np.dot(query_vec, stored_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-10)
            )
            # ChromaDB cosine distance is in [0, 2]
            distance = 1.0 - cos_sim
            scored.append((cid, data, distance))

        scored.sort(key=lambda x: x[2])
        scored = scored[:n_results]

        return {
            "ids": [[s[0] for s in scored]],
            "documents": [[s[1]["document"] for s in scored]],
            "metadatas": [[s[1]["metadata"] for s in scored]],
            "distances": [[s[2] for s in scored]],
        }


# ---------------------------------------------------------------------------
# Fixture: build an EmbeddingStore with fully mocked heavy deps
# ---------------------------------------------------------------------------


@pytest.fixture()
def store_and_collection():
    """Create an EmbeddingStore with mocked SentenceTransformer and ChromaDB.

    Returns (store, collection) so tests can inspect the in-memory collection.
    """
    collection = _InMemoryCollection()

    # Build a fake SentenceTransformer class
    mock_model_instance = MagicMock()
    mock_model_instance.encode = MagicMock(side_effect=_fake_encode)

    mock_st_class = MagicMock(return_value=mock_model_instance)

    # Build a fake chromadb module
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = collection

    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client

    # Temporarily inject mocks into sys.modules so the module-level imports
    # in embedding_store.py resolve instantly.
    st_module = types.ModuleType("sentence_transformers")
    st_module.SentenceTransformer = mock_st_class  # type: ignore[attr-defined]

    saved_modules = {}
    modules_to_mock = {
        "sentence_transformers": st_module,
        "chromadb": mock_chromadb,
    }

    for mod_name, mod_obj in modules_to_mock.items():
        saved_modules[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = mod_obj

    # Remove cached embedding_store module so it re-imports with our mocks
    es_key = "backend.app.embedding_store"
    saved_es = sys.modules.pop(es_key, None)

    try:
        # Force a fresh import that picks up our mocked modules
        import importlib
        mod = importlib.import_module("backend.app.embedding_store")
        importlib.reload(mod)

        es = mod.EmbeddingStore(
            collection_name="test_collection",
            persist_directory="/tmp/fake_chroma",
        )
        yield es, collection
    finally:
        # Restore original modules
        for mod_name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original
        if saved_es is not None:
            sys.modules[es_key] = saved_es
        else:
            sys.modules.pop(es_key, None)


@pytest.fixture()
def store(store_and_collection):
    """Convenience fixture returning just the EmbeddingStore."""
    return store_and_collection[0]


@pytest.fixture()
def collection(store_and_collection):
    """Convenience fixture returning just the in-memory collection."""
    return store_and_collection[1]


# ---------------------------------------------------------------------------
# Tests: add_chunks
# ---------------------------------------------------------------------------


class TestAddChunks:
    """Tests for EmbeddingStore.add_chunks()."""

    def test_add_stores_chunks(self, store, collection):
        """Adding chunks should store them in the collection."""
        chunks = [
            _make_chunk("c1", "Python is a programming language."),
            _make_chunk("c2", "ChromaDB is a vector database."),
        ]
        store.add_chunks(chunks)
        assert collection.count() == 2

    def test_add_empty_list_is_noop(self, store, collection):
        """Adding an empty list should not modify the store."""
        store.add_chunks([])
        assert collection.count() == 0

    def test_upsert_overwrites_existing_chunk(self, store, collection):
        """Upserting a chunk with the same ID should overwrite the old one."""
        store.add_chunks([_make_chunk("c1", "Original text.")])
        store.add_chunks([_make_chunk("c1", "Updated text.")])

        assert collection.count() == 1
        assert collection._store["c1"]["document"] == "Updated text."

    def test_metadata_is_stored_correctly(self, store, collection):
        """Chunk metadata (document_id, page_number, etc.) should be persisted."""
        chunk = _make_chunk("c1", "Some text.", doc_id="doc-42", page=7)
        store.add_chunks([chunk])

        stored_meta = collection._store["c1"]["metadata"]
        assert stored_meta["document_id"] == "doc-42"
        assert stored_meta["page_number"] == 7
        assert stored_meta["token_count"] == chunk.token_count
        assert stored_meta["start_char"] == 0
        assert stored_meta["end_char"] == len("Some text.")

    def test_add_multiple_batches(self, store, collection):
        """Multiple add_chunks calls should accumulate in the store."""
        store.add_chunks([_make_chunk("c1", "First chunk.")])
        store.add_chunks([
            _make_chunk("c2", "Second chunk."),
            _make_chunk("c3", "Third chunk."),
        ])
        assert collection.count() == 3


# ---------------------------------------------------------------------------
# Tests: search
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for EmbeddingStore.search()."""

    def test_search_empty_store_returns_empty(self, store):
        """Searching an empty store should return an empty list."""
        results = store.search("anything")
        assert results == []

    def test_search_returns_retrieval_results(self, store):
        """Search should return RetrievalResult objects with correct fields."""
        store.add_chunks([
            _make_chunk("c1", "Test document content.", doc_id="d42", page=3),
        ])
        results = store.search("test", top_k=1)

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, RetrievalResult)
        assert r.chunk.chunk_id == "c1"
        assert r.chunk.document_id == "d42"
        assert r.chunk.page_number == 3
        assert r.chunk.text == "Test document content."
        assert isinstance(r.score, float)
        assert isinstance(r.distance, float)

    def test_search_score_in_valid_range(self, store):
        """Scores should be in [0, 1] (cosine similarity mapped from distance)."""
        store.add_chunks([_make_chunk("c1", "Machine learning algorithms.")])
        results = store.search("machine learning", top_k=1)

        assert len(results) == 1
        assert 0.0 <= results[0].score <= 1.0
        assert results[0].distance >= 0.0

    def test_search_respects_top_k(self, store):
        """Search should return at most top_k results."""
        chunks = [
            _make_chunk(f"c{i}", f"Document chunk number {i}.")
            for i in range(10)
        ]
        store.add_chunks(chunks)

        results = store.search("document", top_k=3)
        assert len(results) == 3

    def test_search_top_k_larger_than_store(self, store):
        """When top_k exceeds stored chunks, return all available chunks."""
        store.add_chunks([_make_chunk("c1", "Only one chunk.")])
        results = store.search("chunk", top_k=10)
        assert len(results) == 1

    def test_results_sorted_by_score_descending(self, store):
        """Results should be sorted from most to least relevant."""
        chunks = [
            _make_chunk("c1", "The capital of France is Paris."),
            _make_chunk("c2", "Dogs are popular pets around the world."),
            _make_chunk("c3", "Paris is known for the Eiffel Tower."),
        ]
        store.add_chunks(chunks)

        results = store.search("Tell me about Paris", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_relevant_results_first(self, store):
        """Semantically relevant chunks should rank higher than unrelated ones."""
        chunks = [
            _make_chunk("c1", "Machine learning is a subset of artificial intelligence."),
            _make_chunk("c2", "Cooking pasta requires boiling water and salt."),
            _make_chunk("c3", "Neural networks are used in deep learning."),
        ]
        store.add_chunks(chunks)

        results = store.search("What is deep learning?", top_k=2)
        assert len(results) == 2
        # The cooking chunk should not be in the top-2 for a deep learning query
        top_ids = {r.chunk.chunk_id for r in results}
        assert "c2" not in top_ids

    def test_search_preserves_chunk_metadata(self, store):
        """All DocumentChunk fields should be reconstructed from stored metadata."""
        original = _make_chunk("c1", "Metadata preservation test.", doc_id="doc-99", page=5)
        store.add_chunks([original])

        results = store.search("metadata", top_k=1)
        chunk = results[0].chunk
        assert chunk.chunk_id == original.chunk_id
        assert chunk.document_id == original.document_id
        assert chunk.page_number == original.page_number
        assert chunk.token_count == original.token_count
        assert chunk.start_char == original.start_char
        assert chunk.end_char == original.end_char

    def test_search_with_default_top_k(self, store):
        """Default top_k should be 5 as per design spec."""
        chunks = [
            _make_chunk(f"c{i}", f"Chunk about topic {i}.")
            for i in range(10)
        ]
        store.add_chunks(chunks)

        results = store.search("topic")
        assert len(results) == 5

    def test_search_different_documents(self, store):
        """Chunks from different documents should all be searchable."""
        chunks = [
            _make_chunk("c1", "Python programming basics.", doc_id="doc-a"),
            _make_chunk("c2", "Java programming basics.", doc_id="doc-b"),
            _make_chunk("c3", "Cooking recipes for beginners.", doc_id="doc-c"),
        ]
        store.add_chunks(chunks)

        results = store.search("programming", top_k=3)
        doc_ids = {r.chunk.document_id for r in results}
        assert len(doc_ids) > 1  # Results should span multiple documents
