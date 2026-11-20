"""Hybrid retrieval combining dense vector search and BM25 sparse search with RRF.

Implements vector top-k via ChromaDB (fastembed BAAI/bge-small-en-v1.5) and
lexical top-k via rank_bm25.BM25Okapi over the same chunk corpus, fused
using Reciprocal Rank Fusion (RRF). Supports metadata filtering by document_type
and other attributes.
"""

from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Optional

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Keep the rebuilt corpus separate from the original database, which may be
# held open by a running local app process.
DEFAULT_CHROMA_DIR = Path("data/chroma_updated")
DEFAULT_COLLECTION_NAME = "real_estate_collection"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60  # Standard Reciprocal Rank Fusion constant


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25."""
    return re.findall(r"\w+", text.lower())


def build_chroma_where(
    source: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build a Chroma-compatible where filter clause from source and custom filters."""
    clauses: list[dict[str, Any]] = []

    if source:
        clauses.append({"source": source.lower().strip()})

    if filters:
        for k, v in filters.items():
            if v is not None:
                clauses.append({k: v.lower() if isinstance(v, str) else v})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class HybridRetriever:
    """Hybrid vector + BM25 retriever with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Optional[chromadb.ClientAPI] = None,
        embedder: Optional[Any] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Initialize hybrid retriever and build BM25 index over collection documents.

        Args:
            chroma_dir: Path to Chroma persistent directory.
            collection_name: Name of collection in Chroma.
            client: Optional pre-configured Chroma client (useful for tests/in-memory).
            embedder: Optional pre-configured embedding model instance.
            model_name: Fastembed model identifier if embedder is None.
        """
        self.chroma_dir = chroma_dir or DEFAULT_CHROMA_DIR
        self.collection_name = collection_name
        self._embedder = embedder
        self.model_name = model_name

        if client is not None:
            self.client = client
        else:
            self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Internal corpus store: doc_id -> {"text": str, "metadata": dict}
        self.doc_store: dict[str, dict[str, Any]] = {}
        self.doc_ids: list[str] = []
        self.bm25: Optional[BM25Okapi] = None

        self._initialize_corpus()

    @property
    def embedder(self) -> Any:
        """Lazy loader for fastembed TextEmbedding model."""
        if self._embedder is None:
            logger.info("Initializing fastembed model '%s'...", self.model_name)
            configured_threads = os.getenv("FASTEMBED_THREADS")
            self._embedder = TextEmbedding(
                model_name=self.model_name,
                threads=int(configured_threads) if configured_threads else None,
            )
        return self._embedder

    def _initialize_corpus(self) -> None:
        """Load corpus chunks from Chroma or chunks.json cache and fit BM25Okapi."""
        raw_data = self.collection.get(include=["documents", "metadatas"])
        ids = raw_data.get("ids", [])
        documents = raw_data.get("documents", [])
        metadatas = raw_data.get("metadatas", [])

        self.doc_store = {}
        self.doc_ids = []
        tokenized_corpus: list[list[str]] = []

        for doc_id, doc_text, meta in zip(ids, documents, metadatas):
            self.doc_ids.append(doc_id)
            self.doc_store[doc_id] = {
                "text": doc_text or "",
                "metadata": meta or {},
            }
            tokenized_corpus.append(tokenize(doc_text or ""))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("Fitted BM25 index over %d documents", len(tokenized_corpus))
        else:
            self.bm25 = None
            logger.info("BM25 index initialized with empty corpus")

    def reload(self) -> None:
        """Reload collection and re-index BM25 corpus."""
        self._initialize_corpus()

    def _matches_filters(
        self,
        meta: dict[str, Any],
        source: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Check if metadata matches source and custom filter criteria."""
        if source:
            meta_src = str(meta.get("source", "")).lower()
            if meta_src != source.lower():
                return False

        if filters:
            for k, expected in filters.items():
                if expected is None:
                    continue
                actual = meta.get(k)
                if isinstance(expected, str):
                    if str(actual).lower() != expected.lower():
                        return False
                elif actual != expected:
                    return False

        return True

    def retrieve(
        self,
        query: str,
        k: int = 5,
        source: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid retrieval with Reciprocal Rank Fusion.

        Args:
            query: User search query string.
            k: Top-k documents to return.
            source: Optional metadata filter: 'darglobal'.
            filters: Optional dictionary of metadata key-value filters.

        Returns:
            List of dictionaries: [{"text": str, "metadata": dict, "score": float}, ...]
        """
        if not query.strip() or not self.doc_ids:
            return []

        # 1. Dense Vector Retrieval
        vector_ranked_ids: list[str] = []
        where_clause = build_chroma_where(source=source, filters=filters)
        candidate_pool = min(max(k * 4, 20), len(self.doc_ids))

        try:
            # Generate query embedding
            raw_emb = list(self.embedder.embed([query]))[0]
            query_vec = raw_emb.tolist() if hasattr(raw_emb, "tolist") else list(raw_emb)

            v_res = self.collection.query(
                query_embeddings=[query_vec],
                n_results=candidate_pool,
                where=where_clause,
            )
            if v_res.get("ids") and v_res["ids"][0]:
                vector_ranked_ids = v_res["ids"][0]
        except Exception as err:
            logger.warning("Vector retrieval query failed (%s); continuing with BM25 only", err)

        # 2. Sparse BM25 Retrieval
        bm25_ranked_ids: list[str] = []
        if self.bm25 is not None:
            query_tokens = tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)

            scored_candidates: list[tuple[float, str]] = []
            for doc_id, score in zip(self.doc_ids, bm25_scores):
                # Filter candidates by metadata
                doc_meta = self.doc_store[doc_id]["metadata"]
                if not self._matches_filters(doc_meta, source=source, filters=filters):
                    continue
                if score > 0:
                    scored_candidates.append((score, doc_id))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            bm25_ranked_ids = [doc_id for _, doc_id in scored_candidates[:candidate_pool]]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = defaultdict(float)

        for rank, doc_id in enumerate(vector_ranked_ids, start=1):
            rrf_scores[doc_id] += 1.0 / (RRF_K + rank)

        for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
            rrf_scores[doc_id] += 1.0 / (RRF_K + rank)

        if not rrf_scores:
            return []

        # Sort all retrieved items by combined RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_docs[:k]

        results = [
            {
                "text": self.doc_store[doc_id]["text"],
                "metadata": self.doc_store[doc_id]["metadata"],
                "score": round(score, 5),
            }
            for doc_id, score in top_k
            if doc_id in self.doc_store
        ]

        return results


# Global singleton instance for retrieve() convenience function
_DEFAULT_RETRIEVER: Optional[HybridRetriever] = None


def get_retriever(reload: bool = False) -> HybridRetriever:
    """Retrieve or create the global default HybridRetriever."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None or reload or not getattr(_DEFAULT_RETRIEVER, "doc_ids", None):
        _DEFAULT_RETRIEVER = HybridRetriever()
    return _DEFAULT_RETRIEVER


def retrieve(
    query: str,
    k: int = 5,
    source: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks using hybrid vector + BM25 search with Reciprocal Rank Fusion.

    Args:
        query: Search query string.
        k: Maximum number of chunks to return (default: 5).
        source: Optional source filter ('darglobal').
        filters: Optional dictionary of metadata key-value filters.

    Returns:
        List of dictionaries with format:
            {"text": str, "metadata": dict, "score": float}
    """
    retriever = get_retriever()
    return retriever.retrieve(query=query, k=k, source=source, filters=filters)
