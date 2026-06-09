"""
Task 5 - Semantic search over the Task 4 local vector index.

Task 4 indexed document chunks with Gemini `gemini-embedding-001` using the
RETRIEVAL_DOCUMENT task type. Queries must use the same model with
RETRIEVAL_QUERY so vectors live in the same embedding space.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .task4_chunking_indexing import (
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MANIFEST_PATH,
    _get_gemini_api_key,
    _gemini_ssl_verify,
    _normalize_vectors,
)

QUERY_TASK_TYPE = "RETRIEVAL_QUERY"


@lru_cache(maxsize=1)
def _load_index() -> tuple[list[dict], np.ndarray, dict]:
    """Load chunks, normalized embeddings, and manifest from data/index."""
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Vector index not found. Run Task 4 first: "
            "python src/task4_chunking_indexing.py"
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    manifest = (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if MANIFEST_PATH.exists()
        else {}
    )

    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding index has shape {embeddings.shape}; expected (*, {EMBEDDING_DIM})"
        )
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Chunk count mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} vectors"
        )
    if manifest.get("embedding_model") and manifest["embedding_model"] != EMBEDDING_MODEL:
        raise ValueError(
            "Index embedding model does not match Task 5 configuration: "
            f"{manifest['embedding_model']} != {EMBEDDING_MODEL}"
        )

    return chunks, embeddings, manifest


def _embed_query(query: str) -> np.ndarray:
    """Embed one query with Gemini Embedding API and normalize it."""
    import requests

    if not _gemini_ssl_verify():
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBEDDING_MODEL}:embedContent?key={_get_gemini_api_key()}"
    )
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": query}]},
        "taskType": QUERY_TASK_TYPE,
    }
    response = requests.post(
        url,
        json=payload,
        timeout=120,
        verify=_gemini_ssl_verify(),
    )
    response.raise_for_status()
    values = response.json().get("embedding", {}).get("values", [])
    if len(values) != EMBEDDING_DIM:
        raise ValueError(
            f"Query embedding dimension {len(values)} does not match {EMBEDDING_DIM}"
        )
    return np.asarray(_normalize_vectors([[float(value) for value in values]])[0], dtype=np.float32)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search semantically using cosine similarity over the local vector index.

    Args:
        query: Search query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    chunks, embeddings, _ = _load_index()
    if embeddings.shape[0] == 0:
        return []

    query_vector = _embed_query(query)
    scores = embeddings @ query_vector
    limit = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:limit]

    results: list[dict] = []
    for index in top_indices:
        chunk = chunks[int(index)]
        results.append(
            {
                "content": chunk["content"],
                "score": float(scores[int(index)]),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['metadata']} {result['content'][:100]}...")
