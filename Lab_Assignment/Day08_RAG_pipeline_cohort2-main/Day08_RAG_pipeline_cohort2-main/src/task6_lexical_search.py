"""
Task 6 - Lexical search with BM25.

The lexical corpus is the same chunk set produced by Task 4. This keeps dense
and sparse retrieval aligned for the hybrid pipeline in Task 9.
"""

import json
import re
import unicodedata
from functools import lru_cache

import numpy as np

from .task4_chunking_indexing import CHUNKS_PATH

TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)


def _normalize_text(text: str) -> str:
    """Normalize OCR/news text without removing Vietnamese accents."""
    normalized = unicodedata.normalize("NFC", text.lower())
    return normalized.replace("ý", "ý")


def tokenize(text: str) -> list[str]:
    """
    Tokenize Vietnamese text for BM25.

    Vietnamese word segmentation is not required for the assignment tests; this
    tokenizer keeps accented syllables and numbers, which works well for legal
    terms such as "điều 249", "ma túy", and named entities in news.
    """
    return TOKEN_PATTERN.findall(_normalize_text(text))


def load_corpus() -> list[dict]:
    """Load Task 4 chunk records as the BM25 corpus."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Chunk index not found. Run Task 4 first: "
            "python src/task4_chunking_indexing.py"
        )
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return [
        {
            "content": item["content"],
            "metadata": item.get("metadata", {}),
        }
        for item in chunks
        if item.get("content")
    ]


def build_bm25_index(corpus: list[dict]):
    """
    Build a BM25Okapi index from corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(document["content"]) for document in corpus]
    return BM25Okapi(tokenized_corpus)


@lru_cache(maxsize=1)
def _cached_bm25() -> tuple[list[dict], object]:
    corpus = load_corpus()
    return corpus, build_bm25_index(corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by keyword relevance using BM25.

    Args:
        query: Search query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        BM25 score descending.
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    corpus, bm25 = _cached_bm25()
    if not corpus:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]

    results: list[dict] = []
    for index in ranked_indices:
        score = float(scores[int(index)])
        if score <= 0:
            continue
        document = corpus[int(index)]
        results.append(
            {
                "content": document["content"],
                "score": score,
                "metadata": document.get("metadata", {}),
            }
        )
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    results = lexical_search("Điều 249 tàng trữ trái phép chất ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['metadata']} {result['content'][:100]}...")
