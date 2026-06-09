"""
Task 9 - Unified hybrid retrieval pipeline.

Pipeline:
1. Gemini semantic search (Task 5) + BM25 lexical search (Task 6)
2. Reciprocal Rank Fusion (Task 7)
3. Offline reranking (Task 7)
4. PageIndex/vectorless fallback (Task 8) when semantic confidence is weak
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search
from .security_guardrails import ascii_fold

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"
CANDIDATE_MULTIPLIER = 3


def _safe_semantic_search(query: str, top_k: int) -> list[dict]:
    """Allow hybrid retrieval to continue if Gemini is temporarily unavailable."""
    try:
        return semantic_search(query, top_k=top_k)
    except Exception as exc:
        print(f"Semantic search unavailable: {exc}")
        return []


def _safe_lexical_search(query: str, top_k: int) -> list[dict]:
    try:
        return lexical_search(query, top_k=top_k)
    except Exception as exc:
        print(f"Lexical search unavailable: {exc}")
        return []


def _semantic_confidence(results: list[dict]) -> float:
    if not results:
        return 0.0
    return max(float(item.get("score", 0.0) or 0.0) for item in results)


def _mark_hybrid(results: list[dict]) -> list[dict]:
    marked: list[dict] = []
    for result in results:
        item = result.copy()
        item["source"] = "hybrid"
        item.setdefault("metadata", result.get("metadata", {}))
        marked.append(item)
    return marked


def _is_famous_people_query(query: str) -> bool:
    folded = ascii_fold(query)
    return (
        "nguoi noi tieng" in folded
        or "nghe si" in folded
        or "ca si" in folded
        or "dien vien" in folded
        or "nguoi mau" in folded
    ) and "ma tuy" in folded


def _news_coverage_results(
    final_results: list[dict],
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
) -> list[dict]:
    """Prefer diverse news articles for broad celebrity/news questions."""
    selected: list[dict] = []
    seen_sources: set[str] = set()

    for candidate in [*final_results, *dense_results, *sparse_results]:
        metadata = candidate.get("metadata", {}) or {}
        if metadata.get("type") != "news":
            continue
        source_id = metadata.get("path") or metadata.get("source") or candidate.get("content", "")[:80]
        if source_id in seen_sources:
            continue
        selected.append(candidate)
        seen_sources.add(source_id)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for candidate in final_results:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieve relevant chunks using hybrid search with vectorless fallback.

    `score_threshold` is compared against the original Gemini semantic score,
    because BM25, RRF, and reranker scores use different numerical scales.
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    candidate_count = max(top_k * CANDIDATE_MULTIPLIER, top_k)
    dense_results = _safe_semantic_search(query, candidate_count)
    sparse_results = _safe_lexical_search(query, candidate_count)

    confidence = _semantic_confidence(dense_results)
    ranked_lists = [results for results in (dense_results, sparse_results) if results]
    merged = (
        rerank_rrf(ranked_lists, top_k=candidate_count)
        if ranked_lists
        else []
    )

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    if _is_famous_people_query(query):
        final_results = _news_coverage_results(final_results, dense_results, sparse_results, top_k)

    final_results = _mark_hybrid(final_results)

    if not final_results or confidence < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Nghệ sĩ nào bị bắt vì sử dụng ma túy",
        "Luật phòng chống ma túy quy định gì về cai nghiện",
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        for index, result in enumerate(retrieve(query, top_k=3), 1):
            print(
                f"{index}. [{result['score']:.3f}] [{result['source']}] "
                f"{result['content'][:100]}..."
            )
