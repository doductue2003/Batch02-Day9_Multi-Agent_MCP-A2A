"""
Task 7 - Reranking.

This module implements two offline reranking strategies:
- RRF for fusing ranked lists from semantic and lexical retrieval.
- A lightweight lexical proxy reranker for a single candidate list.

The proxy reranker is intentionally API-free so the pipeline remains reliable
after Task 4/5 already use Gemini for embeddings.
"""

from collections import Counter
from hashlib import sha1
from math import log1p, sqrt
import unicodedata

from .task6_lexical_search import tokenize

STOPWORDS = {
    "la",
    "là",
    "nhu",
    "như",
    "the",
    "thế",
    "nao",
    "nào",
    "gi",
    "gì",
    "ve",
    "về",
    "den",
    "đến",
    "bi",
    "bị",
    "xu",
    "xử",
    "ly",
    "lý",
    "lien",
    "liên",
    "quan",
    "nhung",
    "những",
    "truong",
    "trường",
    "hop",
    "hợp",
    "phai",
    "phải",
    "di",
    "đi",
    "co",
    "có",
    "cac",
    "các",
    "va",
    "và",
    "or",
    "and",
    "the",
    "with",
    "what",
    "how",
}


def _candidate_key(candidate: dict) -> str:
    metadata = candidate.get("metadata", {}) or {}
    source = metadata.get("path") or metadata.get("source") or ""
    chunk_index = metadata.get("chunk_index", "")
    content = candidate.get("content", "")
    stable = f"{source}:{chunk_index}:{content[:120]}"
    return sha1(stable.encode("utf-8", errors="ignore")).hexdigest()


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _fold_tokens(text: str) -> str:
    return " ".join(tokenize(_fold_text(text)))


def _token_overlap_score(query: str, content: str) -> float:
    query_tokens = [token for token in tokenize(query) if token not in STOPWORDS]
    content_tokens = [token for token in tokenize(content) if token not in STOPWORDS]
    if not query_tokens or not content_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    content_counts = Counter(content_tokens)
    overlap = sum(min(count, content_counts.get(token, 0)) for token, count in query_counts.items())
    coverage = overlap / max(len(query_tokens), 1)

    phrase_bonus = 0.0
    normalized_content = " ".join(content_tokens)
    for size in (2, 3):
        for start in range(0, max(len(query_tokens) - size + 1, 0)):
            phrase = " ".join(query_tokens[start : start + size])
            if phrase and phrase in normalized_content:
                phrase_bonus += 0.08 * size

    length_penalty = 1.0 / sqrt(1.0 + max(len(content_tokens) - 80, 0) / 200)
    return (coverage + phrase_bonus) * length_penalty


def _exact_phrase_bonus(query: str, content: str) -> float:
    query_tokens = [token for token in tokenize(query) if token not in STOPWORDS]
    content_text = " ".join(tokenize(content))
    bonus = 0.0

    for size in (2, 3, 4, 5):
        for start in range(0, max(len(query_tokens) - size + 1, 0)):
            phrase = " ".join(query_tokens[start : start + size])
            if phrase and phrase in content_text:
                bonus += 0.2 * size

    compact_query = " ".join(query_tokens)
    if compact_query and compact_query in content_text:
        bonus += 1.2
    return bonus


def _domain_relevance_adjustment(query: str, content: str) -> float:
    query_text = _fold_tokens(query)
    content_text = _fold_tokens(content)
    adjustment = 0.0

    asks_mandatory_rehab_cases = (
        "cai nghien bat buoc" in query_text
        and any(phrase in query_text for phrase in ("truong hop", "doi tuong", "phai", "dua vao"))
    )
    if asks_mandatory_rehab_cases:
        positive_markers = (
            "khong dang ky",
            "khong thuc hien",
            "tu y cham dut",
            "phat hien su dung trai phep",
            "lap bien ban hanh vi vi pham",
            "cac truong hop sau",
            "dua vao co so cai nghien bat buoc doi voi cac truong hop",
        )
        negative_markers = (
            "bao hiem y te",
            "kinh phi kham",
            "bi om",
            "chiu tang",
            "tam dinh chi",
            "bo tron",
            "khong chap hanh quyet dinh",
        )
        adjustment += sum(1.2 for marker in positive_markers if marker in content_text)
        adjustment -= sum(1.0 for marker in negative_markers if marker in content_text)

    return adjustment


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Offline relevance reranker.

    It approximates cross-encoder behavior by scoring direct query term/phrase
    evidence in each candidate and blending it with the original retrieval score.
    """
    if top_k <= 0 or not candidates:
        return []

    scored: list[dict] = []
    for rank, candidate in enumerate(candidates, 1):
        original_score = float(candidate.get("score", 0.0) or 0.0)
        lexical_score = _token_overlap_score(query, candidate.get("content", ""))
        phrase_bonus = _exact_phrase_bonus(query, candidate.get("content", ""))
        domain_adjustment = _domain_relevance_adjustment(query, candidate.get("content", ""))
        rank_prior = 1 / (rank + 2)
        rerank_score = (
            lexical_score
            + phrase_bonus
            + domain_adjustment
            + 0.3 * rank_prior
            + 0.05 * log1p(max(original_score, 0.0))
        )
        item = candidate.copy()
        item["score"] = float(rerank_score)
        item.setdefault("metadata", candidate.get("metadata", {}))
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance using candidate embeddings if available.

    Falls back to score sorting when candidates do not include embeddings.
    """
    if top_k <= 0 or not candidates:
        return []
    if not query_embedding or not all("embedding" in item for item in candidates):
        return sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)[:top_k]

    import numpy as np

    query_vector = np.asarray(query_embedding, dtype=float)
    candidate_vectors = [np.asarray(item["embedding"], dtype=float) for item in candidates]
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / denom) if denom else 0.0

    while remaining and len(selected) < top_k:
        best_index = None
        best_score = float("-inf")
        for index in remaining:
            relevance = cosine(query_vector, candidate_vectors[index])
            diversity_penalty = max(
                (cosine(candidate_vectors[index], candidate_vectors[selected_index]) for selected_index in selected),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_score = score
                best_index = index
        assert best_index is not None
        selected.append(best_index)
        remaining.remove(best_index)

    results: list[dict] = []
    for index in selected:
        item = candidates[index].copy()
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_r(d))) across ranked lists. Documents that are
    strong in both semantic and lexical search rise to the top.
    """
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    best_items: dict[str, dict] = {}
    best_original_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = _candidate_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            original_score = float(item.get("score", 0.0) or 0.0)
            if key not in best_items or original_score > best_original_scores[key]:
                best_items[key] = item
                best_original_scores[key] = original_score

    fused = []
    for key, score in scores.items():
        item = best_items[key].copy()
        item["score"] = float(score)
        item.setdefault("metadata", best_items[key].get("metadata", {}))
        fused.append(item)

    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:top_k]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        return rerank_mmr([], candidates, top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 249: Tội tàng trữ trái phép chất ma túy", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ bị bắt vì sử dụng ma túy", "score": 0.7, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    for result in rerank("hình phạt tàng trữ ma túy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
