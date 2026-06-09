"""
Task 8 - PageIndex-style vectorless fallback retrieval.

The real PageIndex service requires an account/API key. For this lab repository
we provide a runnable vectorless fallback that does not use embeddings: it reads
the Task 4 chunks and scores them with lexical/structural evidence. Results are
marked with source='pageindex' so Task 9 can use them as the fallback path.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import CHUNKS_PATH, STANDARDIZED_DIR
from .task6_lexical_search import lexical_search, tokenize

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PAGEINDEX_BASE_URL = "https://api.pageindex.ai"
PAGEINDEX_LOCAL_MANIFEST = Path(__file__).parent.parent / "data" / "index" / "pageindex_local_manifest.json"


def _pageindex_ssl_verify() -> bool:
    value = os.getenv("PAGEINDEX_SSL_VERIFY", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _pageindex_headers() -> dict[str, str]:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Missing PAGEINDEX_API_KEY in .env")
    return {"api_key": PAGEINDEX_API_KEY}


def _pageindex_request(method: str, path: str, **kwargs):
    import requests

    if not _pageindex_ssl_verify():
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    kwargs.setdefault("timeout", 60)
    kwargs.setdefault("verify", _pageindex_ssl_verify())
    headers = kwargs.pop("headers", {})
    response = requests.request(
        method,
        f"{PAGEINDEX_BASE_URL}{path}",
        headers={**_pageindex_headers(), **headers},
        **kwargs,
    )
    response.raise_for_status()
    return response


def upload_documents() -> dict:
    """
    Prepare documents for PageIndex-style retrieval.

    If the official PageIndex SDK and API key are available, this function can be
    extended to upload files. In the current lab it records a local manifest so
    the fallback retrieval path is explicit and reproducible.
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        documents.append(
            {
                "source": md_file.name,
                "path": md_file.relative_to(STANDARDIZED_DIR).as_posix(),
                "type": md_file.parent.name,
                "size": md_file.stat().st_size,
            }
        )

    PAGEINDEX_LOCAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "local_vectorless_fallback",
        "document_count": len(documents),
        "documents": documents,
        "pageindex_api_configured": bool(PAGEINDEX_API_KEY),
    }
    PAGEINDEX_LOCAL_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def list_pageindex_documents(limit: int = 50, offset: int = 0) -> dict:
    """List documents already uploaded to the PageIndex cloud account."""
    response = _pageindex_request("GET", f"/docs/?limit={limit}&offset={offset}")
    return response.json()


def upload_pdf_to_pageindex(file_path: str | Path) -> dict:
    """
    Upload one PDF to PageIndex cloud.

    This is explicit because PageIndex charges credits per processed page.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as file_obj:
        response = _pageindex_request(
            "POST",
            "/doc/",
            files={"file": file_obj},
            data={"if_retrieval": True},
            timeout=300,
        )
    return response.json()


def _cloud_doc_ids() -> list[str]:
    if not PAGEINDEX_LOCAL_MANIFEST.exists():
        return []
    manifest = json.loads(PAGEINDEX_LOCAL_MANIFEST.read_text(encoding="utf-8"))
    return [doc["doc_id"] for doc in manifest.get("cloud_documents", []) if doc.get("doc_id")]


def _cloud_pageindex_search(query: str, top_k: int) -> list[dict]:
    results: list[dict] = []
    for doc_id in _cloud_doc_ids():
        response = _pageindex_request(
            "GET",
            f"/doc/{doc_id}/",
            params={"query": query},
            timeout=120,
        )
        data = response.json()
        retrieval_items = data.get("results") or data.get("result") or []
        if isinstance(retrieval_items, dict):
            retrieval_items = retrieval_items.get("results", [])
        for item in retrieval_items:
            text = item.get("text") or item.get("content") or item.get("markdown") or str(item)
            score = item.get("score") or item.get("relevance_score") or 1.0
            results.append(
                {
                    "content": text,
                    "score": float(score),
                    "metadata": {"doc_id": doc_id, "raw": item},
                    "source": "pageindex",
                }
            )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        return []
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def _structural_score(query: str, item: dict) -> float:
    content = item.get("content", "")
    metadata = item.get("metadata", {}) or {}
    query_tokens = set(tokenize(query))
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    content_token_set = set(content_tokens)
    overlap = len(query_tokens & content_token_set) / max(len(query_tokens), 1)

    heading_bonus = 0.0
    first_lines = "\n".join(content.splitlines()[:4]).lower()
    for token in query_tokens:
        if token in first_lines:
            heading_bonus += 0.05

    source_bonus = 0.03 if metadata.get("type") == "legal" else 0.0
    phrase_bonus = 0.12 if " ".join(tokenize(query)[:2]) in " ".join(content_tokens) else 0.0
    return overlap + min(heading_bonus, 0.2) + source_bonus + phrase_bonus


def _local_vectorless_search(query: str, top_k: int) -> list[dict]:
    """
    Vectorless fallback using lexical search plus simple document structure cues.
    """
    lexical_candidates = lexical_search(query, top_k=max(top_k * 4, 10))
    seen = {candidate["content"] for candidate in lexical_candidates}

    # Add a few structural candidates so fallback still works for obscure terms.
    for item in _load_chunks():
        if item.get("content") in seen:
            continue
        score = _structural_score(query, item)
        if score > 0:
            lexical_candidates.append(
                {
                    "content": item["content"],
                    "score": score,
                    "metadata": item.get("metadata", {}),
                }
            )
            seen.add(item["content"])

    results: list[dict] = []
    for candidate in lexical_candidates:
        item = candidate.copy()
        item["score"] = float(candidate.get("score", 0.0)) + _structural_score(query, candidate)
        item["source"] = "pageindex"
        item.setdefault("metadata", candidate.get("metadata", {}))
        results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: max(top_k, 0)]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval fallback.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'pageindex'}.
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    if PAGEINDEX_API_KEY and _cloud_doc_ids():
        try:
            cloud_results = _cloud_pageindex_search(query, top_k)
            if cloud_results:
                return cloud_results
        except Exception:
            pass

    # Local fallback keeps Task 8 and Task 9 runnable when no cloud documents
    # have been uploaded, or if PageIndex is temporarily unavailable.
    return _local_vectorless_search(query, top_k)


if __name__ == "__main__":
    upload_documents()
    for result in pageindex_search("hình phạt sử dụng ma túy", top_k=3):
        print(f"[{result['score']:.3f}] {result['metadata']} {result['content'][:100]}...")
