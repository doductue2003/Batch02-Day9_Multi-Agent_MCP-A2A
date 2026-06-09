"""
Task 4 - Load, chunk, embed, and index standardized Markdown documents.

The pipeline uses a local NumPy vector index so it runs without a database
server. Task 5 can load the same artifacts for cosine-similarity search.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# Recursive chunking is robust for mixed OCR legal text and news Markdown.
# 800 characters keeps each chunk focused while preserving enough legal context.
CHUNK_SIZE = 800
# 120 characters preserves sentences and article clauses across chunk boundaries.
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"

# Gemini Embedding is multilingual and strong for Vietnamese legal/news text.
# It requires GEMINI_API_KEY and the same model must be reused for query
# embeddings in Task 5.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"

# A local NumPy index is deterministic and needs no external database service.
VECTOR_STORE = "local_numpy"


def _validate_documents(documents: list[dict]) -> None:
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise TypeError(f"Document {index} must be a dict")
        if not isinstance(document.get("content"), str):
            raise TypeError(f"Document {index} must contain string content")
        if not isinstance(document.get("metadata", {}), dict):
            raise TypeError(f"Document {index} metadata must be a dict")


def load_documents() -> list[dict]:
    """
    Load all non-empty Markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str,
        'path': str}}.
    """
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "path": relative_path.as_posix(),
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents recursively while preserving source metadata.

    Returns:
        List of {'content': str, 'metadata': dict}.
    """
    _validate_documents(documents)
    if not documents:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "Install langchain-text-splitters before chunking documents."
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        splits = splitter.split_text(document["content"])
        for chunk_index, text in enumerate(splits):
            cleaned = text.strip()
            if not cleaned:
                continue
            chunks.append(
                {
                    "content": cleaned,
                    "metadata": {
                        **document.get("metadata", {}),
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def _get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to .env before indexing.")
    return api_key


def _gemini_ssl_verify() -> bool:
    value = os.getenv("GEMINI_SSL_VERIFY", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _extract_embedding_values(response) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        single_embedding = getattr(response, "embedding", None)
        embeddings = [single_embedding] if single_embedding is not None else []

    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None)
        if values is None and isinstance(embedding, dict):
            values = embedding.get("values")
        if values is None:
            raise ValueError("Gemini embedding response did not include values")
        vectors.append([float(value) for value in values])
    return vectors


def _normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text using Gemini Embedding REST API."""
    if not texts:
        return []

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests before embedding with Gemini.") from exc

    api_key = _get_gemini_api_key()
    if not _gemini_ssl_verify():
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBEDDING_MODEL}:batchEmbedContents?key={api_key}"
    )
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        payload = {
            "requests": [
                {
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": EMBEDDING_TASK_TYPE,
                }
                for text in batch
            ]
        }
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=120,
                    verify=_gemini_ssl_verify(),
                )
                response.raise_for_status()
                vectors = [
                    [float(value) for value in item["values"]]
                    for item in response.json().get("embeddings", [])
                ]
                if len(vectors) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} embeddings, received {len(vectors)}"
                    )
                all_vectors.extend(vectors)
                print(f"Embedded {len(all_vectors)}/{len(texts)} texts")
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    embeddings = _normalize_vectors(all_vectors)
    if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding dimension: {len(embeddings[0])} "
            f"(expected {EMBEDDING_DIM})"
        )
    return embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add a normalized embedding vector to each chunk.

    Returns:
        New chunk dicts with an 'embedding' key.
    """
    _validate_documents(chunks)
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    return [
        {**chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings)
    ]


def index_to_vectorstore(chunks: list[dict]) -> dict:
    """Persist chunk metadata and vectors as a local NumPy vector index."""
    import numpy as np

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    vectors: list[list[float]] = []

    for index, chunk in enumerate(chunks):
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Chunk {index} has an invalid embedding")
        records.append(
            {
                "id": index,
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
            }
        )
        vectors.append(embedding)

    matrix = np.asarray(vectors, dtype=np.float32)
    if not vectors:
        matrix = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    CHUNKS_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(EMBEDDINGS_PATH, matrix)

    manifest = {
        "document_count": len({r["metadata"].get("path") for r in records}),
        "chunk_count": len(records),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_method": CHUNKING_METHOD,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_task_type": EMBEDDING_TASK_TYPE,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "vector_store": VECTOR_STORE,
        "normalized_embeddings": True,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def run_pipeline() -> dict:
    """Run load -> chunk -> embed -> index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"Vector store: {VECTOR_STORE}")
    print("=" * 60)

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")
    manifest = index_to_vectorstore(embedded_chunks)
    print(f"Indexed to {INDEX_DIR}")
    return manifest


if __name__ == "__main__":
    run_pipeline()
