"""Serve the custom web UI and JSON API for the RAG workspace."""

from __future__ import annotations

import json
import os
import time
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.security_guardrails import REFUSAL_MESSAGE, should_refuse_query
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation

PROJECT_DIR = Path(__file__).parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MANIFEST_PATH = PROJECT_DIR / "data" / "index" / "manifest.json"
DEFAULT_PORT = int(os.getenv("RAG_WEB_PORT", "8787"))


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


def _safe_source(source: dict) -> dict:
    metadata = source.get("metadata", {}) or {}
    return {
        "content": (source.get("content") or "")[:1400],
        "score": float(source.get("score", 0.0) or 0.0),
        "source": source.get("source", "unknown"),
        "metadata": {
            "source": metadata.get("source") or metadata.get("path") or "Unknown source",
            "type": metadata.get("type", "unknown"),
            "chunk_index": metadata.get("chunk_index", "n/a"),
        },
    }


class RAGRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/manifest":
            manifest = {}
            if MANIFEST_PATH.exists():
                manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            self._send_json(manifest)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        started = time.perf_counter()
        try:
            body = _read_json_body(self)
            message = str(body.get("message", "")).strip()
            mode = str(body.get("mode", "answer"))
            top_k = int(body.get("top_k", 5))
            top_k = max(3, min(8, top_k))

            guardrail = should_refuse_query(message)
            if guardrail.blocked:
                self._send_json(
                    {
                        "answer": REFUSAL_MESSAGE,
                        "sources": [],
                        "retrieval_source": "blocked",
                        "latency": time.perf_counter() - started,
                    }
                )
                return

            if mode == "inspect":
                sources = retrieve(message, top_k=top_k)
                answer = f"Retrieved {len(sources)} evidence chunks."
                retrieval_source = sources[0].get("source", "none") if sources else "none"
            else:
                result = generate_with_citation(message, top_k=top_k)
                answer = result["answer"]
                sources = result["sources"]
                retrieval_source = result["retrieval_source"]

            self._send_json(
                {
                    "answer": answer,
                    "sources": [_safe_source(source) for source in sources],
                    "retrieval_source": retrieval_source,
                    "latency": time.perf_counter() - started,
                }
            )
        except Exception as exc:
            self._send_json(
                {"error": str(exc), "answer": "The request could not be completed."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", DEFAULT_PORT), RAGRequestHandler)
    (PROJECT_DIR / "frontend-ui.log").write_text(
        f"RAG web UI running at http://127.0.0.1:{DEFAULT_PORT}\n",
        encoding="utf-8",
    )
    try:
        print(f"RAG web UI running at http://127.0.0.1:{DEFAULT_PORT}", flush=True)
    except Exception:
        pass
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (PROJECT_DIR / "frontend-ui.err.log").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise
