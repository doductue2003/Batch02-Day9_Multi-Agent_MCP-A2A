"""Professional Streamlit UI for the Day 8 Vietnamese legal RAG pipeline."""

import json
import html
import re
import time
from pathlib import Path

import streamlit as st

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation
from src.security_guardrails import REFUSAL_MESSAGE, should_refuse_query

PROJECT_DIR = Path(__file__).parent
MANIFEST_PATH = PROJECT_DIR / "data" / "index" / "manifest.json"

st.set_page_config(
    page_title="DrugLaw RAG Workspace",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_latency", None)
    st.session_state.setdefault("pending_prompt", None)


def _source_name(source: dict) -> str:
    metadata = source.get("metadata", {}) or {}
    return metadata.get("source") or metadata.get("path") or "Unknown source"


def _source_caption(source: dict) -> str:
    metadata = source.get("metadata", {}) or {}
    source_type = metadata.get("type", "unknown")
    chunk_index = metadata.get("chunk_index", "n/a")
    score = float(source.get("score", 0.0) or 0.0)
    retrieval_source = source.get("source", "unknown")
    return (
        f"{source_type.upper()}  |  chunk {chunk_index}  |  "
        f"{retrieval_source}  |  score {score:.3f}"
    )


def _clean_answer_text(text: str) -> str:
    """Remove inline evidence markers from the visible assistant answer."""
    cleaned = re.sub(r"\s*\[[^\]]+?,\s*chunk\s*\d+\]", "", text)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _render_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #171a1f;
            --muted: #667085;
            --line: #d9dde5;
            --panel: #f7f8fa;
            --legal: #a5212b;
            --teal: #087c74;
            --amber: #a15c00;
        }
        .stApp { background: #ffffff; color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(255,255,255,.96); }
        [data-testid="stSidebar"] {
            background: #f4f5f7;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #3f4754;
        }
        .block-container {
            max-width: 1500px;
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        h1 { font-size: 1.55rem !important; margin-bottom: .1rem !important; }
        h2 { font-size: 1.05rem !important; }
        h3 { font-size: .95rem !important; }
        .app-kicker {
            color: var(--legal);
            font-size: .72rem;
            font-weight: 750;
            text-transform: uppercase;
            margin-bottom: .2rem;
        }
        .app-subtitle {
            color: var(--muted);
            font-size: .9rem;
            margin: 0 0 1rem 0;
        }
        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem 1.2rem;
            padding: .72rem .85rem;
            border: 1px solid var(--line);
            border-left: 3px solid var(--teal);
            background: var(--panel);
            font-size: .79rem;
            margin-bottom: 1rem;
        }
        .status-strip strong { color: var(--ink); }
        [data-testid="stChatMessage"] {
            border: 1px solid #e1e4e9;
            border-radius: 6px;
            padding: .2rem .45rem;
            background: #ffffff;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border-left: 3px solid var(--legal);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-left: 3px solid var(--teal);
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 5px;
            background: #ffffff;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 5px;
            padding: .6rem .7rem;
        }
        [data-testid="stMetricValue"] { font-size: 1.15rem; }
        .empty-state {
            border-top: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
            padding: 1.4rem 0;
            margin: .5rem 0 1rem 0;
        }
        .empty-state strong { color: var(--ink); }
        .empty-state p { color: var(--muted); margin: .3rem 0 0 0; }
        .source-meta {
            color: var(--muted);
            font-size: .72rem;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }
        .source-snippet {
            color: #303641;
            font-size: .83rem;
            line-height: 1.5;
        }
        .stButton > button {
            border-radius: 5px;
            border-color: #c9ced7;
            min-height: 2.3rem;
        }
        .stButton > button:hover {
            border-color: var(--legal);
            color: var(--legal);
        }
        [data-testid="stChatInput"] textarea { border-radius: 6px; }
        @media (max-width: 800px) {
            .block-container { padding-top: .8rem; }
            .status-strip { display: block; }
            .status-strip span { display: block; margin-bottom: .25rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(manifest: dict) -> tuple[str, int]:
    with st.sidebar:
        st.markdown("### DrugLaw RAG")
        st.caption("Vietnamese legal and news intelligence")

        mode = st.radio(
            "Workspace mode",
            ["Ask with citations", "Inspect retrieval"],
            captions=["Generate a grounded answer", "Review ranked evidence only"],
        )
        top_k = st.slider("Evidence chunks", min_value=3, max_value=8, value=5)

        st.divider()
        st.markdown("#### Pipeline")
        left, right = st.columns(2)
        left.metric("Documents", manifest.get("document_count", 0))
        right.metric("Chunks", manifest.get("chunk_count", 0))
        st.caption(f"Embedding: `{manifest.get('embedding_model', 'not indexed')}`")
        st.caption(f"Vector shape: `{manifest.get('embedding_dim', 0)} dimensions`")

        st.divider()
        st.markdown("#### Session")
        message_count = len(st.session_state.messages)
        st.caption(f"{message_count} messages in this conversation")
        if st.button(
            "Clear conversation",
            use_container_width=True,
            icon=":material/delete_sweep:",
        ):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.last_latency = None
            st.rerun()

        st.divider()
        st.caption(
            "Answers are generated from the indexed documents and may require "
            "professional legal review."
        )
    return mode, top_k


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No evidence has been retrieved in this session.")
        return

    for index, source in enumerate(sources, 1):
        name = _source_name(source)
        with st.expander(f"{index}. {name}", expanded=index == 1):
            st.markdown(
                f'<div class="source-meta">{_source_caption(source)}</div>',
                unsafe_allow_html=True,
            )
            snippet = source.get("content", "").strip()
            st.markdown(
                f'<div class="source-snippet">{html.escape(snippet)}</div>',
                unsafe_allow_html=True,
            )


def _render_answer_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Citations used in this answer", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {}) or {}
            label = metadata.get("source") or metadata.get("path") or f"Source {index}"
            chunk_index = metadata.get("chunk_index", "n/a")
            score = float(source.get("score", 0.0) or 0.0)
            st.markdown(
                f"- **{index}. {label}** `chunk {chunk_index}`  "
                f"`score {score:.3f}`"
            )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <strong>Start with a legal question or inspect a reported case.</strong>
          <p>The workspace retrieves from Vietnamese legal documents and curated news, then cites the evidence used.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    examples = [
        "Luật Phòng, chống ma túy quy định gì về cai nghiện?",
        "Những người nổi tiếng nào bị bắt vì liên quan ma túy?",
        "Hình phạt cho hành vi tàng trữ trái phép chất ma túy là gì?",
    ]
    columns = st.columns(3)
    for column, example in zip(columns, examples):
        if column.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()


def _run_query(prompt: str, mode: str, top_k: int) -> dict:
    started = time.perf_counter()
    guardrail = should_refuse_query(prompt)
    if guardrail.blocked:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "retrieval_source": "blocked",
            "latency": time.perf_counter() - started,
        }
    if mode == "Inspect retrieval":
        sources = retrieve(prompt, top_k=top_k)
        answer = (
            f"Retrieved {len(sources)} evidence chunks. Review their ranking "
            "and source metadata in the evidence panel."
        )
        retrieval_source = sources[0].get("source", "none") if sources else "none"
    else:
        result = generate_with_citation(prompt, top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]
        retrieval_source = result["retrieval_source"]

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "latency": time.perf_counter() - started,
    }


_init_state()
_render_css()
manifest = _load_manifest()
mode, top_k = _render_sidebar(manifest)

st.markdown('<div class="app-kicker">Evidence-grounded workspace</div>', unsafe_allow_html=True)
st.title("Vietnam Drug Law Research")
st.markdown(
    '<p class="app-subtitle">Hybrid retrieval, vectorless fallback, and cited Gemini responses.</p>',
    unsafe_allow_html=True,
)

latency_text = (
    f"{st.session_state.last_latency:.2f}s"
    if st.session_state.last_latency is not None
    else "not run"
)
st.markdown(
    f"""
    <div class="status-strip">
      <span><strong>Mode</strong> {mode}</span>
      <span><strong>Index</strong> {manifest.get('chunk_count', 0)} chunks ready</span>
      <span><strong>Last response</strong> {latency_text}</span>
      <span><strong>Citation policy</strong> required</span>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_col, evidence_col = st.columns([1.7, 1], gap="large")

with chat_col:
    st.markdown("### Conversation")
    if not st.session_state.messages:
        _render_empty_state()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(
                    f"{message.get('retrieval_source', 'unknown')} retrieval  |  "
                    f"{message.get('latency', 0):.2f}s"
                )
                _render_answer_sources(message.get("sources", []))

with evidence_col:
    st.markdown("### Evidence")
    _render_sources(st.session_state.last_sources)

prompt = st.chat_input("Ask about Vietnamese drug law or related cases")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Retrieving evidence and preparing response..."):
        try:
            result = _run_query(prompt, mode, top_k)
        except Exception as exc:
            result = {
                "answer": f"The request could not be completed: {exc}",
                "sources": [],
                "retrieval_source": "error",
                "latency": 0.0,
            }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": _clean_answer_text(result["answer"]),
            "raw_answer": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
            "latency": result["latency"],
        }
    )
    st.session_state.last_sources = result["sources"]
    st.session_state.last_latency = result["latency"]
    st.rerun()
