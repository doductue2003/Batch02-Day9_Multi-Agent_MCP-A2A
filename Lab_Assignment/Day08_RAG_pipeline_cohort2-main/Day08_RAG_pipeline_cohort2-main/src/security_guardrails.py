"""Shared guardrails for prompt injection and unsafe user requests."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_QUERY_CHARS = 2000

REFUSAL_MESSAGE = (
    "Tôi chỉ trả lời câu hỏi dựa trên tài liệu đã lập chỉ mục và sẽ không làm theo "
    "yêu cầu bỏ qua quy tắc, tiết lộ prompt, hoặc ép định dạng đầu ra."
)

INJECTION_PATTERNS = (
    r"\bignore\b.*\binstructions?\b",
    r"\bignore\b.*\bprevious\b.*\binstructions?\b",
    r"\bignore\b.*\bhướng dẫn\b",
    r"\bbypass\b.*\bguardrails?\b",
    r"\b(system|developer)\s*prompt\b",
    r"\breveal\b.*\bprompt\b",
    r"\bshow\b.*\bprompt\b",
    r"\boutput\s+only\b",
    r"\bjust\s+say\b",
    r"\bact\s+as\b",
    r"\byou are now\b",
    r"\boverride\b",
    r"\bcitation\s+spoof\b",
    r"\bcitation\b.*\b(fake|spoof|giả)\b",
    r"\btrich dan\b.*\b(fake|spoof|gia)\b",
    r"\b(mot tu|one word)\b",
    r"\b(chi bang|only)\b",
    r"\b(dong vai tro|role)\b",
    r"\b(he thong|system)\b",
    r"\btrả lời\s+chỉ\b",
    r"\bbỏ qua\b.*\bhướng dẫn\b",
    r"\btiết lộ\b.*\bprompt\b",
    r"\bchỉ in ra\b",
    r"\bkhông\s+nhắc\s+nguồn\b",
)

HARMFUL_INTENT_PATTERNS = (
    r"\bhide\b.*\btrace",
    r"\bche giu\b.*\bdau vet\b",
    r"\bevade\b",
    r"\bbypass\b.*\blaw\b",
    r"\btron\b.*\bluat\b",
    r"\bvi pham phap luat\b",
    r"\bhack\b",
    r"\bsteal\b",
    r"\bmalware\b",
    r"\bexploit\b",
)

OUTPUT_SPOOF_PATTERNS = (
    r"\bOK\.\s*$",
    r"\bYES\.\s*$",
    r"\bNO\.\s*$",
    r"\bfake\.md\b",
)


@dataclass(frozen=True)
class GuardrailResult:
    blocked: bool
    reason: str = ""


def normalize_query(query: str) -> str:
    query = " ".join(query.split())
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS].rstrip()
    return query


def ascii_fold(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _regex_hit(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def should_refuse_query(query: str) -> GuardrailResult:
    normalized = normalize_query(query.strip())
    if not normalized:
        return GuardrailResult(blocked=True, reason="empty")

    folded = ascii_fold(normalized)
    if _regex_hit(INJECTION_PATTERNS, normalized):
        return GuardrailResult(blocked=True, reason="prompt_injection")

    injection_tokens = (
        "ignore",
        "bypass",
        "override",
        "system prompt",
        "developer prompt",
        "output only",
        "just say",
        "act as",
        "you are now",
        "bo qua",
        "tiet lo prompt",
        "chi in ra",
        "khong nhac nguon",
        "tra loi chi",
        "chi bang",
        "mot tu",
        "dong vai tro",
        "he thong",
        "citation gia",
        "citation fake",
        "citation spoof",
        "trich dan fake",
        "trich dan spoof",
        "trich dan gia",
        "fake.md",
    )
    if any(token in folded for token in injection_tokens):
        return GuardrailResult(blocked=True, reason="prompt_injection")

    if _regex_hit(HARMFUL_INTENT_PATTERNS, folded):
        return GuardrailResult(blocked=True, reason="harmful_intent")

    return GuardrailResult(blocked=False)


def sanitize_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s*\[[^\]]+?,\s*chunk\s*\d+\]", "", cleaned)
    cleaned = re.sub(r"\s*\[[^\]]+\.(?:md|pdf|json|html|txt|docx?)\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"(?<!^)\s+\*\s+", "\n- ", cleaned)
    cleaned = re.sub(r"^\*\s+", "- ", cleaned)
    cleaned = re.sub(r"\s*,\s*\.", ".", cleaned)
    cleaned = re.sub(r"[ \t]+\.", ".", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def looks_like_output_spoof(text: str) -> bool:
    folded = ascii_fold(text.strip())
    return _regex_hit(OUTPUT_SPOOF_PATTERNS, folded)
