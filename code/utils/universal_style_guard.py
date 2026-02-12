'''
# code/utils/universal_style_guard.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass(frozen=True)
class UniversalStyleConfig:
    """
    Universal style bans for *all* personas in normal conversation.

    Quote rule:
    - Lines starting with '>' are treated as verbatim document quotes.
      We do NOT sanitize inside quote lines (verbatim fidelity).
    """
    forbidden_phrases: Tuple[str, ...] = (
        "เพื่อความถูกต้อง",
        "ขออนุญาต",
        "รบกวน",
        "ขอข้อมูลเพิ่ม",
        "ช่วยระบุให้ชัดเจน",
        "ต้องการเปลี่ยนเป็น",  # should appear only in persona-switch confirmation
    )

    # remove typical "preface" patterns at the beginning of a line
    forbidden_preface_patterns: Tuple[re.Pattern, ...] = (
        re.compile(r"^\s*(เพื่อความถูกต้อง|เพื่อให้แน่ใจ|เพื่อความชัดเจน)\b", re.IGNORECASE),
        re.compile(r"^\s*(ขออนุญาต|ขอสอบถาม|ขอถาม)\b", re.IGNORECASE),
        re.compile(r"^\s*(รบกวน|ขอความกรุณา)\b", re.IGNORECASE),
        re.compile(r"^\s*ขอข้อมูลเพิ่ม\b", re.IGNORECASE),
    )

    # practical constraints (used as a safety net)
    max_lines: int = 6
    max_bullets: int = 5
    max_chars: int = 900  # allow a bit for academic, but still guard runaways

    # question policy (used when action=ask)
    question_max_chars: int = 180

    # quote behavior
    allow_forbidden_inside_quotes: bool = True


DEFAULT_UNIVERSAL = UniversalStyleConfig()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _normalize(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\r\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def _split_lines_keep(text: str) -> List[str]:
    return [ln.rstrip() for ln in (text or "").split("\n") if ln.strip()]


def _is_quote_line(line: str) -> bool:
    return (line or "").lstrip().startswith(">")


def _count_bullets(lines: List[str]) -> int:
    return sum(1 for ln in lines if ln.lstrip().startswith(("-", "•", "*")))


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    for p in phrases:
        if p and p in text:
            return True
    return False


def _strip_preface(line: str, cfg: UniversalStyleConfig) -> str:
    s = line
    for pat in cfg.forbidden_preface_patterns:
        s = pat.sub("", s).strip()
    # common colon variants after preface
    s = re.sub(r"^\s*[:：\-–]\s*", "", s).strip()
    return s


def _remove_forbidden_inline(line: str, cfg: UniversalStyleConfig) -> str:
    s = line
    # remove inline occurrences conservatively
    for p in cfg.forbidden_phrases:
        if p:
            s = s.replace(p, "")
    # cleanup duplicated spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _enforce_length(text: str, cfg: UniversalStyleConfig) -> str:
    t = _normalize(text)
    lines = _split_lines_keep(t)

    # hard line cap
    lines = lines[: cfg.max_lines]

    # bullet cap
    out: List[str] = []
    bullet_count = 0
    for ln in lines:
        if ln.lstrip().startswith(("-", "•", "*")):
            bullet_count += 1
            if bullet_count > cfg.max_bullets:
                continue
        out.append(ln)

    t2 = "\n".join(out).strip()

    # char cap
    if len(t2) > cfg.max_chars:
        t2 = t2[: cfg.max_chars].rstrip()

    return t2.strip()


def sanitize_text_universal(
    text: str,
    cfg: UniversalStyleConfig = DEFAULT_UNIVERSAL,
    *,
    treat_as_question: bool = False,
    bypass: bool = False,
) -> str:
    """
    Deterministic universal sanitizer:
    - Remove forbidden prefaces and forbidden phrases from non-quote lines
    - Preserve quote lines verbatim (if allow_forbidden_inside_quotes)
    - Enforce basic brevity constraints
    """
    if bypass:
        return _enforce_length(text, cfg)

    t = _normalize(text)
    if not t:
        return t

    lines = _split_lines_keep(t)
    cleaned: List[str] = []

    for ln in lines:
        if cfg.allow_forbidden_inside_quotes and _is_quote_line(ln):
            cleaned.append(ln)  # verbatim
            continue

        s = _strip_preface(ln, cfg)
        s = _remove_forbidden_inline(s, cfg)

        # if line becomes empty after stripping, drop it
        if s:
            cleaned.append(s)

    out = "\n".join(cleaned).strip()
    out = _enforce_length(out, cfg)

    if treat_as_question:
        q = out.split("\n")[0].strip() if out else ""
        if len(q) > cfg.question_max_chars:
            q = q[: cfg.question_max_chars].rstrip()

        # Ensure question mark for Thai if not already question-like
        if q and not (q.endswith("?") or q.endswith("ไหม") or q.endswith("หรือไม่") or q.endswith("หรือเปล่า")):
            q = q.rstrip(".") + "?"
        out = q

    return out.strip()
'''
