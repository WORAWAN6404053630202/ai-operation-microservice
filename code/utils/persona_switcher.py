# code/utils/persona_switcher.py
"""
Production-grade Persona Switcher
=================================
- 2 personas only: academic, practical
- Conservative, intent-first switching
- Deterministic first, LLM fallback as last resort
- Safe for real product usage

UPDATED:
- Better support for implicit style-switch requests like:
  "ขอแบบละเอียด", "ขอแบบสั้น", "เอาแบบกระชับ", "ขอเชิงลึก"
  even when user does NOT mention "โหมด/persona"
- Still conservative: only triggers when message is dominantly a style request
  OR explicit switch intent exists.
"""

import re
from difflib import SequenceMatcher
from typing import Optional, Tuple, Dict, Callable, Any

# ---------------------------------------------------------------------
# Canonical personas
# ---------------------------------------------------------------------
PERSONA_CANONICAL = {"academic", "practical"}

# ---------------------------------------------------------------------
# Aliases (Thai + English)
# Conservative on purpose to avoid accidental switching
# ---------------------------------------------------------------------
ALIASES = {
    # English
    "academic": "academic",
    "practical": "practical",
    "acad": "academic",
    "prac": "practical",

    # Thai (explicit)
    "วิชาการ": "academic",
    "เชิงลึก": "academic",
    "ละเอียด": "academic",
    "ทางการ": "academic",

    "กระชับ": "practical",
    "สั้น": "practical",
    "เร็ว": "practical",
    "ใช้งานจริง": "practical",
    "เอาไปใช้จริง": "practical",
}

# ---------------------------------------------------------------------
# Switch intent markers
# ---------------------------------------------------------------------
_SWITCH_MARKERS = (
    "persona",
    "mode",
    "โหมด",
    "บุคลิก",
)

_SWITCH_VERBS = (
    "เปลี่ยน",
    "สลับ",
    "ปรับเป็น",
    "ขอเป็น",
    "ใช้โหมด",
    "เปลี่ยนโหมด",
    "สลับโหมด",
)

# Slash command: /persona <token>
_CMD_RE = re.compile(r"(?:^|\s)/persona\s+([^\s]+)", flags=re.IGNORECASE)

# ---------------------------------------------------------------------
# NEW: implicit style-switch heuristics (light + conservative)
# ---------------------------------------------------------------------
_STYLE_REQ_RE = re.compile(
    r"(?:^|\s)(ขอ|เอา|ขอเป็น|เอาเป็น|ขอแบบ|เอาแบบ|ขอสไตล์|ปรับ|ช่วยปรับ)\s*(?:ให้)?\s*"
    r"(ละเอียด|เชิงลึก|วิชาการ|ทางการ|สั้น|กระชับ|เร็ว|สรุป|สรุปสั้น)\b",
    flags=re.IGNORECASE,
)

# If message is basically "style request only", allow switch without explicit markers
_STYLE_ONLY_GUARD_RE = re.compile(
    r"^(?:\s*(ขอ|เอา|ขอเป็น|เอาเป็น|ขอแบบ|เอาแบบ|ช่วย|ปรับ)\s*)"
    r"(ละเอียด|เชิงลึก|วิชาการ|ทางการ|สั้น|กระชับ|เร็ว|สรุป|สรุปสั้น)"
    r"(?:\s*ๆ+|\s*)$",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _norm(token: str) -> str:
    return re.sub(r"[\s\-_]+", "", (token or "").strip().lower())


def _fuzzy_persona(token: str, threshold: float = 0.84) -> Optional[str]:
    """
    Conservative fuzzy matching:
    - exact canonical
    - exact alias
    - fuzzy match with high threshold
    """
    t = _norm(token)
    if not t:
        return None

    if t in PERSONA_CANONICAL:
        return t

    for k, v in ALIASES.items():
        if _norm(k) == t:
            return v

    best_key, best_score = None, 0.0
    for k in list(PERSONA_CANONICAL) + list(ALIASES.keys()):
        score = _similar(t, _norm(k))
        if score > best_score:
            best_key, best_score = k, score

    if best_score < threshold:
        return None

    return ALIASES.get(best_key, best_key)


def _has_switch_intent(text: str) -> bool:
    """
    Detect explicit intent to switch persona.
    """
    t = (text or "").lower()
    if not t:
        return False

    if _CMD_RE.search(t):
        return True

    if any(m in t for m in _SWITCH_MARKERS):
        return True

    if any(v in t for v in _SWITCH_VERBS):
        return True

    return False


def _extract_token(text: str) -> Optional[str]:
    """
    Extract persona token after explicit marker or verb.
    """
    m = re.search(
        r"(?:persona|mode|โหมด|บุคลิก)\s*[:\-]?\s*([^\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    m = re.search(
        r"(?:ขอเป็น|เปลี่ยนเป็น|ปรับเป็น)\s*([^\s]+)",
        text,
    )
    if m:
        return m.group(1).strip()

    return None


def _clean_switch_phrase(text: str) -> str:
    """
    Remove only the switch phrase, keep the rest intact.
    """
    raw = text or ""

    raw = _CMD_RE.sub("", raw, count=1)

    raw = re.sub(
        r"(?:persona|mode|โหมด|บุคลิก)\s*[:\-]?\s*[^\s]+\b",
        "",
        raw,
        count=1,
        flags=re.IGNORECASE,
    )

    raw = re.sub(
        r"(?:ขอเป็น|เปลี่ยนเป็น|ปรับเป็น)\s*[^\s]+\b",
        "",
        raw,
        count=1,
    )

    return raw.strip()


def _implicit_style_switch_det(text: str) -> Tuple[Optional[str], float]:
    """
    Returns (persona_id_or_none, confidence).
    Conservative:
    - if message is style-only -> high confidence
    - if message contains a clear style request phrase -> medium confidence
    """
    t = (text or "").strip()
    if not t:
        return None, 0.0

    # style-only guard (very safe)
    m = _STYLE_ONLY_GUARD_RE.match(t)
    if m:
        style = (m.group(2) or "").strip().lower()
        pid = _fuzzy_persona(style)  # maps "ละเอียด"->academic, "สั้น"->practical
        if pid:
            return pid, 0.92
        return None, 0.0

    # style request inside text (still conservative)
    m2 = _STYLE_REQ_RE.search(t)
    if m2:
        style = (m2.group(2) or "").strip().lower()
        pid = _fuzzy_persona(style)
        if pid:
            return pid, 0.78
        return None, 0.0

    return None, 0.0


# ---------------------------------------------------------------------
# Deterministic parser
# ---------------------------------------------------------------------
def parse_persona_switch_det(
    user_text: str,
) -> Tuple[Optional[str], str, Dict]:
    """
    Deterministic + conservative parsing.
    """
    text = (user_text or "").strip()
    if not text:
        return None, user_text, {
            "method": None,
            "confidence": 0.0,
            "should_confirm": False,
            "needs_llm_fallback": False,
        }

    # 0) Implicit style-switch (no explicit markers)
    # Only triggers when confidence is high enough.
    pid2, conf2 = _implicit_style_switch_det(text)
    if pid2 and conf2 >= 0.90:
        return pid2, _clean_switch_phrase(text), {
            "method": "implicit_style_only",
            "confidence": conf2,
            "should_confirm": True,
            "needs_llm_fallback": False,
        }

    # 1) Slash command
    m = _CMD_RE.search(text.lower())
    if m:
        token = m.group(1)
        persona = _fuzzy_persona(token)
        if persona:
            return persona, _clean_switch_phrase(text), {
                "method": "command",
                "confidence": 1.0,
                "should_confirm": True,
                "needs_llm_fallback": False,
            }
        return None, user_text, {
            "method": "command_unparsed",
            "confidence": 0.0,
            "should_confirm": False,
            "needs_llm_fallback": True,
        }

    # 2) No explicit switch intent -> allow medium implicit style request to go LLM fallback
    if not _has_switch_intent(text):
        pid3, conf3 = _implicit_style_switch_det(text)
        if pid3 and conf3 >= 0.75:
            return None, user_text, {
                "method": "implicit_style_needs_confirm",
                "confidence": conf3,
                "should_confirm": True,
                "needs_llm_fallback": True,  # let LLM confirm (smarter)
                "style_hint": pid3,
            }
        return None, user_text, {
            "method": None,
            "confidence": 0.0,
            "should_confirm": False,
            "needs_llm_fallback": False,
        }

    # 3) Marker / verb based
    token = _extract_token(text)
    if token:
        persona = _fuzzy_persona(token)
        if persona:
            return persona, _clean_switch_phrase(text), {
                "method": "marker_or_verb",
                "confidence": 0.93,
                "should_confirm": True,
                "needs_llm_fallback": False,
            }

    # 4) Weak alias (still intent-present)
    lower = text.lower()
    for k, v in ALIASES.items():
        if k in lower:
            return v, user_text, {
                "method": "alias_weak",
                "confidence": 0.80,
                "should_confirm": True,
                "needs_llm_fallback": True,
            }

    return None, user_text, {
        "method": "intent_unparsed",
        "confidence": 0.0,
        "should_confirm": False,
        "needs_llm_fallback": True,
    }


# ---------------------------------------------------------------------
# LLM fallback (last resort)
# ---------------------------------------------------------------------
def classify_persona_with_llm(
    llm_call_json: Callable[[str], Any],
    user_text: str,
    style_hint: Optional[str] = None,
) -> Dict:
    """
    LLM must be conservative but smart:
    - If user explicitly asks for "ละเอียด/เชิงลึก/วิชาการ" => switch=true academic
    - If user explicitly asks for "สั้น/กระชับ/สรุป" => switch=true practical
    - If user just asks to summarize content without implying persona change => switch=false
    """
    hint_line = f'\nstyle_hint_from_det: "{style_hint}"\n' if style_hint else "\n"

    prompt = f"""
คุณมีหน้าที่ตัดสินว่า "ผู้ใช้ตั้งใจสลับ persona หรือไม่"

persona ที่อนุญาต:
- academic (ละเอียด/เชิงลึก/วิชาการ)
- practical (สั้น/กระชับ/เอาไปใช้จริง)

กติกา (สำคัญ):
- ถ้าผู้ใช้ "ขอแบบละเอียด/เชิงลึก/วิชาการ" ให้ถือว่าเป็นการขอสลับไป academic ได้ แม้ไม่ได้พูดคำว่าโหมด
- ถ้าผู้ใช้ "ขอแบบสั้น/กระชับ/สรุปสั้น" ให้ถือว่าเป็นการขอสลับไป practical ได้ แม้ไม่ได้พูดคำว่าโหมด
- ถ้าผู้ใช้แค่ขอ "สรุป" ของคำตอบเดิม โดยไม่ได้สื่อว่าอยากเปลี่ยนสไตล์ต่อเนื่อง ให้ switch=false
- ถ้าไม่มั่นใจ ให้ switch=false
- ห้ามเดา

ตอบเป็น JSON เท่านั้น:
{{
  "switch": true/false,
  "persona_id": "academic|practical|",
  "confidence": 0.0
}}

{hint_line}
ข้อความผู้ใช้:
{user_text}
""".strip()

    out = llm_call_json(prompt)
    if not isinstance(out, dict):
        return {"switch": False, "persona_id": None, "confidence": 0.0}

    pid = (out.get("persona_id") or "").lower().strip()
    try:
        conf = float(out.get("confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0

    if pid not in PERSONA_CANONICAL:
        pid = None

    return {
        "switch": bool(out.get("switch", False)),
        "persona_id": pid,
        "confidence": conf,
    }


# ---------------------------------------------------------------------
# Final resolver (public API)
# ---------------------------------------------------------------------
def resolve_persona_switch(
    user_text: str,
    llm_call_json: Optional[Callable[[str], Any]] = None,
    llm_conf_threshold: float = 0.85,
) -> Tuple[Optional[str], str, Dict]:
    """
    Final resolver used by the system.
    """
    persona, cleaned, meta = parse_persona_switch_det(user_text)
    if persona:
        return persona, cleaned, meta

    # If det found a style hint but asked for LLM verification:
    style_hint = meta.get("style_hint") if isinstance(meta, dict) else None

    if meta.get("needs_llm_fallback") and llm_call_json:
        res = classify_persona_with_llm(llm_call_json, user_text, style_hint=style_hint)

        if (
            res.get("switch")
            and res.get("persona_id")
            and res.get("confidence", 0.0) >= llm_conf_threshold
        ):
            return res["persona_id"], user_text, {
                "method": "llm_fallback",
                "confidence": res["confidence"],
                "should_confirm": True,
                "needs_llm_fallback": False,
            }

        return None, user_text, {
            "method": "llm_fallback_rejected",
            "confidence": res.get("confidence", 0.0),
            "should_confirm": False,
            "needs_llm_fallback": False,
        }

    return None, user_text, meta
