# code/utils/persona_switcher.py
"""
Enterprise Persona Switcher (Explicit-Intent Only)
==================================================
- 2 personas only: academic, practical
- Switch ONLY when user has explicit switch intent (marker/verb or /persona command)
- NO style-only inference here (Supervisor owns "propose switch" + confirm policy)
- Deterministic parsing first; NO LLM fallback unless explicit switch intent exists
"""

import re
from difflib import SequenceMatcher
from typing import Optional, Tuple, Dict, Callable, Any

PERSONA_CANONICAL = {"academic", "practical"}

ALIASES = {
    "academic": "academic",
    "practical": "practical",
    "acad": "academic",
    "prac": "practical",

    # Thai aliases
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

_SWITCH_MARKERS = ("persona", "mode", "โหมด", "บุคลิก")
_SWITCH_VERBS = ("เปลี่ยน", "สลับ", "ปรับเป็น", "ขอเป็น", "ใช้โหมด", "เปลี่ยนโหมด", "สลับโหมด")

_CMD_RE = re.compile(r"(?:^|\s)/persona\s+([^\s]+)", flags=re.IGNORECASE)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _norm(token: str) -> str:
    return re.sub(r"[\s\-_]+", "", (token or "").strip().lower())


def _fuzzy_persona(token: str, threshold: float = 0.84) -> Optional[str]:
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
    Explicit switch intent only:
    - /persona ...
    - has switch markers (persona/mode/โหมด/บุคลิก)
    - or switch verbs (เปลี่ยน/สลับ/ปรับเป็น/ขอเป็น/ใช้โหมด/เปลี่ยนโหมด/สลับโหมด)
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
    # marker: "โหมด: academic" / "persona practical"
    m = re.search(
        r"(?:persona|mode|โหมด|บุคลิก)\s*[:\-]?\s*([^\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # verb: "ขอเป็น practical" / "เปลี่ยนเป็น academic"
    m = re.search(r"(?:ขอเป็น|เปลี่ยนเป็น|ปรับเป็น)\s*([^\s]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def _clean_switch_phrase(text: str) -> str:
    """
    Remove only the explicit switch command/phrase, keep the rest (if any).
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
    raw = re.sub(r"(?:ขอเป็น|เปลี่ยนเป็น|ปรับเป็น)\s*[^\s]+\b", "", raw, count=1, flags=re.IGNORECASE)
    return raw.strip()


# -------------------------------------------------------------------
# Deterministic explicit parser (NO style-only inference)
# -------------------------------------------------------------------

def parse_persona_switch_det(user_text: str) -> Tuple[Optional[str], str, Dict]:
    """
    Returns:
      (persona_id_or_none, cleaned_text, meta)

    Enterprise rules:
    - If no explicit switch intent => NEVER switch, NEVER ask LLM here.
    - If explicit intent exists but token unparsed => meta.needs_llm_fallback=True
      (Supervisor MAY decide whether to call LLM, but recommended to confirm with user instead).
    """
    text = (user_text or "").strip()
    if not text:
        return None, user_text, {
            "method": None,
            "confidence": 0.0,
            "should_confirm": False,
            "needs_llm_fallback": False,
        }

    # 1) Slash command: /persona academic
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

    # 2) No explicit intent => no switching, no LLM
    if not _has_switch_intent(text):
        return None, user_text, {
            "method": None,
            "confidence": 0.0,
            "should_confirm": False,
            "needs_llm_fallback": False,
        }

    # 3) Marker / verb based token extraction
    token = _extract_token(text)
    if token:
        persona = _fuzzy_persona(token)
        if persona:
            return persona, _clean_switch_phrase(text), {
                "method": "marker_or_verb",
                "confidence": 0.95,
                "should_confirm": True,
                "needs_llm_fallback": False,
            }
        return None, user_text, {
            "method": "token_unparsed",
            "confidence": 0.0,
            "should_confirm": False,
            "needs_llm_fallback": True,
        }

    # 4) As a last deterministic attempt: alias presence, BUT ONLY because intent markers/verbs exist
    lower = text.lower()
    for k, v in ALIASES.items():
        if k.lower() in lower:
            return v, _clean_switch_phrase(text), {
                "method": "alias_with_intent",
                "confidence": 0.88,
                "should_confirm": True,
                "needs_llm_fallback": False,
            }

    # Intent exists but cannot parse persona token => allow optional LLM fallback
    return None, user_text, {
        "method": "intent_unparsed",
        "confidence": 0.0,
        "should_confirm": False,
        "needs_llm_fallback": True,
    }


# -------------------------------------------------------------------
# Optional LLM classifier (ONLY allowed when explicit switch intent exists)
# -------------------------------------------------------------------

def classify_persona_with_llm(
    llm_call_json: Callable[[str], Any],
    user_text: str,
) -> Dict:
    prompt = f"""
คุณมีหน้าที่ตัดสินว่า "ผู้ใช้ตั้งใจสลับ persona หรือไม่" โดยพิจารณาเฉพาะกรณีที่มีสัญญาณชัดเจนว่า "เปลี่ยน/สลับ/โหมด/persona/mode"

persona ที่อนุญาต:
- academic
- practical

กติกา:
- ถ้าไม่ชัดเจน ให้ switch=false
- ห้ามตีความคำขอสไตล์อย่างเดียว (เช่น "สั้นๆ" / "ละเอียดหน่อย") ว่าเป็นการสลับ persona ในโมดูลนี้
- ต้องอิงจากข้อความที่เป็นการ "เปลี่ยน/สลับ/โหมด/persona/mode" จริงๆ

ตอบเป็น JSON เท่านั้น:
{{
  "switch": true/false,
  "persona_id": "academic|practical|",
  "confidence": 0.0
}}

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


def resolve_persona_switch(
    user_text: str,
    llm_call_json: Optional[Callable[[str], Any]] = None,
    llm_conf_threshold: float = 0.90,
) -> Tuple[Optional[str], str, Dict]:
    """
    Enterprise resolution:
    - Deterministic explicit parsing only.
    - LLM fallback is allowed ONLY if explicit switch intent exists AND token cannot be parsed.
    - If no explicit intent => do not call LLM, return None.
    """
    persona, cleaned, meta = parse_persona_switch_det(user_text)
    if persona:
        return persona, cleaned, meta

    # No explicit intent => never LLM
    if not _has_switch_intent(user_text or ""):
        return None, user_text, meta

    # Explicit intent exists, but unparsed => optional LLM fallback (still conservative)
    if meta.get("needs_llm_fallback") and llm_call_json:
        res = classify_persona_with_llm(llm_call_json, user_text)

        if (
            res.get("switch")
            and res.get("persona_id")
            and res.get("confidence", 0.0) >= llm_conf_threshold
        ):
            return res["persona_id"], cleaned, {
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
