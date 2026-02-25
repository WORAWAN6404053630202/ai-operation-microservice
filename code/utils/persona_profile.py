# code/utils/persona_profile.py
"""
Persona Profile Utilities
Deterministic persona behavior configuration for production use

IMPORTANT:
- This module MUST NOT contain any system prompt text.
- Canonical prompts live only in:
  - utils/prompts_academic.py
  - utils/prompts_practical.py
- This module is "policy knobs" + persona id normalization + switch UX strings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


ACADEMIC = "academic"
PRACTICAL = "practical"
DEFAULT_PERSONA = PRACTICAL


# -------------------------------------------------------------------
# Persona policy knobs (config only; no prompt content)
# -------------------------------------------------------------------
PERSONA_PROFILES: Dict[str, Dict[str, Any]] = {
    ACADEMIC: {
        # history / context
        "max_recent_messages": 18,
        # behavior knobs
        "ask_before_answer": False,
        "require_citations": True,
        "verbosity": "high",
        "allow_assumptions": False,
        "focus": "legal_accuracy",
        "strict_mode": True,
        # switching policy
        "require_switch_confirmation": True,
    },
    PRACTICAL: {
        "max_recent_messages": 10,
        "ask_before_answer": True,
        "require_citations": False,
        "verbosity": "low",
        "allow_assumptions": True,
        "focus": "actionable_guidance",
        "strict_mode": False,
        "require_switch_confirmation": True,
    },
}


# -------------------------------------------------------------------
# Switch UX strings (still config, not prompts)
# -------------------------------------------------------------------
PERSONA_SWITCH_CONFIRMATION_PROMPTS: Dict[str, str] = {
    ACADEMIC: "ต้องการเปลี่ยนเป็นโหมด Academic จริง ๆ ใช่ไหม?",
    PRACTICAL: "ต้องการเปลี่ยนเป็นโหมด Practical จริง ๆ ใช่ไหม?",
}

PERSONA_SWITCH_SUCCESS_MESSAGES: Dict[str, str] = {
    ACADEMIC: "เปลี่ยนเป็นโหมด Academic แล้ว",
    PRACTICAL: "เปลี่ยนเป็นโหมด Practical แล้ว",
}


# -------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------
def normalize_persona_id(persona_id: Optional[str]) -> str:
    """
    Normalize persona id into {academic, practical}.
    Keep aliases here ONLY for id mapping (no behavior text).
    """
    if not persona_id:
        return DEFAULT_PERSONA

    pid = str(persona_id).strip().lower()

    if pid in PERSONA_PROFILES:
        return pid

    alias_map = {
        "acad": ACADEMIC,
        "prac": PRACTICAL,
        "expert": ACADEMIC,
        "balanced": PRACTICAL,
        "minimal": PRACTICAL,
        "วิชาการ": ACADEMIC,
        "เชิงลึก": ACADEMIC,
        "ละเอียด": ACADEMIC,
        "ทางการ": ACADEMIC,
        "สั้น": PRACTICAL,
        "กระชับ": PRACTICAL,
        "เร็ว": PRACTICAL,
        "โหมดละเอียด": ACADEMIC,
        "โหมดสั้น": PRACTICAL,
    }

    return alias_map.get(pid, DEFAULT_PERSONA)


# -------------------------------------------------------------------
# Strict profile builder
# -------------------------------------------------------------------
def build_strict_profile(persona_id: str, current: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge persona baseline policy into current strict_profile (if any).
    Baseline always wins for keys it defines.
    """
    pid = normalize_persona_id(persona_id)
    base = PERSONA_PROFILES.get(pid, {})

    merged: Dict[str, Any] = {}
    if isinstance(current, dict):
        merged.update(deepcopy(current))

    for k, v in base.items():
        merged[k] = v

    return merged


# -------------------------------------------------------------------
# Switch UX helpers
# -------------------------------------------------------------------
def get_switch_confirmation_prompt(persona_id: str) -> str:
    pid = normalize_persona_id(persona_id)
    return PERSONA_SWITCH_CONFIRMATION_PROMPTS.get(pid, "")


def get_switch_success_message(persona_id: str) -> str:
    pid = normalize_persona_id(persona_id)
    return PERSONA_SWITCH_SUCCESS_MESSAGES.get(pid, "")


# -------------------------------------------------------------------
# Apply into state.context (supervisor owns when to call this)
# -------------------------------------------------------------------
def apply_persona_profile(context: Dict[str, Any], strict_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store effective persona policy into context in a structured way.
    This should be the only "write" this module does.
    """
    if not isinstance(context, dict):
        return {}

    ctx = deepcopy(context)
    ctx["persona_profile"] = {
        "effective": deepcopy(strict_profile) if isinstance(strict_profile, dict) else {},
        "persona_id": ctx.get("persona_id"),
    }
    return ctx
