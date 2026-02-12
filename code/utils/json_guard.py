# code/utils/json_guard.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple, Callable


ALLOWED_ACTIONS = {"retrieve", "ask", "answer"}
ALLOWED_INPUT_TYPES = {"greeting", "new_question", "follow_up"}


def strip_code_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers if present.
    """
    t = (text or "").strip()
    if "```" not in t:
        return t

    # prefer ```json
    m = re.search(r"```json\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"```\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    return t


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Best-effort extraction of the first JSON object from a string.
    Useful when model returns extra prose around JSON.
    """
    t = (text or "").strip()
    if not t:
        return None

    # quick path: starts with {
    if t.lstrip().startswith("{") and t.rstrip().endswith("}"):
        return t

    # scan for first balanced {...}
    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1].strip()

    return None


def parse_json_safely(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (parsed_json_or_none, cleaned_text_used_for_parsing)
    """
    cleaned = strip_code_fences(text)
    candidate = extract_first_json_object(cleaned) or cleaned
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj, candidate
        return None, candidate
    except Exception:
        return None, candidate


def validate_agent_json(obj: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Lightweight schema validation for your agent contract.
    """
    if not isinstance(obj, dict):
        return False, "not_a_dict"

    # required top-level keys
    for k in ("input_type", "analysis", "action", "execution"):
        if k not in obj:
            return False, f"missing_top_level:{k}"

    if not isinstance(obj.get("execution"), dict):
        return False, "execution_not_object"

    action = obj.get("action")
    if action not in ALLOWED_ACTIONS:
        return False, f"invalid_action:{action}"

    it = obj.get("input_type")
    if it is not None and isinstance(it, str):
        if it not in ALLOWED_INPUT_TYPES:
            # not fatal, but keep strict to stabilize downstream
            return False, f"invalid_input_type:{it}"
    else:
        return False, "input_type_invalid"

    # Ensure execution has the expected field for the action
    ex = obj.get("execution") or {}
    if action == "retrieve" and not isinstance(ex.get("query", ""), str):
        return False, "retrieve_missing_query"
    if action == "ask" and not isinstance(ex.get("question", ""), str):
        return False, "ask_missing_question"
    if action == "answer" and not isinstance(ex.get("answer", ""), str):
        return False, "answer_missing_answer"

    return True, "ok"


def build_repair_prompt(
    raw_text: str,
    schema_reminder: str,
    error_reason: str,
) -> str:
    """
    Prompt the LLM to repair invalid output into valid JSON only.
    """
    return f"""
You produced an invalid response. Fix it.

ERROR REASON:
{error_reason}

RULES:
- Output MUST be valid JSON only.
- No markdown. No extra text. No explanations.
- Must match this schema exactly:
{schema_reminder}

INVALID OUTPUT:
{raw_text}

Return the corrected JSON only:
""".strip()


DEFAULT_SCHEMA_REMINDER = """
{
  "input_type": "greeting | new_question | follow_up",
  "analysis": "",
  "action": "retrieve | ask | answer",
  "execution": {
    "query": "",
    "question": "",
    "answer": "",
    "context_update": {}
  }
}
""".strip()


def call_llm_json_with_repair(
    llm_invoke_text: Callable[[str], str],
    prompt: str,
    max_attempts: int = 2,
    enable_repair: bool = True,
    schema_reminder: str = DEFAULT_SCHEMA_REMINDER,
) -> Dict[str, Any]:
    """
    1) call LLM
    2) parse + validate
    3) if fail and enable_repair -> 1 repair attempt
    4) fallback ask
    """
    # attempt 1: original
    raw1 = llm_invoke_text(prompt)
    obj1, used1 = parse_json_safely(raw1)
    if obj1:
        ok, reason = validate_agent_json(obj1)
        if ok:
            return obj1
    else:
        reason = "json_parse_failed"

    if not enable_repair or max_attempts <= 1:
        return {
            "input_type": "follow_up",
            "analysis": f"fallback_due_to:{reason}",
            "action": "ask",
            "execution": {"question": "ขออภัยค่ะ ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้งได้ไหมคะ", "context_update": {}},
        }

    # attempt 2: repair
    repair_prompt = build_repair_prompt(
        raw_text=used1 if isinstance(used1, str) else str(raw1),
        schema_reminder=schema_reminder,
        error_reason=reason,
    )
    raw2 = llm_invoke_text(repair_prompt)
    obj2, _used2 = parse_json_safely(raw2)
    if obj2:
        ok2, reason2 = validate_agent_json(obj2)
        if ok2:
            return obj2
        reason = f"repair_invalid:{reason2}"
    else:
        reason = "repair_parse_failed"

    # final fallback
    return {
        "input_type": "follow_up",
        "analysis": f"fallback_due_to:{reason}",
        "action": "ask",
        "execution": {"question": "ขออภัยค่ะ ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้งได้ไหมคะ", "context_update": {}},
    }
