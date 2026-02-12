# code/utils/persona_profile.py
"""
Persona Profile Utilities
Deterministic persona behavior configuration for production use
"""

from copy import deepcopy
from typing import Dict, Optional


ACADEMIC = "academic"
PRACTICAL = "practical"
DEFAULT_PERSONA = PRACTICAL


PERSONA_PROFILES: Dict[str, Dict] = {
    ACADEMIC: {
        "max_recent_messages": 18,
        "min_questions": 2,
        "ask_before_answer": False,
        "require_citations": True,
        "verbosity": "high",
        "allow_assumptions": False,
        "focus": "legal_accuracy",
        "strict_mode": True,
        "require_switch_confirmation": True,
    },
    PRACTICAL: {
        "max_recent_messages": 10,
        "min_questions": 1,
        "ask_before_answer": True,
        "require_citations": False,
        "verbosity": "low",
        "allow_assumptions": True,
        "focus": "actionable_guidance",
        "strict_mode": False,
        "require_switch_confirmation": True,
    },
}


PERSONA_SYSTEM_PROMPTS: Dict[str, str] = {
    ACADEMIC: """
คุณคือผู้ช่วย AI บุคลิก "Academic"

ลักษณะนิสัย:
- ตอบละเอียด เชิงลึก ครบถ้วน
- อธิบายทั้งภาพรวมและเชิงเทคนิค
- ไม่ถามซ้ำถ้าข้อมูลเพียงพอ
- ให้เหตุผล ประกอบ และ trade-off ชัดเจน

แนวทาง:
- เริ่มจากภาพรวม → ลงรายละเอียด
- ใช้โครงสร้างชัดเจน
- ความถูกต้องมาก่อนความสั้น

ข้อกำหนดสำคัญ:
- หากข้อมูลอยู่ในเอกสาร ให้ตอบโดยอ้างอิงจากเอกสารนั้น
- ห้ามปฏิเสธการตอบเพียงเพราะ “ไม่แน่ใจ” หากข้อมูลมีอยู่แล้ว
""".strip(),
    PRACTICAL: """
คุณคือผู้ช่วย AI บุคลิก "Practical"

ลักษณะนิสัย:
- ตอบสั้น กระชับ ใช้งานได้จริง
- โฟกัส action / checklist / step
- ข้ามทฤษฎีที่ไม่จำเป็น
- เปิดให้ผู้ใช้ถามต่อเพื่อขยาย

แนวทาง:
- bullet / step สั้น ๆ
- ตรงประเด็น
- ความเร็วและการใช้งานจริงมาก่อน

ข้อกำหนดสำคัญ:
- หากเอกสารมีข้อมูล ให้สรุปและตอบทันที
- ห้ามอ้างว่า “ไม่มีข้อมูล” หากเอกสารเกี่ยวข้องโดยตรง
""".strip(),
}


PERSONA_SWITCH_CONFIRMATION_PROMPTS: Dict[str, str] = {
    ACADEMIC: "ต้องการเปลี่ยนเป็นโหมด Academic จริง ๆ ใช่ไหม?",
    PRACTICAL: "ต้องการเปลี่ยนเป็นโหมด Practical จริง ๆ ใช่ไหม?",
}

PERSONA_SWITCH_SUCCESS_MESSAGES: Dict[str, str] = {
    ACADEMIC: "เปลี่ยนเป็นโหมด Academic แล้ว",
    PRACTICAL: "เปลี่ยนเป็นโหมด Practical แล้ว",
}


def normalize_persona_id(persona_id: Optional[str]) -> str:
    if not persona_id:
        return DEFAULT_PERSONA

    pid = str(persona_id).strip().lower()

    if pid in PERSONA_PROFILES:
        return pid

    alias_map = {
        "expert": ACADEMIC,
        "balanced": PRACTICAL,
        "minimal": PRACTICAL,
        "โหมดละเอียด": ACADEMIC,
        "โหมดสั้น": PRACTICAL,
    }

    return alias_map.get(pid, DEFAULT_PERSONA)


def build_strict_profile(
    persona_id: str,
    current: Optional[Dict] = None
) -> Dict:
    persona_id = normalize_persona_id(persona_id)
    base = PERSONA_PROFILES.get(persona_id, {})

    merged: Dict = {}
    if isinstance(current, dict):
        merged.update(deepcopy(current))

    for k, v in base.items():
        merged[k] = v

    return merged


def get_persona_system_prompt(persona_id: str) -> str:
    pid = normalize_persona_id(persona_id)
    return PERSONA_SYSTEM_PROMPTS.get(pid, PERSONA_SYSTEM_PROMPTS[DEFAULT_PERSONA])


def get_switch_confirmation_prompt(persona_id: str) -> str:
    pid = normalize_persona_id(persona_id)
    return PERSONA_SWITCH_CONFIRMATION_PROMPTS.get(pid)


def get_switch_success_message(persona_id: str) -> str:
    pid = normalize_persona_id(persona_id)
    return PERSONA_SWITCH_SUCCESS_MESSAGES.get(pid, "")


def apply_persona_profile(context: Dict, strict_profile: Dict) -> Dict:
    if not isinstance(context, dict):
        return {}

    ctx = deepcopy(context)
    ctx["persona_profile"] = {
        "effective": strict_profile,
        "persona_id": ctx.get("persona_id"),
    }
    return ctx
