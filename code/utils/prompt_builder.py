# code/utils/prompt_builder.py
"""
Prompt Builder (Single Source of Truth)
"""

from utils.persona_profile import normalize_persona_id

from utils.prompts_academic import SYSTEM_PROMPT as ACADEMIC_SYSTEM_PROMPT
from utils.prompts_practical import SYSTEM_PROMPT as PRACTICAL_SYSTEM_PROMPT


def get_system_prompt(persona_id: str) -> str:
    pid = normalize_persona_id(persona_id or "academic")
    if pid == "academic":
        return ACADEMIC_SYSTEM_PROMPT.strip()
    return PRACTICAL_SYSTEM_PROMPT.strip()
