# code/utils/prompt_builder.py
"""
Prompt Builder (Single Source of Truth)

Policy:
- The ONLY canonical system prompts live in:
  - utils/prompts_academic.py
  - utils/prompts_practical.py
- Any other module (including persona_profile) must NOT duplicate prompt strings.

This module intentionally does NOT import PERSONA_SYSTEM_PROMPTS (to avoid drift cycles).
"""

from __future__ import annotations

from utils.persona_profile import normalize_persona_id
from utils.prompts_academic import SYSTEM_PROMPT as ACADEMIC_SYSTEM_PROMPT
from utils.prompts_practical import SYSTEM_PROMPT as PRACTICAL_SYSTEM_PROMPT


def get_system_prompt(persona_id: str) -> str:
    """
    Return the canonical system prompt for the given persona.

    NOTE:
    - normalize_persona_id is used only for id normalization (NOT for prompt content).
    - Prompt content must come from prompts_academic/prompts_practical only.
    """
    pid = normalize_persona_id((persona_id or "practical").strip())
    if pid == "academic":
        return (ACADEMIC_SYSTEM_PROMPT or "").strip()
    return (PRACTICAL_SYSTEM_PROMPT or "").strip()