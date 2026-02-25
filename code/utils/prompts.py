# code/utils/prompts.py
"""
Compatibility wrapper (do not add prompt strings here).

This module exists so older imports (utils.prompts.get_system_prompt)
continue to work, while keeping a single source of truth in prompt_builder.
"""

from __future__ import annotations

from utils.prompt_builder import get_system_prompt as _get_prompt_builder_prompt


def get_system_prompt(persona_id: str | None) -> str:
    pid = (persona_id or "practical").strip().lower()
    return _get_prompt_builder_prompt(pid)