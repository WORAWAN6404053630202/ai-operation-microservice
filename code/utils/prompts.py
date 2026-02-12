# code/utils/prompts.py
from utils.prompt_builder import get_system_prompt as _get_prompt_builder_prompt


def get_system_prompt(persona_id: str | None) -> str:
    pid = (persona_id or "academic").strip().lower()
    return _get_prompt_builder_prompt(pid)