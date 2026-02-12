# code/utils/context_manager.py
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str   # system | user | assistant
    content: str


class ContextManager:
    def __init__(
        self,
        max_history: int = 10,
        system_prefix: Optional[str] = None,
    ):
        self.max_history = max_history
        self.system_prefix = system_prefix or ""

    def build_context(
        self,
        persona_prompt: str,
        chat_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:

        messages: List[ChatMessage] = []

        if self.system_prefix.strip():
            messages.append(ChatMessage(role="system", content=self.system_prefix.strip()))

        messages.append(ChatMessage(role="system", content=persona_prompt.strip()))

        trimmed_history = self._trim_history(chat_history)

        for msg in trimmed_history:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

        return [{"role": m.role, "content": m.content} for m in messages]

    def _trim_history(
        self,
        chat_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:

        if not chat_history:
            return []

        filtered = [m for m in chat_history if m["role"] in ("user", "assistant")]
        return filtered[-self.max_history:]
    