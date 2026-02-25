# code/utils/context_manager.py
from __future__ import annotations

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str   # system | user | assistant
    content: str


class ContextManager:
    """
    Context builder for LLM calls.

    Production guarantees:
    - Trimming affects ONLY user/assistant chat turns (never touches state.context).
    - Preserve "pending/intake" continuity by keeping the most recent turns and
      biasing to include the last assistant message (menus/questions).
    - Provide a shared system_prefix for cross-persona invariants to reduce drift.

    NOTE:
    - Persona services MUST store intake/slot/stage in state.context, not rely on old history.
      This component intentionally does not try to reconstruct flows from history.
    """

    # Shared invariants (safe defaults). You can override by passing system_prefix in __init__.
    _DEFAULT_SYSTEM_PREFIX = """
กติกากลางของระบบ (ห้ามผิด):
- ตอบภาษาไทย 100%
- ห้ามสมมติข้อมูลนอกเอกสาร/นอก DOCUMENTS
- ถ้าไม่มีข้อมูลในเอกสาร ให้บอกว่า "ไม่พบในเอกสาร" อย่างชัดเจน
- ห้ามเปิดเผยรหัส/ตัวระบุภายใน เช่น row_id, doc_id, uuid, chroma id, ชื่อคอลเลกชัน, ชื่อไฟล์, จำนวนแถว/จำนวนเอกสาร
- ห้ามพูดถึงขั้นตอน/โครงสร้างภายในของระบบ (เช่น chunk, vector store, retriever) ในคำตอบผู้ใช้
""".strip()

    def __init__(
        self,
        max_history: int = 10,
        system_prefix: Optional[str] = None,
        include_default_system_prefix: bool = True,
    ):
        self.max_history = int(max_history or 0)
        base = (system_prefix or "").strip()

        if include_default_system_prefix:
            if base:
                self.system_prefix = (self._DEFAULT_SYSTEM_PREFIX + "\n\n" + base).strip()
            else:
                self.system_prefix = self._DEFAULT_SYSTEM_PREFIX
        else:
            self.system_prefix = base

    def build_context(
        self,
        persona_prompt: str,
        chat_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        messages: List[ChatMessage] = []

        sp = (self.system_prefix or "").strip()
        if sp:
            messages.append(ChatMessage(role="system", content=sp))

        pp = (persona_prompt or "").strip()
        if pp:
            messages.append(ChatMessage(role="system", content=pp))

        trimmed_history = self._trim_history(chat_history)

        for msg in trimmed_history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append(ChatMessage(role=role, content=content))

        return [{"role": m.role, "content": m.content} for m in messages]

    # --------------------------
    # Trimming strategy
    # --------------------------
    def _trim_history(self, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Keep only last N user/assistant messages, but avoid breaking short-lived
        selection flows by ensuring:
        - We keep the last assistant message if it exists (menus/questions).
        - If the last message is user, we also keep the immediate previous assistant
          (so the model sees the prompt/menu the user responded to).

        This DOES NOT guarantee any "flow state" — that must live in state.context.
        """
        if not chat_history or self.max_history <= 0:
            return []

        filtered = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in chat_history
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m.get("content").strip()
        ]
        if not filtered:
            return []

        # Basic tail trim
        tail = filtered[-self.max_history :]

        # Ensure last assistant context is present for menu/question binding
        last_role = filtered[-1]["role"]

        if last_role == "user":
            # If user just replied, include the assistant right before it if it was trimmed out
            if len(filtered) >= 2 and filtered[-2]["role"] == "assistant":
                prev_assistant = filtered[-2]
                if prev_assistant not in tail:
                    # make room
                    tail = ([prev_assistant] + tail)[-self.max_history :]
        else:
            # last is assistant; ensure it exists (should already), but be defensive
            last_assistant = filtered[-1]
            if last_assistant not in tail:
                tail = (tail + [last_assistant])[-self.max_history :]

        return tail