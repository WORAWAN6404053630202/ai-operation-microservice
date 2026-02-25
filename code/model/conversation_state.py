# code/model/conversation_state.py
"""
Conversation State Model (v2 Final)

Single source of truth for conversation lifecycle.
This model is intentionally logic-light and framework-agnostic.

Production stability improvements:
- ✅ Single source-of-truth for retrieval tracking: state.last_retrieval_query (+ optional cached mirror in context)
- ✅ Explicit conversation locks in context (supervisor-level policy can rely on these)
- ✅ Append-only + dedupe helpers: add_user_message_once / add_assistant_message_once
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class ConversationState(BaseModel):
    """
    Maintains full conversation state, including:

    - Session identity
    - Persona & behavior configuration
    - User-visible conversation history
    - Internal traces (not shown to users)
    - Context memory (slots / facts / flags)
    - Retrieved documents (RAG)
    - Multi-step round tracking
    """

    # allow forward compatibility
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    session_id: str = Field(default="", description="Conversation session identifier")

    # ------------------------------------------------------------------
    # Persona & behavior
    # ------------------------------------------------------------------

    persona_id: str = Field(
        default="practical",
        description="Active persona id (academic / practical)",
    )

    strict_profile: Dict[str, Any] = Field(
        default_factory=lambda: {
            "ask_before_answer": True,
            "require_citations": True,
            "max_recent_messages": 18,
            "verbosity": "high",
            "strict_mode": True,
        },
        description="Effective behavior knobs derived from persona",
    )

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="User-visible chat history",
    )

    internal_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Internal system / agent traces (hidden from user)",
    )

    # ------------------------------------------------------------------
    # Context & memory
    # ------------------------------------------------------------------

    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured context memory (facts, slots, flags)",
    )

    requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Latest requirements inferred by LLM (optional)",
    )

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------

    current_docs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Documents retrieved for current turn (RAG)",
    )

    # ------------------------------------------------------------------
    # Retrieval tracking (deterministic guardrails)
    # ------------------------------------------------------------------

    last_retrieval_query: Optional[str] = Field(
        default=None,
        description="Last retrieval query used (source-of-truth)",
    )

    last_retrieval_topic: Optional[str] = Field(
        default=None,
        description="Optional last topic label (if available)",
    )

    # ------------------------------------------------------------------
    # Control & debug
    # ------------------------------------------------------------------

    round: int = Field(
        default=0,
        description="Current multi-step round counter",
    )

    last_action: Optional[str] = Field(
        default=None,
        description="Last high-level action taken by agent (ask / retrieve / answer)",
    )

    # ------------------------------------------------------------------
    # Helpers (NO business logic)
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_user_message_once(self, content: str) -> None:
        """
        Append-only + dedupe:
        - If last visible message is identical user content, do not append.
        """
        c = (content or "")
        if not c.strip():
            return
        if self.messages and self.messages[-1].get("role") == "user" and (self.messages[-1].get("content") or "") == c:
            return
        self.messages.append({"role": "user", "content": c})

    def add_assistant_message_once(self, content: str) -> None:
        """
        Append-only + dedupe:
        - If last visible message is identical assistant content, do not append.
        """
        c = (content or "").strip()
        if not c:
            return
        if self.messages and self.messages[-1].get("role") == "assistant":
            if ((self.messages[-1].get("content") or "").strip() == c):
                return
        self.messages.append({"role": "assistant", "content": c})

    def add_internal_message(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        msg = {"content": content}
        if meta:
            msg["meta"] = meta
        self.internal_messages.append(msg)

    # --------------------------
    # Locks (explicit supervisor contract)
    # --------------------------
    def set_persona_lock(self, persona_id: Optional[str]) -> None:
        """
        Explicit lock that supervisor can rely on.
        - persona_id: "academic" / "practical" / None
        Stored in context as: context["lock_persona"]
        """
        self.context = self.context or {}
        if persona_id:
            self.context["lock_persona"] = str(persona_id)
        else:
            self.context.pop("lock_persona", None)

    def get_persona_lock(self) -> Optional[str]:
        self.context = self.context or {}
        v = self.context.get("lock_persona")
        return str(v) if isinstance(v, str) and v.strip() else None

    # --------------------------
    # Retrieval tracking helpers
    # --------------------------
    def set_last_retrieval_query(self, query: Optional[str], cache_to_context: bool = True) -> None:
        """
        Source-of-truth setter for retrieval query.
        Optionally mirrors into context for backward compatibility only.
        """
        q = (query or "").strip() if query is not None else None
        self.last_retrieval_query = q if q else None

        if cache_to_context:
            self.context = self.context or {}
            if self.last_retrieval_query:
                self.context["last_retrieval_query"] = self.last_retrieval_query
            else:
                self.context.pop("last_retrieval_query", None)

    def get_last_retrieval_query(self) -> Optional[str]:
        """
        Source-of-truth getter.
        If last_retrieval_query is None, will fall back to context cache (legacy) and normalize.
        """
        if self.last_retrieval_query and str(self.last_retrieval_query).strip():
            return str(self.last_retrieval_query).strip()

        self.context = self.context or {}
        v = self.context.get("last_retrieval_query")
        if isinstance(v, str) and v.strip():
            # Do NOT permanently promote automatically (avoid surprising writes),
            # but return for compatibility.
            return v.strip()
        return None

    # --------------------------
    # Round / docs helpers
    # --------------------------
    def reset_round(self) -> None:
        self.round = 0

    def increment_round(self) -> None:
        self.round += 1

    def clear_docs(self) -> None:
        self.current_docs = []

    def snapshot(self) -> Dict[str, Any]:
        """
        Lightweight snapshot for debugging / logging.
        """
        return {
            "session_id": self.session_id,
            "persona_id": self.persona_id,
            "round": self.round,
            "num_messages": len(self.messages),
            "num_docs": len(self.current_docs),
            "last_retrieval_query": self.last_retrieval_query,
            "lock_persona": (self.context or {}).get("lock_persona"),
        }
    