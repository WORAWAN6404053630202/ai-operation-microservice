# code/model/conversation_state.py
"""
Conversation State Model (v2 Final)

Single source of truth for conversation lifecycle.
This model is intentionally logic-light and framework-agnostic.
"""

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

    # Helper methods (NO business logic)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_internal_message(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        msg = {"content": content}
        if meta:
            msg["meta"] = meta
        self.internal_messages.append(msg)

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
        }
