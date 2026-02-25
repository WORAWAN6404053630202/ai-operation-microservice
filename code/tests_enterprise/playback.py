#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/playback.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Dict, Any

from model.conversation_state import ConversationState


@dataclass
class TurnResult:
    user: str
    assistant: str
    persona_id: str
    fsm_state: str
    retrieval_calls: int
    llm_calls: int


def last_assistant_text(state: ConversationState) -> str:
    for m in reversed(state.messages or []):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def get_fsm_state(state: ConversationState) -> str:
    ctx = state.context or {}
    return str(ctx.get("fsm_state") or "S_IDLE_MENU")


def run_playback(
    sup,
    state: ConversationState,
    turns: List[str],
    retriever_spy,
    llm_stats,
) -> Tuple[ConversationState, List[TurnResult]]:
    out: List[TurnResult] = []
    for u in turns:
        before_r = len(retriever_spy.queries)
        before_l = llm_stats.count()

        state, reply = sup.handle(state, u)

        after_r = len(retriever_spy.queries)
        after_l = llm_stats.count()

        out.append(
            TurnResult(
                user=u,
                assistant=reply or "",
                persona_id=str(state.persona_id),
                fsm_state=get_fsm_state(state),
                retrieval_calls=after_r - before_r,
                llm_calls=after_l - before_l,
            )
        )
    return state, out