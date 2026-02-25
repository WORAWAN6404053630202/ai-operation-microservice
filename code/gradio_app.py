# code/gradio_app.py
from __future__ import annotations

import uuid
from typing import List, Dict

import gradio as gr

from model.state_manager import StateManager
from model.conversation_state import ConversationState
from model.persona_supervisor import PersonaSupervisor
from service.local_vector_store import get_retriever


# -----------------------------
# State + Debug
# -----------------------------
def create_initial_state(session_id: str) -> ConversationState:
    return ConversationState(
        session_id=session_id,
        persona_id="practical",
        context={},
    )


def _debug_line(state: ConversationState) -> str:
    ctx = state.context or {}

    pending = ctx.get("pending_slot")
    pending_txt = "-"
    if isinstance(pending, dict):
        key = pending.get("key") or "-"
        opts = pending.get("options") or []
        try:
            nopts = len(opts)
        except Exception:
            nopts = "?"
        pending_txt = f"{key}({nopts})"
    elif pending:
        pending_txt = str(pending)

    last_action = getattr(state, "last_action", "-")
    fsm = (ctx.get("fsm_state") or "-")
    did_greet = bool(ctx.get("did_greet"))
    greet_streak = ctx.get("greet_streak", "-")
    persona_id = getattr(state, "persona_id", "-") or "-"

    return (
        f"last_action={last_action} | fsm={fsm} | pending_slot={pending_txt} | "
        f"did_greet={did_greet} | greet_streak={greet_streak} | persona={persona_id}"
    )


def _pretty_debug(state: ConversationState) -> str:
    return f"```text\nDEBUG {_debug_line(state)}\n```"


# -----------------------------
# Lazy Runtime
# -----------------------------
_STATE_MANAGER = StateManager()
_RETRIEVER = None
_SUPERVISOR = None


def _ensure_runtime() -> PersonaSupervisor:
    global _RETRIEVER, _SUPERVISOR
    if _SUPERVISOR is not None:
        return _SUPERVISOR
    _RETRIEVER = get_retriever(fail_if_empty=True)
    _SUPERVISOR = PersonaSupervisor(retriever=_RETRIEVER)
    return _SUPERVISOR


# -----------------------------
# Session Logic
# -----------------------------
def _new_session_id() -> str:
    return str(uuid.uuid4())[:8]


def init_session():
    supervisor = _ensure_runtime()
    session_id = _new_session_id()

    state = create_initial_state(session_id)
    _STATE_MANAGER.save(session_id, state)

    state, greet = supervisor.handle(state=state, user_input="")
    _STATE_MANAGER.save(session_id, state)

    history: List[Dict] = []
    if greet:
        history.append({"role": "assistant", "content": greet})

    return session_id, state, history, _pretty_debug(state)


def on_send(user_text: str, session_id: str, history: List[Dict]):
    supervisor = _ensure_runtime()

    user_text = (user_text or "").strip()
    if not user_text:
        st = _STATE_MANAGER.load(session_id) or create_initial_state(session_id)
        return "", session_id, history, _pretty_debug(st), session_id, history

    state = _STATE_MANAGER.load(session_id)
    if state is None:
        state = create_initial_state(session_id)
        _STATE_MANAGER.save(session_id, state)

    state, reply = supervisor.handle(state=state, user_input=user_text)
    _STATE_MANAGER.save(session_id, state)

    history = history or []
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})

    return "", session_id, history, _pretty_debug(state), session_id, history


def on_reset(session_id: str):
    supervisor = _ensure_runtime()

    if not session_id:
        sid, st, hist, dbg = init_session()
        return sid, st, hist, dbg, sid, hist

    state = create_initial_state(session_id)
    _STATE_MANAGER.save(session_id, state)

    state, greet = supervisor.handle(state=state, user_input="")
    _STATE_MANAGER.save(session_id, state)

    history: List[Dict] = []
    if greet:
        history.append({"role": "assistant", "content": greet})

    return session_id, state, history, _pretty_debug(state), session_id, history


def on_new_session():
    sid, st, hist, dbg = init_session()
    return sid, st, hist, dbg, sid, hist


def on_clear_chat(session_id: str):
    st = _STATE_MANAGER.load(session_id) or create_initial_state(session_id)
    _STATE_MANAGER.save(session_id, st)
    hist: List[Dict] = []
    return session_id, st, hist, _pretty_debug(st), session_id, hist


# -----------------------------
# UI (SaaS-like)
# -----------------------------
CUSTOM_CSS = """
:root{
  --bg0:#0b1220;
  --bg1:#0f1a2d;
  --card:rgba(255,255,255,0.06);
  --card2:rgba(255,255,255,0.08);
  --border:rgba(255,255,255,0.10);
  --text:rgba(255,255,255,0.92);
  --muted:rgba(255,255,255,0.70);
}

body, .gradio-container{
  background: radial-gradient(1200px 600px at 20% 0%, rgba(120,70,255,0.25), transparent 60%),
              radial-gradient(1000px 600px at 90% 10%, rgba(60,180,255,0.18), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1));
}

.gradio-container{ max-width: 1180px !important; }

#appbar{
  border-radius: 22px;
  padding: 16px 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
  border: 1px solid var(--border);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  color: var(--text);
}

.badge{
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 12px;
}

.card{
  border-radius: 18px;
  padding: 12px 12px;
  background: var(--card);
  border: 1px solid var(--border);
  box-shadow: 0 8px 22px rgba(0,0,0,0.22);
  color: var(--text);
}

.card h3{
  margin: 6px 0 10px 0;
}

#chatwrap{
  border-radius: 18px;
  background: var(--card);
  border: 1px solid var(--border);
  overflow: hidden;
  box-shadow: 0 10px 28px rgba(0,0,0,0.25);
}

#composer{
  border-radius: 18px;
  padding: 12px;
  background: var(--card);
  border: 1px solid var(--border);
}

#sendbtn button{
  height: 44px;
  border-radius: 14px !important;
}

#controls button{
  border-radius: 14px !important;
}

.small-muted{
  color: var(--muted);
  font-size: 13px;
}

.gr-markdown, .gr-label, label, .wrap{
  color: var(--text) !important;
}

textarea, input{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}

footer{ display:none !important; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RESTBIZ Persona AI") as demo:
        session_id_state = gr.State("")
        state_obj = gr.State(None)
        chat_history = gr.State([])

        gr.Markdown(
            """
<div id="appbar">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:14px;">
    <div>
      <div style="font-size:22px; font-weight:800; letter-spacing:0.2px;">RESTBIZ Persona AI</div>
      <div class="small-muted">Regulatory assistant • Persona supervisor • RAG-backed</div>
    </div>
    <div style="display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end;">
      <span class="badge">⚡ Deterministic greet</span>
      <span class="badge">🧠 StateManager</span>
      <span class="badge">🔎 Retriever k=20</span>
    </div>
  </div>
</div>
""",
            elem_id="appbar",
        )

        with gr.Row():
            # Chat column
            with gr.Column(scale=5):
                with gr.Group(elem_id="chatwrap"):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=600,
                    )

                with gr.Group(elem_id="composer"):
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Message",
                            placeholder="พิมพ์คำถาม…",
                            lines=2,
                            max_lines=6,
                        )
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary", elem_id="sendbtn")

            # Side column
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["card"], elem_id="controls"):
                    gr.Markdown("### Session Controls")
                    btn_new = gr.Button("➕ New session", variant="primary")
                    btn_reset = gr.Button("🔄 Reset session")
                    btn_clear = gr.Button("🧹 Clear chat")
                    session_view = gr.Textbox(
                        label="Session ID",
                        interactive=False,
                        placeholder="(auto)",
                    )

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown("### Debug")
                    debug_box = gr.Markdown("```text\nDEBUG -\n```")

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown(
                        """
### Tips
- ถ้า state แปลก ๆ หรือ `pending_slot` ค้าง → กด **Reset session**
- ถ้าจะเริ่มใหม่ทั้งหมด → กด **New session**
- ถ้าจะล้างหน้าจออย่างเดียว → กด **Clear chat**
"""
                    )

        # ---- init on load ----
        def _init():
            sid, st, hist, dbg = init_session()
            return sid, st, hist, dbg, sid, hist

        demo.load(
            _init,
            outputs=[session_id_state, state_obj, chat_history, debug_box, session_view, chatbot],
        )

        # ---- send ----
        send_btn.click(
            on_send,
            inputs=[user_input, session_id_state, chat_history],
            outputs=[user_input, session_id_state, chat_history, debug_box, session_view, chatbot],
        )
        user_input.submit(
            on_send,
            inputs=[user_input, session_id_state, chat_history],
            outputs=[user_input, session_id_state, chat_history, debug_box, session_view, chatbot],
        )

        # ---- controls ----
        btn_reset.click(
            on_reset,
            inputs=[session_id_state],
            outputs=[session_id_state, state_obj, chat_history, debug_box, session_view, chatbot],
        )
        btn_new.click(
            on_new_session,
            inputs=[],
            outputs=[session_id_state, state_obj, chat_history, debug_box, session_view, chatbot],
        )
        btn_clear.click(
            on_clear_chat,
            inputs=[session_id_state],
            outputs=[session_id_state, state_obj, chat_history, debug_box, session_view, chatbot],
        )

    return demo


def main():
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            radius_size=gr.themes.sizes.radius_lg,
            spacing_size=gr.themes.sizes.spacing_lg,
            text_size=gr.themes.sizes.text_md,
        ),
    )


if __name__ == "__main__":
    main()