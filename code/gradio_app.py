# code/gradio_app.py
import re
import uuid
from typing import List, Optional, Tuple, Dict, Any

import gradio as gr

from model.conversation_state import ConversationState
from model.state_manager import StateManager
from model.persona_supervisor import PersonaSupervisor
from service.local_vector_store import get_retriever


# ============================================================
# 1) Menu parsing utilities (text-only selection; no buttons)
# ============================================================

_MENU_LINE_RE = re.compile(r"^\s*(\d+)\s*[\)\.\-:]\s*(.+?)\s*$")

def extract_numbered_choices(text: str) -> List[str]:
    """
    Extract choices from lines like:
      1) ขั้นตอน
      2. เอกสาร
      3- ค่าธรรมเนียม
    Returns list of choice strings in order.
    """
    if not text:
        return []
    choices: List[str] = []
    for line in (text or "").splitlines():
        m = _MENU_LINE_RE.match(line)
        if m:
            choices.append(m.group(2).strip())
    return choices

def pick_by_number(choices: List[str], user_input: str) -> Optional[str]:
    s = (user_input or "").strip()
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
    return None


# ============================================================
# 2) Build backend (your existing stack)
# ============================================================

def build_backend() -> Tuple[PersonaSupervisor, StateManager]:
    """
    Create retriever + supervisor + state_manager using your existing code.
    """
    # API server uses fail_if_empty=False, but for local UI you often want fail-fast
    retriever = get_retriever(fail_if_empty=True)
    supervisor = PersonaSupervisor(retriever=retriever)
    state_manager = StateManager()
    return supervisor, state_manager


# ============================================================
# 3) Gradio chat function (wraps supervisor.handle)
# ============================================================

def gradio_chat_fn(
    user_message: str,
    history: List[Tuple[str, str]],
    ui_state: Dict[str, Any],
    conv_state: ConversationState,
    supervisor: PersonaSupervisor,
    state_manager: StateManager,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any], ConversationState, Dict[str, Any], str]:
    """
    ui_state schema:
      {
        "awaiting": bool,
        "choices": [str, ...],
        "session_id": str,
      }

    Returns:
      (history, ui_state, conv_state, payload, cleared_text)
    """
    history = history or []
    ui_state = ui_state or {}

    msg = (user_message or "").strip()
    if not msg:
        return history, ui_state, conv_state, {}, ""

    # 1) If awaiting a menu selection, map "2" -> choice text BEFORE calling supervisor
    if ui_state.get("awaiting") and ui_state.get("choices"):
        mapped = pick_by_number(ui_state["choices"], msg)
        if mapped:
            msg = mapped

    # 2) Call your production entrypoint
    conv_state, bot_reply = supervisor.handle(state=conv_state, user_input=msg)

    # 3) Persist (optional but useful; keeps parity with your CLI/API behavior)
    session_id = ui_state.get("session_id") or conv_state.session_id or "gradio"
    conv_state.session_id = session_id
    state_manager.save(session_id, conv_state)

    # 4) Update chat history (show original user text the user typed, not mapped msg)
    history = history + [(user_message, bot_reply)]

    # 5) Detect if bot reply contains a numbered menu -> set awaiting + store choices
    choices = extract_numbered_choices(bot_reply)
    if choices:
        ui_state = {
            **ui_state,
            "awaiting": True,
            "choices": choices,
        }
    else:
        ui_state = {
            **ui_state,
            "awaiting": False,
            "choices": [],
        }

    # 6) Debug payload (lightweight)
    payload = {
        "session_id": session_id,
        "persona_id": conv_state.persona_id,
        "last_action": conv_state.last_action,
        "round": conv_state.round,
        "snapshot": conv_state.snapshot(),
        "ui_state": ui_state,
        "num_current_docs": len(conv_state.current_docs or []),
        "top_doc_metadata_keys": (
            list((conv_state.current_docs[0].get("metadata") or {}).keys())[:10]
            if (conv_state.current_docs and isinstance(conv_state.current_docs[0], dict))
            else []
        ),
    }

    return history, ui_state, conv_state, payload, ""


# ============================================================
# 4) Launch Gradio
# ============================================================

def launch_gradio(share: bool = False):
    supervisor, state_manager = build_backend()

    session_id = f"g_{uuid.uuid4().hex[:8]}"
    initial_state = ConversationState(session_id=session_id, persona_id="practical", context={})

    with gr.Blocks() as demo:
        gr.Markdown("## Restbiz — PersonaSupervisor (Gradio UI)")

        # Gradio compatibility: some versions don't support show_copy_button
        try:
            chatbot = gr.Chatbot(height=560, show_copy_button=True)
        except TypeError:
            chatbot = gr.Chatbot(height=560)

        ui_state = gr.State({"awaiting": False, "choices": [], "session_id": session_id})
        conv_state = gr.State(initial_state)

        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="พิมพ์คำถาม… (ถ้ามีเมนูเลขรอบก่อน พิมพ์ 1/2/3 ได้เลย)",
                lines=1,
                scale=8,
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Backend payload (debug)", open=False):
            payload_box = gr.JSON(label="payload")

        def _on_send(user_message, history, ui_state_dict, conv_state_obj):
            try:
                return gradio_chat_fn(
                    user_message=user_message,
                    history=history,
                    ui_state=ui_state_dict,
                    conv_state=conv_state_obj,
                    supervisor=supervisor,
                    state_manager=state_manager,
                )
            except Exception as e:
                history = (history or []) + [(user_message, f"Error: {e}")]
                safe_ui = {"awaiting": False, "choices": [], "session_id": session_id}
                payload = {"error": str(e)}
                return history, safe_ui, conv_state_obj, payload, ""

        send.click(
            _on_send,
            inputs=[msg, chatbot, ui_state, conv_state],
            outputs=[chatbot, ui_state, conv_state, payload_box, msg],
        )
        msg.submit(
            _on_send,
            inputs=[msg, chatbot, ui_state, conv_state],
            outputs=[chatbot, ui_state, conv_state, payload_box, msg],
        )

    demo.launch(share=share)

if __name__ == "__main__":
    launch_gradio(share=True)
