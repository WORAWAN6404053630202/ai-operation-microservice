from __future__ import annotations

import socket
import uuid
from typing import List, Dict, Optional, Tuple

import gradio as gr

from model.state_manager import StateManager
from model.conversation_state import ConversationState
from model.persona_supervisor import PersonaSupervisor
from service.local_vector_store import get_retriever


# ============================================================
# Runtime singletons
# ============================================================
_STATE_MANAGER = StateManager()
_RETRIEVER = None
_SUPERVISOR: Optional[PersonaSupervisor] = None


def _ensure_runtime() -> PersonaSupervisor:
    global _RETRIEVER, _SUPERVISOR
    if _SUPERVISOR is not None:
        return _SUPERVISOR
    _RETRIEVER = get_retriever(fail_if_empty=True)
    _SUPERVISOR = PersonaSupervisor(retriever=_RETRIEVER)
    return _SUPERVISOR


# ============================================================
# State helpers
# ============================================================
def _new_session_id() -> str:
    return str(uuid.uuid4())[:8]


def create_initial_state(session_id: str) -> ConversationState:
    return ConversationState(
        session_id=session_id,
        persona_id="practical",
        context={},
    )


def _safe_load_state(session_id: str) -> ConversationState:
    st = _STATE_MANAGER.load(session_id) if session_id else None
    if st is None:
        sid = session_id or _new_session_id()
        st = create_initial_state(sid)
        _STATE_MANAGER.save(sid, st)
    return st


def _session_label(st: ConversationState) -> str:
    sid = getattr(st, "session_id", "") or "-"
    persona = getattr(st, "persona_id", "") or "-"
    return f"{sid} • mode: {persona}"


# ============================================================
# Session operations
# Chat UI expects: List[{"role":"user|assistant", "content":"..."}]
# ============================================================
def _init_session() -> Tuple[str, ConversationState, List[Dict[str, str]], str]:
    supervisor = _ensure_runtime()
    sid = _new_session_id()
    st = create_initial_state(sid)
    _STATE_MANAGER.save(sid, st)

    st, greet = supervisor.handle(state=st, user_input="")
    _STATE_MANAGER.save(sid, st)

    history: List[Dict[str, str]] = []
    if greet:
        history.append({"role": "assistant", "content": greet})

    return sid, st, history, _session_label(st)


def on_send(user_text: str, session_id: str, history: List[Dict[str, str]]):
    supervisor = _ensure_runtime()
    user_text = (user_text or "").strip()

    st = _safe_load_state(session_id)

    if not user_text:
        history = history or []
        return "", st.session_id, history, _session_label(st), history

    st, reply = supervisor.handle(state=st, user_input=user_text)
    _STATE_MANAGER.save(st.session_id, st)

    history = history or []
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply or ""})

    return "", st.session_id, history, _session_label(st), history


def on_new_session():
    sid, st, history, label = _init_session()
    return sid, st, history, label, history


def on_reset(session_id: str):
    supervisor = _ensure_runtime()

    if not session_id:
        sid, st, history, label = _init_session()
        return sid, st, history, label, history

    st = create_initial_state(session_id)
    _STATE_MANAGER.save(session_id, st)

    st, greet = supervisor.handle(state=st, user_input="")
    _STATE_MANAGER.save(session_id, st)

    history: List[Dict[str, str]] = []
    if greet:
        history.append({"role": "assistant", "content": greet})

    return session_id, st, history, _session_label(st), history


def on_clear_chat(session_id: str):
    st = _safe_load_state(session_id)
    history: List[Dict[str, str]] = []
    return st.session_id, st, history, _session_label(st), history


# ============================================================
# Networking helper
# ============================================================
def _best_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        try:
            s.close()
        except Exception:
            pass


# ============================================================
# UI/UX CSS
# ============================================================
CUSTOM_CSS = """
:root{
  --bg0:#070a12;
  --bg1:#0b1220;

  --panel:rgba(255,255,255,0.07);
  --panel2:rgba(255,255,255,0.04);
  --border:rgba(255,255,255,0.14);

  --text:rgba(255,255,255,0.95);
  --muted:rgba(255,255,255,0.74);

  --brand:rgba(255,140,0,0.95);
  --brandSoft:rgba(255,140,0,0.18);

  --shadow: 0 18px 60px rgba(0,0,0,0.38);
}

body, .gradio-container{
  background:
    radial-gradient(1200px 700px at 15% 0%, rgba(116,72,255,0.22), transparent 60%),
    radial-gradient(1000px 700px at 90% 10%, rgba(60,180,255,0.14), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

.gradio-container{ max-width: 1180px !important; }
.gr-row { flex-wrap: wrap !important; gap: 14px !important; }

#topbar{
  border-radius: 22px;
  padding: 16px 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  color: var(--text);
}
#brand_title{
  font-size: 28px;
  font-weight: 900;
  letter-spacing: 0.4px;
  line-height: 1.2;
  color: var(--text);
}
#brand_subtitle{
  margin-top: 6px;
  font-size: 14px;
  line-height: 1.45;
  color: var(--muted);
  max-width: 900px;
}

.card{
  border-radius: 18px;
  padding: 12px 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: 0 10px 32px rgba(0,0,0,0.25);
  color: var(--text);
}

#chatwrap{
  border-radius: 18px;
  background: var(--panel);
  border: 1px solid var(--border);
  overflow: hidden;
  box-shadow: var(--shadow);
}

#composer{
  border-radius: 18px;
  padding: 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: 0 10px 32px rgba(0,0,0,0.20);
}

.gr-text-input textarea,
.gr-text-input input,
textarea, input{
  background: rgba(10,14,24,0.78) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  caret-color: var(--text) !important;
  border-radius: 14px !important;
  -webkit-text-fill-color: var(--text) !important;
}

.gr-text-input textarea::placeholder,
.gr-text-input input::placeholder,
textarea::placeholder, input::placeholder{
  color: rgba(255,255,255,0.55) !important;
}

#sendbtn button{
  height: 46px;
  border-radius: 14px !important;
  font-weight: 800;
  background: rgba(255,140,0,0.30) !important;
  border: 1px solid rgba(255,140,0,0.35) !important;
}

#helpbox{
  margin-top: 10px;
  padding: 10px 10px;
  border-radius: 14px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  color: rgba(255,255,255,0.82);
  font-size: 13px;
  line-height: 1.45;
}

@media (max-width: 980px){
  .gradio-container{ padding: 10px !important; }
  #topbar{ border-radius: 16px; }
  #chatwrap{ border-radius: 16px; }
  #composer{ border-radius: 16px; }
}

footer{ display:none !important; }
"""


# ============================================================
# JS (run via gradio event system): Enter send, Shift+Enter newline
# IMPORTANT: This must be a JS FUNCTION BODY string (no <script> tags)
# ============================================================
ENTER_TO_SEND_JS_ON_LOAD = r"""
() => {
  const SELECTORS = [
    "#user_input textarea",
    "[id='user_input'] textarea",
    "[id*='user_input'] textarea",
  ];

  function findTextarea() {
    for (const sel of SELECTORS) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    // fallback: first visible textarea in the composer area
    const all = Array.from(document.querySelectorAll("textarea"));
    return all.find(t => t.offsetParent !== null) || null;
  }

  function findSendButton() {
    return document.querySelector("#sendbtn button") || null;
  }

  function insertNewline(textarea) {
    const start = textarea.selectionStart ?? textarea.value.length;
    const end = textarea.selectionEnd ?? textarea.value.length;
    textarea.setRangeText("\n", start, end, "end");
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function bind() {
    const textarea = findTextarea();
    const sendBtn = findSendButton();
    if (!textarea || !sendBtn) return false;

    if (textarea.dataset.boundEnterSend === "1") return true;
    textarea.dataset.boundEnterSend = "1";

    // Capture phase on textarea itself
    textarea.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;

      // Shift+Enter -> newline
      if (e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        insertNewline(textarea);
        return;
      }

      // Enter -> send
      e.preventDefault();
      e.stopPropagation();
      sendBtn.click();
    }, true);

    return true;
  }

  // Retry because Gradio mounts asynchronously
  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    const ok = bind();
    if (ok || tries > 80) clearInterval(timer);
  }, 200);

  return [];
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RESTBIZ Persona AI") as demo:
        session_id_state = gr.State("")
        state_obj = gr.State(None)
        chat_history = gr.State([])  # List[Dict] messages

        gr.Markdown(
            """
<div id="topbar">
  <div id="brand_title">RESTBIZ</div>
  <div id="brand_subtitle">ผู้ช่วยด้านกฎหมาย สำหรับธุรกิจร้านอาหาร</div>
</div>
""",
            elem_id="topbar",
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=7, min_width=320, elem_id="chat_col"):
                with gr.Group(elem_id="chatwrap"):
                    chatbot = gr.Chatbot(label="", height=640)

                with gr.Group(elem_id="composer"):
                    user_input = gr.Textbox(
                        label="Message",
                        placeholder="พิมพ์คำถาม…",
                        lines=3,
                        max_lines=10,
                        elem_id="user_input",
                    )
                    send_btn = gr.Button("Send", variant="primary", elem_id="sendbtn")

            with gr.Column(scale=3, min_width=260, elem_id="side_col"):
                try:
                    container = gr.Accordion("Session & Controls", open=True)
                except Exception:
                    container = gr.Group()

                with container:
                    with gr.Group(elem_classes=["card"], elem_id="sidebar_actions"):
                        session_view = gr.Textbox(
                            label="Session",
                            interactive=False,
                            placeholder="(auto)",
                        )

                        gr.Markdown(
                            """
<div id="helpbox">
  <div><b>New session</b> — เริ่มเซสชันใหม่</div>
  <div><b>Reset session</b> — รีเซ็ตสถานะของเซสชันเดิม</div>
  <div><b>Clear chat</b> — ล้างข้อความบนหน้าจอ</div>
</div>
""",
                        )

                        btn_new = gr.Button("➕ New session", variant="primary")
                        btn_reset = gr.Button("🔄 Reset session")
                        btn_clear = gr.Button("🧹 Clear chat")

        def _init():
            sid, st, hist, label = _init_session()
            return sid, st, hist, label, hist

        # ✅ JS is executed reliably here (not via <script>)
        demo.load(
            _init,
            outputs=[session_id_state, state_obj, chat_history, session_view, chatbot],
            js=ENTER_TO_SEND_JS_ON_LOAD,
        )

        send_btn.click(
            on_send,
            inputs=[user_input, session_id_state, chat_history],
            outputs=[user_input, session_id_state, chat_history, session_view, chatbot],
        )

        # NOTE: DO NOT use user_input.submit — it conflicts with desired behavior.

        btn_new.click(
            on_new_session,
            inputs=[],
            outputs=[session_id_state, state_obj, chat_history, session_view, chatbot],
        )
        btn_reset.click(
            on_reset,
            inputs=[session_id_state],
            outputs=[session_id_state, state_obj, chat_history, session_view, chatbot],
        )
        btn_clear.click(
            on_clear_chat,
            inputs=[session_id_state],
            outputs=[session_id_state, state_obj, chat_history, session_view, chatbot],
        )

    return demo


def main():
    demo = build_app()
    port = 3000
    lan_ip = _best_lan_ip()

    print("\n" + "=" * 60)
    print(f"Open locally:  http://127.0.0.1:{port}")
    print(f"Open on LAN:   http://{lan_ip}:{port}")
    print("=" * 60 + "\n")

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            show_error=True,
            share=False,
            css=CUSTOM_CSS,
        )
    except TypeError:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            show_error=True,
            share=False,
        )


if __name__ == "__main__":
    main()