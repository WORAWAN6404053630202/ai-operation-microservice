# code/model/persona_academic.py
import json
import re
import time
from typing import Tuple, Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.json_guard import call_llm_json_with_repair

import conf
from model.conversation_state import ConversationState
from utils.prompts_academic import SYSTEM_PROMPT as SYSTEM_PROMPT_ACADEMIC


class AcademicPersonaService:
    """
    Academic Persona (Agentic, detailed, original-like)
    - Mirrors original AgentService flow (retrieve/ask/answer + topic_selection phase)
    - Adds deterministic "choice binding" so user can reply with:
      - single number: 2
      - multi: 2,3 / 2 และ 3 / 2 3
      - free-text: "เลือกข้อ 2" / "ข้อ2" / "ทั้งหมด"
    """

    persona_id = "academic"

    def __init__(self, retriever):
        self.retriever = retriever
        self._init_llm()

    def _init_llm(self):
        model_name = getattr(conf, "OPENROUTER_MODEL_ACADEMIC", conf.OPENROUTER_MODEL)
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=getattr(conf, "TEMPERATURE_ACADEMIC", 0.3),
            max_tokens=getattr(conf, "MAX_TOKENS_ACADEMIC", 8000),
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    # ----------------------------
    # Choice binding helpers
    # ----------------------------
    _OPTION_LINE_RE = re.compile(r"(?m)^\s*(\d{1,2})\s*[\)\.\:]\s*(.+?)\s*$")

    def _extract_numbered_options(self, text: str) -> Dict[int, str]:
        """
        Extract numbered options from assistant question like:
          1) aaa
          2) bbb
        Returns {1: "aaa", 2:"bbb"}.
        """
        options: Dict[int, str] = {}
        for m in self._OPTION_LINE_RE.finditer(text or ""):
            try:
                k = int(m.group(1))
                v = (m.group(2) or "").strip()
                if v:
                    options[k] = v
            except Exception:
                continue
        return options

    def _parse_user_selection_numbers(self, user_text: str) -> Optional[List[int]]:
        """
        Parse selection numbers from user text.
        Supports: "2", "ข้อ 2", "2,3", "2 และ 3", "2 3", "เลือก 2"
        Returns list[int] or None if not a clear numeric selection.
        """
        t = (user_text or "").strip()
        if not t:
            return None

        # quick reject: if it's long free-text without digits, let LLM handle
        if not re.search(r"\d", t):
            return None

        nums = [int(x) for x in re.findall(r"\d{1,2}", t)]
        nums = [n for n in nums if 0 < n < 100]
        return nums or None

    def _is_select_all(self, user_text: str) -> bool:
        t = (user_text or "").strip().lower()
        if not t:
            return False
        # Thai + common variants
        keywords = [
            "ทั้งหมด", "ทุกข้อ", "ทุกหัวข้อ", "เอาทุกข้อ", "เอาทั้งหมด",
            "all", "everything", "both",
        ]
        return any(k in t for k in keywords)

    def _apply_choice_binding(
        self,
        state: ConversationState,
        user_input: str,
        last_bot: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        If last assistant message had numbered options and user replies with selection,
        convert user input into a resolved text to reduce LLM confusion.
        Returns: (user_input_for_prompt, debug_binding_info)
        """
        ctx = state.context or {}
        pending_options = ctx.get("pending_options") or {}

        # If we don't already have pending options, try extracting from last_bot now (safer).
        if not pending_options and last_bot:
            extracted = self._extract_numbered_options(last_bot)
            if len(extracted) >= 2:
                pending_options = extracted
                ctx["pending_options"] = pending_options
                ctx["pending_question"] = last_bot

        if not pending_options:
            return user_input, {"bound": False}

        # "ทั้งหมด" / "all"
        if self._is_select_all(user_input):
            ctx["resolved_selection"] = {
                "type": "all",
                "selected": sorted(list(pending_options.keys())),
                "resolved": [f"{k}) {pending_options[k]}" for k in sorted(pending_options.keys())],
            }
            # Clear pending to avoid accidental reuse
            ctx["pending_options"] = {}
            ctx["pending_question"] = ""
            state.context = ctx
            injected = "ผู้ใช้เลือก: ทั้งหมด (ทุกข้อ)"
            return injected, {"bound": True, "mode": "all"}

        nums = self._parse_user_selection_numbers(user_input)
        if not nums:
            return user_input, {"bound": False}

        # Keep only valid selections
        valid = [n for n in nums if n in pending_options]
        if not valid:
            # user typed numbers but not matching options -> don't clear pending; let LLM ask again
            return user_input, {"bound": False, "reason": "numbers_not_in_options"}

        # De-dup preserve order
        seen = set()
        valid = [n for n in valid if (n not in seen and not seen.add(n))]

        resolved_lines = [f"{n}) {pending_options[n]}" for n in valid]
        ctx["resolved_selection"] = {
            "type": "numbers",
            "selected": valid,
            "resolved": resolved_lines,
            "raw": user_input,
        }

        # Clear pending to avoid binding future turns accidentally
        ctx["pending_options"] = {}
        ctx["pending_question"] = ""
        state.context = ctx

        injected = "ผู้ใช้เลือกข้อ: " + " | ".join(resolved_lines)
        return injected, {"bound": True, "mode": "numbers", "selected": valid}

    # ----------------------------
    # Core methods
    # ----------------------------
    def _call_llm_json(self, prompt: str, max_retries: int = 2) -> dict:
        last_err = None
        for _ in range(max_retries):
            try:
                t0 = time.perf_counter()
                text = (self.llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
                t1 = time.perf_counter()
                if getattr(conf, "DEBUG_LATENCY", True):
                    print(f"[LATENCY] llm_ms={(t1 - t0) * 1000:.0f} prompt_chars={len(prompt)}")

                # strip fences if any
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as e:
                last_err = e
                continue

        if getattr(conf, "DEBUG_LATENCY", True) and last_err:
            print(f"[WARN] LLM JSON parse failed: {last_err}")

        return {
            "input_type": "new_question",
            "analysis": "Parse error",
            "action": "ask",
            "execution": {
                "question": "ขอโทษครับ ระบบมีปัญหาชั่วคราว รบกวนถามใหม่อีกครั้งได้ไหมครับ",
                "context_update": {},
            },
        }

    def _retrieve_docs(self, query: str) -> List[Dict[str, Any]]:
        docs = self.retriever.invoke(query)
        results: List[Dict[str, Any]] = []
        for d in docs[: getattr(conf, "RETRIEVAL_TOP_K", 20)]:
            results.append(
                {
                    "content": (getattr(d, "page_content", "") or "")[:600],
                    "metadata": getattr(d, "metadata", {}) or {},
                }
            )
        return results
    
    # ----------------------------
    # Output sanitizer (no internal ids)
    # ----------------------------
    _ROW_ID_RE = re.compile(r"\brow[_\s-]*id\s*[:=]?\s*\d+(?:\s*,\s*\d+)*", re.IGNORECASE)
    _DOC_ID_RE = re.compile(r"\b(doc(?:ument)?[_\s-]*id|uuid|chroma[_\s-]*id)\s*[:=]?\s*[a-f0-9-]{6,}\b", re.IGNORECASE)
    _REF_GROUP_RE = re.compile(r"(อ้างอิง|อิงจาก|อ้างถึง)\s*(เอกสาร|ข้อมูล)\s*(กลุ่ม|หมายเลข|ชุด)?\s*[:：]?\s*.*", re.IGNORECASE)

    def _sanitize_answer(self, text: str) -> str:
        """
        Remove internal references like row_id/doc_id/uuid and 'อ้างอิงเอกสารกลุ่ม ...'
        Keep answer professional.
        """
        t = (text or "").strip()
        if not t:
            return t

        # Remove row_id mentions
        t = self._ROW_ID_RE.sub("", t)

        # Remove doc/uuid/chroma id mentions
        t = self._DOC_ID_RE.sub("", t)

        # Remove overly-internal "reference group" lines (best-effort: drop the whole line)
        lines = [ln.rstrip() for ln in t.splitlines()]
        kept = []
        for ln in lines:
            ln2 = ln.strip()
            if not ln2:
                kept.append(ln)
                continue
            # Drop lines that are basically internal citation lines
            if self._REF_GROUP_RE.fullmatch(ln2):
                continue
            if "row_id" in ln2.lower() or "doc_id" in ln2.lower() or "uuid" in ln2.lower():
                continue
            kept.append(ln)

        t = "\n".join([x for x in kept]).strip()

        # Clean repeated blank lines
        t = re.sub(r"\n{3,}", "\n\n", t).strip()

        return t

    def handle(self, state: ConversationState, user_input: str, _internal: bool = False) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        state.persona_id = self.persona_id

        # store message
        if not _internal:
            state.messages.append({"role": "user", "content": user_input})

        user_text = (user_input or "").strip()

        # keep last assistant message
        last_bot = next((m["content"] for m in reversed(state.messages[:-1]) if m["role"] == "assistant"), "")
        recent_msgs = state.messages[-15:]

        # ---- choice binding (critical fix) ----
        # If user replied with "2" (or similar) to the previous numbered list, resolve it before LLM sees it.
        user_input_for_prompt, bind_info = self._apply_choice_binding(state, user_input, last_bot)

        # pack docs (send metadata + small content hint)
        docs_json = []
        for d in (state.current_docs or [])[:20]:
            md = d.get("metadata", {}) or {}
            docs_json.append(
                {
                    "metadata": {k: ("" if v is None else str(v)) for k, v in md.items()},
                    "content": (d.get("content", "") or "")[:250],
                }
            )

        special_phase_note = ""
        if state.context.get("phase") == "topic_selection":
            special_phase_note = """
IMPORTANT (PHASE: TOPIC SELECTION):
The user input is NOT a new question.
The user is selecting which topics they want to see after documents were already retrieved.

You MAY retrieve again only if more documents are REQUIRED for the selected topics.
But you MUST NOT retrieve repeatedly or enter a loop.
After ONE additional retrieve (if needed), you MUST answer using CURRENT_DOCS.
"""

        prompt = f"""
{SYSTEM_PROMPT_ACADEMIC}

{special_phase_note}

USER INPUT (RAW):
{user_input}

USER INPUT (NORMALIZED / MAY INCLUDE RESOLVED CHOICE):
{user_input_for_prompt}

CHOICE_BINDING_DEBUG:
{json.dumps(bind_info, ensure_ascii=False)}

LAST ASSISTANT MESSAGE:
{last_bot}

RECENT MESSAGES:
{json.dumps(recent_msgs, ensure_ascii=False, indent=2)}

CURRENT CONTEXT:
{json.dumps(state.context, ensure_ascii=False, indent=2)}

DOCUMENTS ({len(state.current_docs or [])} found):
{json.dumps(docs_json, ensure_ascii=False, indent=2)}

ROUND: {int(getattr(state, "round", 0) or 0)}/{int(getattr(conf, "MAX_ROUNDS", 7) or 7)}

Your JSON response:
"""

        decision = self._call_llm_json(prompt)
        action = (decision.get("action") or "ask").strip()
        exec_ = decision.get("execution", {}) or {}

        # INFINITE LOOP PREVENTION (topic_selection)
        if state.context.get("phase") == "topic_selection" and action == "retrieve":
            if not state.context.get("topic_retrieved_once"):
                state.context["topic_retrieved_once"] = True
            else:
                action = "answer"

        if action == "retrieve":
            q = exec_.get("query") or user_text or user_input
            state.current_docs = self._retrieve_docs(q)
            return self.handle(state, "__auto_post_retrieve__", _internal=True)

        if action == "ask":
            question = (exec_.get("question") or "ต้องการทราบเรื่องใดเป็นหลักครับ").strip()
            state.messages.append({"role": "assistant", "content": question})

            # phase hook (keep same as original behavior)
            if "ตอนนี้โคโค่มีข้อมูลครบแล้ว" in question:
                state.context["phase"] = "topic_selection"
                state.context["topic_retrieved_once"] = False

            # store pending options if question contains a numbered list
            opts = self._extract_numbered_options(question)
            if len(opts) >= 2:
                state.context["pending_options"] = opts
                state.context["pending_question"] = question
            else:
                # don't keep stale pending options
                state.context["pending_options"] = {}
                state.context["pending_question"] = ""

            if isinstance(exec_.get("context_update", {}), dict):
                state.context.update(exec_.get("context_update", {}))

            state.round = int(getattr(state, "round", 0) or 0) + 1
            return state, question

        if action == "answer":
            ans = (exec_.get("answer") or "").strip() or "ขอโทษครับ ตอนนี้ยังไม่พบข้อมูลที่ยืนยันได้ในเอกสาร"

            # IMPORTANT: sanitize internal ids/citations
            ans = self._sanitize_answer(ans)

            state.messages.append({"role": "assistant", "content": ans})

            state.context["phase"] = None
            state.context["topic_retrieved_once"] = False
            state.context["pending_options"] = {}
            state.context["pending_question"] = ""
            state.round = 0
            return state, ans

        fallback = "ขอโทษครับ ผมยังไม่เข้าใจ รบกวนอธิบายเพิ่มอีกนิดได้ไหมครับ"
        state.messages.append({"role": "assistant", "content": fallback})
        return state, fallback
