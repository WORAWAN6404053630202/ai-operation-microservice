# code/model/persona_academic.py
import json
import re
import time
from typing import Tuple, Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import conf
from model.conversation_state import ConversationState
from utils.prompts_academic import SYSTEM_PROMPT as SYSTEM_PROMPT_ACADEMIC


class AcademicPersonaService:
    """
    Academic Persona (REQUIRED FLOW)
    2.1 Retrieve -> ask REQUIRED SLOTS (single batch)
    2.2 After slots -> ask which SECTIONS (dynamic; only what exists for this case)
    3) Final answer (evidence-only; best-effort)
    4) Mark done for supervisor auto-return

    Production fixes in this file (P0/P1):
    - ✅ Do NOT create academic_flow.stage="idle" automatically (prevents supervisor lock on "idle")
    - ✅ Retrieve ONLY once when starting intake (store academic_question); never re-retrieve on slot replies
    - ✅ Use academic_question for final prompt (never fallback to "1,2" etc.)
    - ✅ Make force_intake meaningful: keep intake stage, re-ask slots on greeting/noise, never reset stage
    - ✅ Stronger greeting/noise detection inside intake (backup safety)
    """

    persona_id = "academic"

    # For parsing numbered options
    _OPTION_LINE_RE = re.compile(r"(?m)^\s*(\d{1,2})\s*[\)\.\:]\s*(.+?)\s*$")

    # Detect "ทั้งหมด"
    _SELECT_ALL_RE = re.compile(r"(ทั้งหมด|ทุกข้อ|ทุกหัวข้อ|เอาทั้งหมด|all|everything)", re.IGNORECASE)

    # Backup greeting/noise detection (should already be handled by supervisor, but keep safe)
    _TH_LAUGH_RE = re.compile(r"^\s*5{3,}\s*$")
    _EN_GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo+)\b", re.IGNORECASE)
    _EN_GOOD_TIME_RE = re.compile(r"^\s*good\s+(morning|afternoon|evening|night)\b", re.IGNORECASE)
    _TH_SAWASDEE_FUZZY_RE = re.compile(r"^\s*สว[^\s]{0,6}ดี", re.IGNORECASE)
    _TH_WATDEE_RE = re.compile(r"^\s*หวัดดี", re.IGNORECASE)
    _FILLER_ONLY_RE = re.compile(r"^\s*(ครับ|คับ|ค่ะ|คะ|จ้า|จ้ะ|ค่า|งับ)\s*$", re.IGNORECASE)
    _NOISE_ONLY_RE = re.compile(r"^\s*(?:[อ-ฮะ-์]+|[a-z]+|[!?.]+)\s*$", re.IGNORECASE)

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
    # Safe append (dedupe)
    # ----------------------------
    def _append_user_once(self, state: ConversationState, content: str) -> None:
        if content is None:
            return
        c = str(content)
        if not c.strip():
            return
        if state.messages and state.messages[-1].get("role") == "user" and (state.messages[-1].get("content") or "") == c:
            return
        state.messages.append({"role": "user", "content": c})

    def _append_assistant(self, state: ConversationState, content: str) -> None:
        """
        Safe assistant append:
        - Avoid duplicate consecutive assistant messages
        """
        if content is None:
            return
        c = str(content).strip()
        if not c:
            return
        state.messages = state.messages or []
        if state.messages:
            last = state.messages[-1]
            if last.get("role") == "assistant" and (last.get("content") or "").strip() == c:
                return
        state.messages.append({"role": "assistant", "content": c})

    # ----------------------------
    # Context helpers (flow)
    # ----------------------------
    def _get_flow(self, state: ConversationState) -> Optional[Dict[str, Any]]:
        """
        IMPORTANT (P0):
        - Do NOT create flow on read.
        - Return None if flow not started yet.
        """
        state.context = state.context or {}
        flow = state.context.get("academic_flow")
        if isinstance(flow, dict) and str(flow.get("stage") or "").strip():
            return flow
        return None

    def _ensure_flow(self, state: ConversationState, stage: str, **kwargs) -> Dict[str, Any]:
        """
        Create/overwrite flow only when we truly start intake or update a known stage.
        """
        state.context = state.context or {}
        flow = state.context.get("academic_flow")
        if not isinstance(flow, dict):
            flow = {}
        flow["stage"] = stage
        flow.update(kwargs)
        state.context["academic_flow"] = flow
        return flow

    def _set_flow(self, state: ConversationState, **kwargs) -> None:
        """
        Update flow if it exists; create only if 'stage' provided.
        """
        state.context = state.context or {}
        flow = state.context.get("academic_flow")
        if not isinstance(flow, dict):
            flow = {}
        if "stage" in kwargs and kwargs.get("stage"):
            flow["stage"] = kwargs.get("stage")
        for k, v in kwargs.items():
            if k == "stage":
                continue
            flow[k] = v
        # Only persist if stage is set (intake truly started)
        if str(flow.get("stage") or "").strip():
            state.context["academic_flow"] = flow

    def _mark_done(self, state: ConversationState) -> None:
        self._set_flow(state, stage="done")
        state.context["auto_return_to_practical"] = True

    # ----------------------------
    # Greeting/noise (backup safety)
    # ----------------------------
    def _looks_like_greeting_or_noise(self, user_text: str) -> bool:
        raw = (user_text or "").strip()
        if not raw:
            return True
        if self._TH_LAUGH_RE.match(raw):
            return True
        if self._FILLER_ONLY_RE.match(raw):
            return True
        low = raw.lower().strip()
        if self._EN_GREETING_RE.match(low) or self._EN_GOOD_TIME_RE.match(low):
            return True
        if self._TH_WATDEE_RE.match(raw) or self._TH_SAWASDEE_FUZZY_RE.match(raw):
            return True
        if len(raw) <= 14 and self._NOISE_ONLY_RE.match(raw):
            return True
        return False

    # ----------------------------
    # Retrieval (only once per intake)
    # ----------------------------
    def _retrieve_docs(self, query: str) -> List[Dict[str, Any]]:
        docs = self.retriever.invoke(query)
        results: List[Dict[str, Any]] = []
        for d in (docs or [])[: int(getattr(conf, "RETRIEVAL_TOP_K", 20) or 20)]:
            results.append(
                {
                    "content": (getattr(d, "page_content", "") or "")[:600],
                    "metadata": getattr(d, "metadata", {}) or {},
                }
            )
        return results

    def _start_intake_with_retrieval(self, state: ConversationState, user_question: str) -> None:
        """
        P0: Retrieve only here (once), store academic_question, set last_retrieval_query.
        """
        q = (user_question or "").strip()
        if not q:
            q = "กฎหมายร้านอาหาร ใบอนุญาต ภาษี VAT จดทะเบียน สุขาภิบาล ประกันสังคม"

        state.context = state.context or {}
        state.context["academic_question"] = q

        state.current_docs = self._retrieve_docs(q)
        state.last_retrieval_query = q
        state.context["last_retrieval_query"] = q

    # ----------------------------
    # Option parsing / binding
    # ----------------------------
    def _extract_numbered_options(self, text: str) -> Dict[int, str]:
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

    def _parse_numbers(self, user_text: str) -> List[int]:
        if not user_text:
            return []
        nums = [int(x) for x in re.findall(r"\d{1,2}", user_text)]
        nums = [n for n in nums if 0 < n < 100]
        seen = set()
        out = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _is_select_all(self, user_text: str) -> bool:
        return bool(self._SELECT_ALL_RE.search(user_text or ""))

    def _bind_choice_if_any(self, state: ConversationState, user_text: str) -> Dict[str, Any]:
        """
        Generic binder for both stages (slots & sections).
        Uses state.context["pending_options"] generated by the bot question.
        """
        ctx = state.context or {}
        pending = ctx.get("pending_options") or {}
        if not isinstance(pending, dict) or not pending:
            return {"bound": False}

        if self._is_select_all(user_text):
            keys = sorted([int(k) for k in pending.keys() if str(k).isdigit()])
            resolved = [f"{k}) {pending.get(k)}" for k in keys]
            ctx["resolved_selection"] = {"type": "all", "selected": keys, "resolved": resolved, "raw": user_text}
            ctx["pending_options"] = {}
            ctx["pending_question"] = ""
            state.context = ctx
            return {"bound": True, "mode": "all", "selected": keys}

        nums = self._parse_numbers(user_text)
        if not nums:
            return {"bound": False}

        valid = [n for n in nums if n in pending]
        if not valid:
            return {"bound": False, "reason": "numbers_not_in_options"}

        resolved = [f"{n}) {pending[n]}" for n in valid]
        ctx["resolved_selection"] = {"type": "numbers", "selected": valid, "resolved": resolved, "raw": user_text}
        ctx["pending_options"] = {}
        ctx["pending_question"] = ""
        state.context = ctx
        return {"bound": True, "mode": "numbers", "selected": valid}

    # ----------------------------
    # Stage 2.1: required slots question (single batch)
    # ----------------------------
    def _ask_required_slots(self, state: ConversationState) -> str:
        msg = (
            "เพื่อให้ตอบได้ตรงกรณี รบกวนตอบ “ข้อมูลจำเป็น” ต่อไปนี้รวดเดียวครับ (ตอบเท่าที่รู้)\n"
            "1) ที่ตั้งร้าน/กิจการ: 1) กทม.  2) ต่างจังหวัด (ระบุจังหวัด)\n"
            "2) รูปแบบผู้ประกอบการ: 1) บุคคลธรรมดา  2) นิติบุคคล\n"
            "3) ลักษณะกิจการแบบย่อ: เช่น ร้านอาหารทั่วไป/คาเฟ่/บุฟเฟ่ต์/ผับบาร์/เดลิเวอรี่เท่านั้น\n"
            "4) เขต/อำเภอ/เทศบาล (ถ้าทราบ)\n"
            "5) สถานะปัจจุบัน: เคยมีใบอนุญาต/จดทะเบียนแล้วหรือยัง (ถ้ารู้)"
        )

        opts = self._extract_numbered_options(msg)
        if len(opts) >= 2:
            state.context["pending_options"] = opts
            state.context["pending_question"] = msg
        else:
            state.context["pending_options"] = {}
            state.context["pending_question"] = ""

        self._set_flow(state, stage="awaiting_slots")
        return msg

    def _save_slots_best_effort(self, state: ConversationState, user_text: str, bind_info: Dict[str, Any]) -> None:
        ctx = state.context or {}
        slots = ctx.get("academic_slots")
        if not isinstance(slots, dict):
            slots = {}

        slots["raw"] = (user_text or "").strip()

        if bind_info.get("bound") and bind_info.get("mode") == "numbers":
            slots["selected_numbers"] = bind_info.get("selected") or []

        ctx["academic_slots"] = slots
        state.context = ctx

    # ----------------------------
    # Stage 2.2: dynamic section menu (only what exists)
    # ----------------------------
    def _available_sections_from_docs(self, state: ConversationState) -> List[Dict[str, str]]:
        docs = state.current_docs or []

        def has_any_anykey(keys: List[str]) -> bool:
            for d in docs:
                md = (d.get("metadata") or {})
                for key in keys:
                    val = md.get(key)
                    if val is None:
                        continue
                    s = str(val).strip()
                    if s and s.lower() != "nan":
                        return True
            return False

        candidates = [
            (["operation_steps", "operation_step", "steps", "procedure", "ขั้นตอนการดำเนินการ"], "ขั้นตอนการดำเนินการ"),
            (["identification_documents", "documents", "required_documents", "เอกสาร ยืนยันตัวตน", "เอกสารที่ต้องใช้"], "เอกสารที่ต้องใช้"),
            (["fees", "fee", "ค่าธรรมเนียม"], "ค่าธรรมเนียม"),
            (["operation_duration", "duration", "ระยะเวลา การดำเนินการ", "ระยะเวลาดำเนินการ"], "ระยะเวลา"),
            (["service_channel", "channel", "ช่องทางการ ให้บริการ", "ช่องทาง", "หน่วยงาน", "department"], "ช่องทาง/สถานที่ยื่น"),
            (["terms_and_conditions", "conditions", "เงื่อนไขและหลักเกณฑ์"], "เงื่อนไขและหลักเกณฑ์"),
            (["legal_regulatory", "law", "regulation", "ข้อกำหนดทางกฎหมาย และข้อบังคับ", "บทลงโทษ"], "ข้อกฎหมาย/ข้อควรระวัง/บทลงโทษ"),
        ]

        out: List[Dict[str, str]] = []
        for keys, label in candidates:
            if has_any_anykey(keys):
                out.append({"key": keys[0], "label": label})
        return out

    def _ask_sections(self, state: ConversationState) -> str:
        sections = self._available_sections_from_docs(state)

        if not sections:
            state.context["selected_sections"] = {"type": "all", "keys": []}
            self._set_flow(state, stage="awaiting_sections")
            return ""

        lines = ["ก่อนสรุปคำตอบ คุณอยากรู้ “ส่วนไหน” ครับ (เลือกได้หลายข้อ/หรือพิมพ์ “ทั้งหมด”)"]
        opts: Dict[int, str] = {}
        for i, sec in enumerate(sections, start=1):
            opts[i] = sec["label"]
            lines.append(f"{i}) {sec['label']}")

        msg = "\n".join(lines)

        state.context["section_catalog"] = sections
        state.context["pending_options"] = opts
        state.context["pending_question"] = msg
        self._set_flow(state, stage="awaiting_sections")
        return msg

    def _save_selected_sections(self, state: ConversationState, user_text: str, bind_info: Dict[str, Any]) -> None:
        ctx = state.context or {}
        catalog = ctx.get("section_catalog") or []
        if not isinstance(catalog, list):
            catalog = []

        if self._is_select_all(user_text) or (bind_info.get("bound") and bind_info.get("mode") == "all"):
            ctx["selected_sections"] = {"type": "all", "keys": [c.get("key") for c in catalog if c.get("key")]}
            state.context = ctx
            return

        nums = bind_info.get("selected") if bind_info.get("bound") and bind_info.get("mode") == "numbers" else []
        if not nums:
            ctx["selected_sections"] = {"type": "all", "keys": [c.get("key") for c in catalog if c.get("key")]}
            state.context = ctx
            return

        picked: List[str] = []
        for n in nums:
            idx = int(n) - 1
            if 0 <= idx < len(catalog):
                k = catalog[idx].get("key")
                if k:
                    picked.append(k)

        if not picked:
            picked = [c.get("key") for c in catalog if c.get("key")]

        ctx["selected_sections"] = {"type": "picked", "keys": picked}
        state.context = ctx

    # ----------------------------
    # Final answer generation (LLM JSON)
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
            "analysis": "LLM JSON parse failed",
            "action": "answer",
            "execution": {
                "answer": "ขอโทษครับ ตอนนี้ยังสรุปคำตอบที่ยืนยันได้จากเอกสารไม่ได้",
                "context_update": {"auto_return_to_practical": True},
            },
        }

    def _build_final_prompt(self, state: ConversationState, user_question: str) -> str:
        ctx = state.context or {}
        slots = ctx.get("academic_slots") or {}
        selected = ctx.get("selected_sections") or {"type": "all", "keys": []}

        docs_json = []
        for d in (state.current_docs or [])[:20]:
            md = d.get("metadata", {}) or {}
            docs_json.append(
                {
                    "metadata": {k: ("" if v is None else str(v)) for k, v in md.items()},
                    "content": (d.get("content", "") or "")[:250],
                }
            )

        return f"""
{SYSTEM_PROMPT_ACADEMIC}

USER_QUESTION:
{user_question}

SLOTS:
{json.dumps(slots, ensure_ascii=False, indent=2)}

SELECTED_SECTIONS:
{json.dumps(selected, ensure_ascii=False, indent=2)}

DOCUMENTS ({len(state.current_docs or [])} found):
{json.dumps(docs_json, ensure_ascii=False, indent=2)}

Return JSON:
""".strip()

    # ----------------------------
    # ENTRYPOINT
    # ----------------------------
    def handle(
        self,
        state: ConversationState,
        user_input: str,
        force_intake: bool = False,
        _internal: bool = False,
    ) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        state.persona_id = self.persona_id

        # DEDUPE: supervisor already appends user; but allow standalone usage too
        if not _internal:
            self._append_user_once(state, user_input)

        user_text = (user_input or "").strip()

        flow = self._get_flow(state)
        stage = (flow.get("stage") if isinstance(flow, dict) else "") or ""

        # -----------------------
        # Start intake (NO implicit idle stage)
        # -----------------------
        if not stage:
            # If we were force-routed here but got greeting/noise/blank -> ask for question (no flow created)
            if force_intake and self._looks_like_greeting_or_noise(user_text):
                msg = "อยากให้ช่วยเรื่องไหนเกี่ยวกับกฎหมาย/ขั้นตอนของร้านอาหารครับ"
                self._append_assistant(state, msg)
                return state, msg

            # If there's a real question -> start intake + retrieve once
            if user_text:
                self._ensure_flow(state, stage="awaiting_slots")
                self._start_intake_with_retrieval(state, user_question=user_text)

                q = self._ask_required_slots(state)
                self._append_assistant(state, q)
                state.round = int(getattr(state, "round", 0) or 0) + 1
                return state, q

            # no question
            msg = "อยากให้ช่วยเรื่องไหนเกี่ยวกับกฎหมาย/ขั้นตอนของร้านอาหารครับ"
            self._append_assistant(state, msg)
            return state, msg

        # Refresh stage after potential creation
        flow = self._get_flow(state) or {}
        stage = (flow.get("stage") or "").strip()

        # -----------------------
        # Stage: awaiting_slots
        # -----------------------
        if stage == "awaiting_slots":
            # force_intake: greeting/noise must re-ask slots and keep stage (no reset)
            if force_intake and self._looks_like_greeting_or_noise(user_text):
                q = self._ask_required_slots(state)
                self._append_assistant(state, q)
                return state, q

            bind = self._bind_choice_if_any(state, user_text)

            # If empty/garbage -> re-ask
            if not user_text or (not bind.get("bound") and self._looks_like_greeting_or_noise(user_text)):
                q = self._ask_required_slots(state)
                self._append_assistant(state, q)
                return state, q

            # Otherwise treat as slot answer (do NOT retrieve again here)
            self._save_slots_best_effort(state, user_text, bind)

            q2 = self._ask_sections(state)
            if q2.strip():
                self._append_assistant(state, q2)
                state.round = int(getattr(state, "round", 0) or 0) + 1
                return state, q2

            # No sections available -> still proceed to awaiting_sections to finalize
            self._set_flow(state, stage="awaiting_sections")

        # -----------------------
        # Stage: awaiting_sections -> final answer
        # -----------------------
        if (self._get_flow(state) or {}).get("stage") == "awaiting_sections":
            bind = self._bind_choice_if_any(state, user_text)
            self._save_selected_sections(state, user_text, bind)

            # P0: Always use academic_question (never use "1,2" slot reply)
            uq = (state.context or {}).get("academic_question") or state.last_retrieval_query or state.context.get("last_retrieval_query") or ""
            uq = (uq or "").strip() or "กฎหมายร้านอาหาร ใบอนุญาต ภาษี VAT จดทะเบียน สุขาภิบาล ประกันสังคม"

            prompt = self._build_final_prompt(state, user_question=uq)
            decision = self._call_llm_json(prompt)

            ex = (decision.get("execution") or {})
            ans = (ex.get("answer") or "").strip()
            if not ans:
                ans = "ขอโทษครับ ตอนนี้ยังไม่พบข้อมูลที่ยืนยันได้ในเอกสาร"

            # apply context update if any
            cu = ex.get("context_update", {})
            if isinstance(cu, dict) and cu:
                state.context.update(cu)

            # mark done (for supervisor auto-return)
            self._mark_done(state)

            # clean transient
            state.context["pending_options"] = {}
            state.context["pending_question"] = ""
            state.context.pop("resolved_selection", None)

            self._append_assistant(state, ans)
            state.round = 0
            return state, ans

        # Stage done or unknown: provide safe fallback
        fallback = "ขอโทษครับ ผมยังไม่เข้าใจ รบกวนอธิบายเพิ่มอีกนิดได้ไหมครับ"
        self._append_assistant(state, fallback)
        return state, fallback