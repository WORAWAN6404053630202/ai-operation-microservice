# code/model/persona_practical.py
import json
import re
import time
from typing import Tuple, Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import conf
from model.conversation_state import ConversationState
from utils.prompts_practical import SYSTEM_PROMPT as SYSTEM_PROMPT_PRACTICAL


class PracticalPersonaService:
    """
    Practical Persona (Agentic, fast, short)

    PATCH (production):
    - Deterministic greeting -> reply + numbered topic choices (from real metadata)
    - Retrieve-first when docs are empty for legal questions
    - Slot/choices guardrail:
        * consume pending_slot BEFORE greeting
        * auto-create pending_slot by parsing numbered list in LAST assistant message if missing
        * robust numeric selection parsing: "2", "1 2", "1,2", "1-3", "12"(if <=9)
        * allow free-text topic if no numeric match (fallback)
    - Satisfaction handling:
        * If user says "โอเค/ขอบคุณ" => acknowledge + do not repeat answer; offer next-topic choices
    - If user says "ไม่รู้/มีประเภทอะไรบ้าง" after we asked "ประเภท" -> provide real choices + pending_slot
    """

    persona_id = "practical"

    # greeting / noise (typo-tolerant)
    _EN_GREET_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo+)\b", re.IGNORECASE)
    _TH_WATDEE_RE = re.compile(r"^\s*หวัด[^\s]{0,6}", re.IGNORECASE)
    _TH_SAWASDEE_RE = re.compile(r"^\s*สว[^\s]{0,8}ดี", re.IGNORECASE)
    _TH_DEE_RE = re.compile(r"^\s*ดี(?:ครับ|คับ|ค่ะ|คะ|งับ|จ้า|จ้ะ|ค่า)?", re.IGNORECASE)

    # satisfaction / close signals
    _THANKS_RE = re.compile(r"(ขอบคุณ|ขอบใจ|thx|thanks)\b", re.IGNORECASE)
    _OK_RE = re.compile(r"^\s*(โอเค|ok|okay|รับทราบ|เข้าใจแล้ว|ได้เลย|เรียบร้อย)\s*(ครับ|คับ|ค่ะ|คะ)?\s*$", re.IGNORECASE)

    # legal signal -> force retrieve first if no docs
    _LEGAL_SIGNAL_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ทะเบียนพาณิชย์|ภาษี|vat|ภพ\.?20|สรรพากร|เทศบาล|สำนักงานเขต|สุขาภิบาล|กรม|ค่าธรรมเนียม|เอกสาร|ขั้นตอน|บทลงโทษ|ประกาศ|พ\.ร\.บ|เปิดร้าน)",
        re.IGNORECASE,
    )

    _DONT_KNOW_RE = re.compile(r"^\s*(ไม่รู้|ไม่แน่ใจ|ไม่ทราบ|งง|แล้วแต่|อะไรก็ได้)\s*$")
    _ASK_TYPES_RE = re.compile(r"(มีประเภทอะไรบ้าง|ประเภทอะไรบ้าง|มีแบบไหนบ้าง|มีอะไรบ้าง)\s*$")

    # parse numbered options from text
    _NUM_OPTION_LINE_RE = re.compile(r"^\s*(\d{1,2})\)\s*(.+?)\s*$")

    # user selection looks like numbers / separators
    _LIKELY_SELECTION_RE = re.compile(r"^\s*[\d\s,/-]+\s*$")

    def __init__(self, retriever):
        self.retriever = retriever
        self._topic_menu_cache: Optional[List[str]] = None  # ["...", ...]
        self._init_llm()

    def _init_llm(self):
        model_name = getattr(conf, "OPENROUTER_MODEL_PRACTICAL", conf.OPENROUTER_MODEL)
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=getattr(conf, "TEMPERATURE_PRACTICAL", 0.2),
            max_tokens=getattr(conf, "MAX_TOKENS_PRACTICAL", 650),
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    # --------------------------
    # Normalization / detectors
    # --------------------------
    def _normalize_for_intent(self, s: str) -> str:
        t = (s or "").strip().lower()
        t = re.sub(r"[!！?？。,，]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        # compress repeated chars: "หวัดีี" -> "หวัดี"
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        return t

    def _looks_like_greeting(self, s: str) -> bool:
        raw = (s or "").strip()
        if not raw:
            return True
        t = self._normalize_for_intent(raw)
        if self._EN_GREET_RE.match(t):
            return True
        if self._TH_WATDEE_RE.match(t):
            return True
        if self._TH_SAWASDEE_RE.match(t):
            return True
        # "ดีค่ะ/ดีคับ/ดีงับ" (but not "ดีไหม")
        if self._TH_DEE_RE.match(t) and ("ไหม" not in t and "?" not in t):
            return True
        return False

    def _looks_like_legal_question(self, s: str) -> bool:
        t = self._normalize_for_intent(s)
        return bool(self._LEGAL_SIGNAL_RE.search(t))

    def _looks_like_satisfaction(self, s: str) -> bool:
        t = self._normalize_for_intent(s)
        if not t:
            return False
        return bool(self._THANKS_RE.search(t) or self._OK_RE.match(t))

    # --------------------------
    # Slot + choices helpers (production guardrails)
    # --------------------------
    def _format_numbered_options(self, options: List[str], max_items: int = 9) -> str:
        opts = [str(x).strip() for x in (options or []) if str(x).strip()]
        opts = opts[:max_items]
        return "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(opts)])

    def _parse_selection_numbers(self, user_text: str, options_count: int) -> List[int]:
        t = (user_text or "").strip().lower()
        if not t:
            return []

        # range 1-3
        m = re.search(r"\b(\d+)\s*-\s*(\d+)\b", t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            out = [x for x in range(a, b + 1) if 1 <= x <= options_count]
            seen, uniq = set(), []
            for x in out:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return uniq

        # "12" -> [1,2] only if <=9 options
        if options_count <= 9 and re.fullmatch(r"\d{2,}", t):
            out = []
            for ch in t:
                n = int(ch)
                if 1 <= n <= options_count:
                    out.append(n)
            seen, uniq = set(), []
            for x in out:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return uniq

        nums = re.findall(r"\d+", t)
        out = []
        for s2 in nums:
            n = int(s2)
            if 1 <= n <= options_count:
                out.append(n)
        seen, uniq = set(), []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _extract_numbered_options(self, text: str, max_items: int = 9) -> List[str]:
        """
        Parse options from assistant message text like:
            1) xxx
            2) yyy
        """
        if not text:
            return []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pairs: List[Tuple[int, str]] = []
        for ln in lines:
            m = self._NUM_OPTION_LINE_RE.match(ln)
            if not m:
                continue
            idx = int(m.group(1))
            label = (m.group(2) or "").strip()
            if idx <= 0 or not label:
                continue
            pairs.append((idx, label))
        if not pairs:
            return []
        pairs.sort(key=lambda x: x[0])
        return [lbl for _, lbl in pairs][:max_items]

    def _infer_slot_key_from_question(self, question: str) -> str:
        q = self._normalize_for_intent(question)
        if "ตารางเมตร" in q or "พื้นที่" in q:
            return "area_size"
        if "บุคคลธรรมดา" in q or "นิติบุคคล" in q or "นิติ" in q:
            return "entity_type"
        if "จังหวัด" in q or "เขต" in q or "เทศบาล" in q or "พื้นที่" in q:
            return "location_scope"
        if "ขายสุรา" in q or "แอลกอฮอล์" in q:
            return "alcohol_business"
        if "topic" in q or "หัวข้อ" in q:
            return "topic"
        return "choice"

    def _maybe_recover_pending_slot_from_last_bot(self, state: ConversationState, user_text: str) -> None:
        """
        Critical fix:
        - If context/pending_slot got lost (common in CLI loops),
          recover it from LAST assistant message when user replies like "1" / "1,2" / "1-3".
        """
        ctx = state.context or {}
        pending = ctx.get("pending_slot")
        if isinstance(pending, dict) and pending.get("options"):
            return  # already have pending_slot

        # only attempt when user likely selecting
        if not user_text or not self._LIKELY_SELECTION_RE.match(user_text.strip()):
            return

        last_bot = next((m.get("content", "") for m in reversed(state.messages or []) if m.get("role") == "assistant"), "")
        opts = self._extract_numbered_options(last_bot)
        if not opts:
            return

        slot_key = "topic" if "เกี่ยวกับร้านอาหาร" in (last_bot or "") else self._infer_slot_key_from_question(last_bot)
        ctx["pending_slot"] = {"key": slot_key, "options": opts, "allow_multi": False}
        state.context = ctx

    def _consume_pending_slot_from_user(self, state: ConversationState, user_text: str) -> Optional[str]:
        ctx = state.context or {}
        pending = ctx.get("pending_slot")
        if not isinstance(pending, dict):
            return None

        key = (pending.get("key") or "").strip()
        options = pending.get("options")
        allow_multi = bool(pending.get("allow_multi", False))

        if not key:
            ctx.pop("pending_slot", None)
            state.context = ctx
            return "FILLED"

        slots = ctx.setdefault("slots", {})
        if key in slots and slots[key] not in (None, "", [], {}):
            ctx.pop("pending_slot", None)
            state.context = ctx
            return "FILLED"

        low = self._normalize_for_intent(user_text)

        if isinstance(options, list) and options and allow_multi:
            if re.search(r"(ทั้งหมด|all\b|ทุกข้อ|ทุกอย่าง)", low):
                slots[key] = [str(x) for x in options]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

        if isinstance(options, list) and options:
            nums = self._parse_selection_numbers(user_text, options_count=len(options))
            chosen = [str(options[n - 1]) for n in nums if 1 <= n <= len(options)]

            if chosen:
                slots[key] = chosen if allow_multi else chosen[0]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            # contains match (best-effort)
            matched = [str(opt) for opt in options if str(opt) and str(opt) in user_text]
            if matched:
                slots[key] = matched if allow_multi else matched[0]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            # IMPORTANT: if user responded free-text (not numeric), accept it as value
            # This makes "free text" work even when options exist.
            if user_text.strip() and not self._LIKELY_SELECTION_RE.match(user_text.strip()):
                slots[key] = user_text.strip()
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            return "INVALID"

        # free-text slot
        if user_text.strip():
            slots[key] = user_text.strip()
            ctx.pop("pending_slot", None)
            state.context = ctx
            return "FILLED"

        return "INVALID"

    # --------------------------
    # Real topic menu from metadata (NO hallucination)
    # --------------------------
    def _build_topic_menu_from_corpus(self) -> List[str]:
        q = "ใบอนุญาต เปิดร้านอาหาร ภาษี VAT จดทะเบียนพาณิชย์ สุขาภิบาลอาหาร"
        docs = self._retrieve_docs(q)

        freq: Dict[str, int] = {}
        for d in docs:
            md = d.get("metadata", {}) or {}
            lt = str(md.get("license_type") or "").strip()
            if lt:
                freq[lt] = freq.get(lt, 0) + 1

        items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        menu = [k for k, _ in items][:6]

        if not menu:
            freq2: Dict[str, int] = {}
            for d in docs:
                md = d.get("metadata", {}) or {}
                dep = str(md.get("department") or "").strip()
                if dep:
                    freq2[dep] = freq2.get(dep, 0) + 1
            items2 = sorted(freq2.items(), key=lambda x: x[1], reverse=True)
            menu = [k for k, _ in items2][:6]

        return menu

    def _get_topic_menu(self, state: ConversationState) -> List[str]:
        if self._topic_menu_cache:
            return self._topic_menu_cache

        cached = (state.context or {}).get("topic_menu")
        if isinstance(cached, list) and all(isinstance(x, str) for x in cached) and cached:
            self._topic_menu_cache = cached
            return cached

        menu = self._build_topic_menu_from_corpus()
        if not menu:
            menu = ["ใบอนุญาต/การเปิดร้าน", "ภาษี/VAT", "จดทะเบียนพาณิชย์", "สุขาภิบาลอาหาร"]

        state.context = state.context or {}
        state.context["topic_menu"] = menu
        self._topic_menu_cache = menu
        return menu

    def _reply_greeting_with_choices(self, state: ConversationState) -> str:
        menu = self._get_topic_menu(state)
        state.context["pending_slot"] = {"key": "topic", "options": menu, "allow_multi": False}
        msg = "สวัสดีครับ อยากให้ช่วยเรื่องไหนเกี่ยวกับร้านอาหารครับ\n" + self._format_numbered_options(menu)
        return msg

    def _reply_satisfaction(self, state: ConversationState) -> str:
        menu = self._get_topic_menu(state)
        state.context["pending_slot"] = {"key": "topic", "options": menu, "allow_multi": False}
        msg = "ยินดีครับ ถ้าจะให้ช่วยต่อ เลือกหัวข้อได้เลยครับ\n" + self._format_numbered_options(menu)
        return msg

    # --------------------------
    # LLM + retrieval
    # --------------------------
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
            "analysis": "Parse error",
            "action": "ask",
            "execution": {"question": "ต้องการให้ช่วยเรื่องอะไรเกี่ยวกับร้านอาหารครับ?", "context_update": {}},
        }

    def _retrieve_docs(self, query: str) -> List[Dict[str, Any]]:
        docs = self.retriever.invoke(query)
        results: List[Dict[str, Any]] = []
        for d in docs[: getattr(conf, "RETRIEVAL_TOP_K", 20)]:
            results.append(
                {"content": (getattr(d, "page_content", "") or "")[:600], "metadata": getattr(d, "metadata", {}) or {}}
            )
        return results

    def _debug_log(self, stage: str, query: str, docs_json: List[Dict[str, Any]]):
        try:
            n = len(docs_json)
            top1 = docs_json[0] if n else {}
            top_meta = top1.get("metadata", {}) if isinstance(top1, dict) else {}
            top_content = (top1.get("content", "") if isinstance(top1, dict) else "")[:120]
            print(f"[DEBUG:{stage}] query={query!r} docs_count={n}")
            if n:
                print(f"[DEBUG:{stage}] top1_metadata_keys={list(top_meta.keys())[:8]}")
                print(f"[DEBUG:{stage}] top1_content_120={top_content!r}")
        except Exception:
            pass

    # --------------------------
    # ENTRYPOINT
    # --------------------------
    def handle(self, state: ConversationState, user_input: str, _internal: bool = False) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        state.persona_id = self.persona_id

        user_text = (user_input or "").strip()
        norm = self._normalize_for_intent(user_text)

        # (0) RECOVER pending_slot from last_bot if context was lost and user replied with numbers
        if not _internal:
            self._maybe_recover_pending_slot_from_last_bot(state, user_text)

        # (1) consume pending_slot BEFORE anything else
        filled_topic_value = None
        if (not _internal) and user_text:
            pending_status = self._consume_pending_slot_from_user(state, user_text)

            # if user picked a topic, lock it into context to avoid looping back to greeting
            if pending_status == "FILLED":
                slots = (state.context or {}).get("slots", {}) or {}
                if isinstance(slots, dict) and "topic" in slots and slots.get("topic"):
                    filled_topic_value = str(slots.get("topic")).strip()
                    state.context["topic"] = filled_topic_value  # stable key for downstream logic

            if pending_status == "INVALID":
                pending = state.context.get("pending_slot") or {}
                options = pending.get("options") if isinstance(pending, dict) else None
                if isinstance(options, list) and options:
                    msg = "ตอบเป็นตัวเลขได้ครับ\n" + self._format_numbered_options(options)
                    state.messages.append({"role": "user", "content": user_input})
                    state.messages.append({"role": "assistant", "content": msg})
                    state.round = int(getattr(state, "round", 0) or 0) + 1
                    return state, msg
            # if FILLED -> continue (do not return)

        # (2) satisfaction shortcut (no repeat)
        if (not _internal) and self._looks_like_satisfaction(user_text):
            state.messages.append({"role": "user", "content": user_input})
            msg = self._reply_satisfaction(state)
            state.messages.append({"role": "assistant", "content": msg})
            state.round = int(getattr(state, "round", 0) or 0) + 1
            return state, msg

        # (3) greeting shortcut (no LLM) — BUT do not trigger if user just selected a topic
        if (not _internal) and self._looks_like_greeting(user_text) and not filled_topic_value:
            state.messages.append({"role": "user", "content": user_input})
            msg = self._reply_greeting_with_choices(state)
            state.messages.append({"role": "assistant", "content": msg})
            state.round = int(getattr(state, "round", 0) or 0) + 1
            return state, msg

        # store user msg
        if not _internal:
            state.messages.append({"role": "user", "content": user_input})

        # last bot (AFTER storing user, so exclude it)
        last_bot = next((m["content"] for m in reversed(state.messages[:-1]) if m["role"] == "assistant"), "")

        # (4) If last bot asked "ประเภท..." and user says "ไม่รู้/มีประเภทอะไรบ้าง" -> show REAL choices deterministically
        if (not _internal) and ("ประเภท" in (last_bot or "")) and (
            self._DONT_KNOW_RE.match(norm) or self._ASK_TYPES_RE.search(norm)
        ):
            menu = self._get_topic_menu(state)
            state.context["pending_slot"] = {"key": "topic", "options": menu, "allow_multi": False}
            msg = "ได้ครับ เลือกหัวข้อหลักที่ใกล้เคียงที่สุดก่อนนะครับ\n" + self._format_numbered_options(menu)
            state.messages.append({"role": "assistant", "content": msg})
            state.round = int(getattr(state, "round", 0) or 0) + 1
            return state, msg

        # (5) If user just selected topic, prefer retrieve-first on that topic to drive next step deterministically
        # This prevents LLM from re-asking greeting/menu again.
        if (not _internal) and filled_topic_value:
            q = filled_topic_value
            state.current_docs = self._retrieve_docs(q)
            tmp = [{"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})} for d in state.current_docs[:1]]
            self._debug_log("post_retrieve(topic)", query=q, docs_json=tmp)
            return self.handle(state, "__auto_post_retrieve__", _internal=True)

        # (6) RETRIEVE-FIRST: legal question but no docs yet -> retrieve deterministically
        if (not _internal) and not (state.current_docs or []) and self._looks_like_legal_question(user_text):
            state.current_docs = self._retrieve_docs(user_text)
            tmp = [{"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})} for d in state.current_docs[:1]]
            self._debug_log("post_retrieve", query=user_text, docs_json=tmp)
            return self.handle(state, "__auto_post_retrieve__", _internal=True)

        recent_msgs = state.messages[-12:]

        docs_json = []
        for d in (state.current_docs or [])[:12]:
            md = d.get("metadata", {}) or {}
            docs_json.append(
                {"metadata": {k: ("" if v is None else str(v)) for k, v in md.items()}, "content": (d.get("content", "") or "")[:250]}
            )

        self._debug_log("pre_llm", query=user_text, docs_json=docs_json)

        prompt = f"""
{SYSTEM_PROMPT_PRACTICAL}

USER INPUT:
{user_input}

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

        # retrieve
        if action == "retrieve":
            q = exec_.get("query") or user_text or user_input
            state.current_docs = self._retrieve_docs(q)
            tmp = [{"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})} for d in state.current_docs[:1]]
            self._debug_log("post_retrieve", query=q, docs_json=tmp)
            return self.handle(state, "__auto_post_retrieve__", _internal=True)

        # ask
        if action == "ask":
            question = (exec_.get("question") or "ต้องการให้ช่วยเรื่องอะไรเกี่ยวกับร้านอาหารครับ?").strip()

            # apply context update first
            if isinstance(exec_.get("context_update", {}), dict):
                state.context.update(exec_.get("context_update", {}))

            # If question contains numbered options but LLM forgot pending_slot -> create it.
            pending = state.context.get("pending_slot")
            if not isinstance(pending, dict):
                parsed_opts = self._extract_numbered_options(question)
                if parsed_opts:
                    slot_key = self._infer_slot_key_from_question(question)
                    state.context["pending_slot"] = {"key": slot_key, "options": parsed_opts, "allow_multi": False}

            # enforce numbered choices if pending_slot exists
            pending2 = state.context.get("pending_slot")
            if isinstance(pending2, dict):
                options = pending2.get("options")
                if isinstance(options, list) and options:
                    menu = self._format_numbered_options(options)
                    if "1)" not in question and any(str(opt) in question for opt in options) is False:
                        question = question.rstrip() + "\n" + menu

            state.messages.append({"role": "assistant", "content": question})
            state.round = int(getattr(state, "round", 0) or 0) + 1
            return state, question

        # answer
        if action == "answer":
            ans = (exec_.get("answer") or "").strip()
            if not ans:
                ans = "ตอนนี้ยังไม่พบข้อมูลที่ยืนยันได้ในเอกสารครับ (บอกพื้นที่/ประเภทกิจการคร่าว ๆ ได้ไหมครับ?)"
            state.messages.append({"role": "assistant", "content": ans})
            state.context["phase"] = None
            state.round = 0
            return state, ans

        fallback = "ผมยังไม่เข้าใจครับ บอกหัวข้อที่อยากรู้เกี่ยวกับร้านอาหารหน่อยครับ"
        state.messages.append({"role": "assistant", "content": fallback})
        return state, fallback