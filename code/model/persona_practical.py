# code/model/persona_practical.py
import json
import logging
import re
from typing import Tuple, Dict, Any, List, Optional, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import conf
from model.conversation_state import ConversationState
from utils.llm_call import llm_invoke
from utils.prompts_practical import SYSTEM_PROMPT as SYSTEM_PROMPT_PRACTICAL

_LOG = logging.getLogger("restbiz.practical")

# P0: practical policy "last gate"
try:
    from utils.practical_lint import enforce_practical_policy  # type: ignore
except Exception:  # pragma: no cover
    enforce_practical_policy = None  # fallback handled in _apply_practical_lint


class PracticalPersonaService:
    """
    Practical Persona (Agentic, fast, short)

    Supervisor-owned menu contract (IMPORTANT):
    - Practical MUST always be able to "consume choices" when pending_slot exists:
        - numeric (1,2,3), ranges (1-3), multi (1,3), digit-pack (123), exact label, and free text.
    - Practical MUST NOT hijack Supervisor's main menu/greeting:
        - If state.context["supervisor_owns_menu"] is True:
            - do NOT render greeting/topic menu
            - do NOT set/override pending_slot for topic menu
    - Pending-slot recovery remains opt-in and additionally gated by owner to prevent hijack.
    """

    persona_id = "practical"

    _EN_GREET_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo+)\b", re.IGNORECASE)
    _TH_WATDEE_RE = re.compile(r"^\s*หวัด[^\s]{0,6}", re.IGNORECASE)
    _TH_SAWASDEE_RE = re.compile(r"^\s*สว[^\s]{0,8}ดี", re.IGNORECASE)
    _TH_DEE_RE = re.compile(r"^\s*ดี(?:ครับ|คับ|ค่ะ|คะ|งับ|จ้า|จ้ะ|ค่า)?", re.IGNORECASE)

    _THANKS_RE = re.compile(
        r"(ขอบคุณ|ขอบใจ|ขอบพระคุณ|ขอบคุณมาก|ขอบคุณนะ|thx|thanks|thank you)",
        re.IGNORECASE,
    )
    _OK_RE = re.compile(
        r"^\s*(โอเค|ok|okay|รับทราบ|เข้าใจแล้ว|เข้าใจ|ได้เลย|เรียบร้อย|เคลียร์|เคลียแล้ว|พอแล้ว|พอครับ|พอค่ะ|ครบแล้ว|got\s*it|clear)\s*(ครับ|คับ|ค่ะ|คะ)?\s*$",
        re.IGNORECASE,
    )

    _LEGAL_SIGNAL_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ทะเบียนพาณิชย์|ภาษี|vat|ภพ\.?20|สรรพากร|เทศบาล|สำนักงานเขต|สุขาภิบาล|กรม|ค่าธรรมเนียม|เอกสาร|ขั้นตอน|บทลงโทษ|ประกาศ|พ\.ร\.บ|เปิดร้าน|ประกันสังคม|กองทุน)",
        re.IGNORECASE,
    )

    _DONT_KNOW_RE = re.compile(r"^\s*(ไม่รู้|ไม่แน่ใจ|ไม่ทราบ|งง|แล้วแต่|อะไรก็ได้)\s*$")
    _ASK_TYPES_RE = re.compile(r"(มีประเภทอะไรบ้าง|ประเภทอะไรบ้าง|มีแบบไหนบ้าง|มีอะไรบ้าง)\s*$")

    _NUM_OPTION_LINE_RE = re.compile(r"^\s*(\d{1,2})\)\s*(.+?)\s*$")
    _LIKELY_SELECTION_RE = re.compile(r"^\s*[\d\s,/-]+\s*$")

    # --------------------------
    # Topic menu sanitation (STRICT)
    # --------------------------
    _TOPIC_MIN_LEN = 3
    _TOPIC_MAX_LEN = 52
    _TOPIC_REJECT_IF_HAS_NEWLINE = True

    def _sanitize_topic_label(self, s: str) -> str:
        raw = (s or "")
        t = raw.strip()
        if not t:
            return ""
        if self._TOPIC_REJECT_IF_HAS_NEWLINE and ("\n" in raw or "\r" in raw):
            return ""
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > self._TOPIC_MAX_LEN:
            return ""
        if "ตามกฎหมาย" in t or "มีสิทธิ" in t or "ผู้ประกอบกิจการ" in t:
            return ""
        return t

    # --------------------------
    # Owner/menu guards (NEW)
    # --------------------------
    def _supervisor_owns_menu(self, state: ConversationState) -> bool:
        ctx = state.context or {}
        return bool(ctx.get("supervisor_owns_menu", False))

    def _get_last_bot_owner(self, state: ConversationState) -> str:
        ctx = state.context or {}
        owner = (ctx.get("last_bot_owner") or "").strip().lower()
        return owner

    def _set_last_bot_owner(self, state: ConversationState, owner: str) -> None:
        state.context = state.context or {}
        state.context["last_bot_owner"] = (owner or "").strip()

    # --------------------------
    # Safe append (dedupe)
    # --------------------------
    def _append_user_once(self, state: ConversationState, content: str) -> None:
        if content is None:
            return
        c = str(content)
        if not c.strip():
            return
        if (
            state.messages
            and state.messages[-1].get("role") == "user"
            and (state.messages[-1].get("content") or "").strip() == c.strip()
        ):
            return
        state.messages.append({"role": "user", "content": c})

    def _append_assistant(self, state: ConversationState, content: str) -> None:
        """
        P0: Dedupe assistant to prevent recursion/forced flows from duplicating turns.
        Also tag last_bot_owner='practical' for safe pending-slot recovery gating.
        """
        c = "" if content is None else str(content)
        if not c.strip():
            c = c.strip()
        if state.messages:
            last = state.messages[-1]
            if last.get("role") == "assistant" and (last.get("content") or "").strip() == c.strip():
                return
        state.messages.append({"role": "assistant", "content": c})
        self._set_last_bot_owner(state, "practical")

    # --------------------------
    # Greeting prefix (Practical fallback)
    # --------------------------
    _GREET_PREFIX_FALLBACKS: Dict[str, str] = {
        "greet": "สวัสดีครับ อยากให้ช่วยเรื่องไหนเกี่ยวกับร้านอาหารครับ",
        "thanks": "ยินดีครับ อยากไปต่อหัวข้อไหนครับ",
        "smalltalk": "แล้วอยากให้ช่วยเรื่องไหนเกี่ยวกับร้านอาหารครับ",
        "blank": "อยากให้ช่วยเรื่องไหนเกี่ยวกับร้านอาหารครับ",
    }

    def _default_greet_llm_call(self) -> Callable[[str, List[str], int], dict]:
        """
        Returns JSON: { "prefix": "..." }
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.35,
            max_tokens=120,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(kind: str, menu: List[str], greet_streak: int) -> dict:
            menu_preview = ", ".join([str(x) for x in (menu or [])[:6]])
            prompt = (
                "หน้าที่: สร้างประโยคทักทาย/ตอบรับแบบมนุษย์ สำหรับบอทกฎหมายร้านอาหารไทย (โหมด practical)\n"
                "เงื่อนไข:\n"
                "- ห้ามสั่งผู้ใช้ว่า 'เลือก.../พิมพ์.../กด...'\n"
                "- ให้สั้น 1-2 ประโยค และถาม 1 คำถามสั้นๆ\n"
                "- โทน: practical (ตรง กระชับ มืออาชีพ)\n"
                "- ถ้า greet_streak >= 2 พยายามเปลี่ยนคำเล็กน้อยไม่ให้ซ้ำ\n"
                "- ต้องลงท้ายด้วย 'ครับ'\n"
                "ตอบเป็น JSON เท่านั้น: {\"prefix\": \"...\"}\n"
                f"kind: {kind}\n"
                f"greet_streak: {greet_streak}\n"
                f"ตัวอย่างหัวข้อในระบบ (เพื่ออ้างอิงคำ): {menu_preview}\n"
            )
            try:
                text = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/greet").content.strip()
            except Exception:
                return {}

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _pick_greet_prefix(self, kind: str, menu: List[str], greet_streak: int) -> str:
        kind2 = kind if kind in {"greet", "thanks", "smalltalk", "blank"} else "greet"
        try:
            res = self.llm_greet_call(kind2, menu, int(greet_streak or 0))
        except Exception:
            res = {}

        prefix = ""
        if isinstance(res, dict):
            prefix = str(res.get("prefix") or "").strip()

        if prefix:
            prefix = re.sub(r"\s+", " ", prefix).strip()
            if len(prefix) > 140:
                prefix = ""
            if re.search(r"(เลือก|พิมพ์|กด)", prefix) and re.search(r"(ข้อ|หัวข้อ|ด้านล่าง)", prefix):
                prefix = ""

        if not prefix:
            prefix = self._GREET_PREFIX_FALLBACKS.get(kind2, self._GREET_PREFIX_FALLBACKS["greet"])

        return prefix.strip()

    # --------------------------
    # Practical policy lint (P0 last gate)
    # --------------------------
    _MULTI_Q_SPLIT_RE = re.compile(r"[?？]\s*")
    _META_TALK_RE = re.compile(
        r"(ในฐานะ(ของ)?(บอท|ผู้ช่วย)|ฉันจะ|ผมจะ|ขออนุญาต|ขออธิบายว่า|ระบบนี้|นโยบาย|policy|ตามที่คุณขอ|ผมไม่สามารถ|ฉันไม่สามารถ|\bdocuments\b)",
        re.IGNORECASE,
    )

    # Strip standalone "documents" lines (LLM prompt bleed-through)
    _DOCUMENTS_LINE_RE = re.compile(r"(?m)^[ \t]*documents[ \t]*$", re.IGNORECASE)

    def _fallback_single_question(self, text: str) -> str:
        t = re.sub(r"\s+", " ", (text or "")).strip()
        if not t:
            return "อยากให้ช่วยเรื่องไหนเกี่ยวกับร้านอาหารครับ?"

        t = self._META_TALK_RE.sub("", t).strip()
        t = re.sub(r"\s+", " ", t).strip()

        if "?" in t or "？" in t:
            first = re.split(r"[?？]", t, maxsplit=1)[0].strip()
            if first:
                t = first

        t = re.sub(r"(\d+\)|[-•])\s*", "", t).strip()

        if not re.search(r"(ไหม|หรือ|ยังไง|อย่างไร|อะไร|มั้ย|ได้ไหม|ต้องการ|อยาก)", t):
            t = "อยากให้ช่วยเรื่องไหนเกี่ยวกับร้านอาหารครับ?"
        else:
            if not t.endswith("ครับ"):
                t = t.rstrip(" .") + "ครับ"
        return t

    def _fallback_practical_answer(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return "ตอนนี้ยังไม่พบข้อมูลที่ยืนยันได้ในเอกสารครับ"

        t = self._META_TALK_RE.sub("", t).strip()
        # Remove standalone "documents" lines (LLM prompt bleed-through)
        t = self._DOCUMENTS_LINE_RE.sub("", t).strip()
        # Remove "ทั้งหมด" menu-option lines that leaked into answer content
        t = re.sub(r"(?m)^[ \t]*\d*[.)]\s*ทั้งหมด\s*(ครับ|ค่ะ|นะ|นะครับ|นะคะ)?\s*$", "", t)
        t = re.sub(r"(?m)^[ \t]*ทั้งหมด\s*(ครับ|ค่ะ|นะ|นะครับ|นะคะ)?\s*$", "", t)
        # Remove fee section when value is zero/free — no value to show user
        _FREE_FEE = r"(?:ไม่มีค่าธรรมเนียม|ไม่เสียค่าธรรมเนียม|ไม่มี|ฟรี|0\s*บาท)"
        t = re.sub(
            rf"(?m)^\d+\)\s*ค่าธรรมเนียม[^\n]*\n(?:[ \t]*[•\-*]?\s*{_FREE_FEE}[^\n]*\n?)*",
            "",
            t,
        )
        # Preserve newlines — only collapse horizontal whitespace (spaces/tabs)
        t = re.sub(r"[ \t]+", " ", t)
        # Insert newlines before numbered sections (1) or 1. format) and bullet points if missing
        t = re.sub(r"(?<!\n)\s+(\d+[).])\s*", r"\n\1 ", t)
        t = re.sub(r"(?<!\n)\s+([-•*])\s+", r"\n\1 ", t)
        t = "\n".join(ln.strip() for ln in t.split("\n") if ln.strip())

        if "?" in t or "？" in t:
            t = re.split(r"[?？]", t, maxsplit=1)[0].strip()

        if not t.endswith("ครับ"):
            last_line = (t.split("\n")[-1] if "\n" in t else t).strip()
            if re.search(r"[a-zA-Z0-9/._\-]$", last_line):
                # Ends with URL/path/non-Thai — add ครับ on separate line to avoid appending to URL
                t = t + "\nครับ"
            else:
                t = t.rstrip(" .") + "ครับ"
        return t

    def _apply_practical_lint(self, text: str, kind: str) -> str:
        t = (text or "").strip()
        if not t:
            return t

        if kind in {"menu", "greet"}:
            t2 = self._META_TALK_RE.sub("", t).strip()
            return t2 or t

        if kind == "answer":
            # Answers skip strict length policy (which would replace long answers with a question).
            # Only apply meta-talk cleanup + newline normalization via _fallback_practical_answer.
            return self._fallback_practical_answer(t)

        # For "ask" only: apply full practical policy enforcement
        if callable(enforce_practical_policy):
            try:
                out = enforce_practical_policy(t)
                if isinstance(out, str):
                    t = out.strip() or t
                elif isinstance(out, tuple) and len(out) == 2:
                    new_t, lint_meta = out  # enforce_practical_policy returns (text, meta_dict)
                    if isinstance(new_t, str) and new_t.strip():
                        t = new_t.strip()
                    if isinstance(lint_meta, dict) and lint_meta.get("ok") is False:
                        raise ValueError("practical_lint_failed")
                elif isinstance(out, dict):
                    new_t = out.get("text") or out.get("output") or out.get("result")
                    if isinstance(new_t, str) and new_t.strip():
                        t = new_t.strip()
                    ok = out.get("ok")
                    if ok is False:
                        raise ValueError("practical_lint_failed")
            except Exception:
                pass

        if kind == "ask":
            # Split question text from numbered option lines before applying single-question lint.
            # _fallback_single_question collapses newlines + strips \d+\) markers — would destroy options.
            lines = t.splitlines()
            opts_start = next(
                (i for i, ln in enumerate(lines) if re.match(r"^\d+[).]", ln.strip())),
                len(lines),
            )
            question_text = " ".join(ln.strip() for ln in lines[:opts_start] if ln.strip())
            opts_text = "\n".join(lines[opts_start:])
            cleaned_q = self._fallback_single_question(question_text)
            return (cleaned_q + "\n" + opts_text).strip() if opts_text else cleaned_q

        return t

    # --------------------------
    # Phase 3 config (unchanged)
    # --------------------------
    _PHASE3_MENU_HEADER = "ตอนนี้มีข้อมูลครบระดับหนึ่งแล้วครับ คุณอยากดูหัวข้อไหนก่อน?"
    _PHASE3_SLOT_KEY = "detail_section"
    _PHASE3_ALL = "ทั้งหมด"

    _PHASE3_CANONICAL = [
        "ขั้นตอนการดำเนินการ",
        "เอกสารที่ต้องใช้",
        "ค่าธรรมเนียม",
        "ระยะเวลาดำเนินการ",
        "ช่องทางยื่นคำขอ / หน่วยงาน",
        "ข้อกำหนดทางกฎหมาย และข้อบังคับ",
        "ฟอร์มเอกสารตัวจริง",
        "ทั้งหมด",
    ]

    _SECTION_SIGNALS: Dict[str, Dict[str, Any]] = {
        "ขั้นตอนการดำเนินการ": {
            "meta_keys": ["operation_steps", "operation_step", "steps", "procedure", "ขั้นตอนการดำเนินการ"],
            "content_keywords": ["ขั้นตอน", "วิธีดำเนินการ", "ลำดับ", "ยื่นคำขอ"],
        },
        "เอกสารที่ต้องใช้": {
            "meta_keys": [
                "identification_documents",
                "documents",
                "required_documents",
                "เอกสาร ยืนยันตัวตน",
                "เอกสารที่ต้องใช้",
            ],
            "content_keywords": ["เอกสาร", "สำเนา", "หลักฐาน", "แนบ", "ใบคำขอ"],
        },
        "ค่าธรรมเนียม": {
            "meta_keys": ["fees", "fee", "ค่าธรรมเนียม"],
            "content_keywords": ["ค่าธรรมเนียม", "บาท", "ชำระเงิน", "ค่าบริการ", "ราคา"],
        },
        "ระยะเวลาดำเนินการ": {
            "meta_keys": ["operation_duration", "duration", "ระยะเวลา การดำเนินการ", "ระยะเวลาดำเนินการ"],
            "content_keywords": ["ระยะเวลา", "วันทำการ", "ภายใน", "ใช้เวลา"],
        },
        "ช่องทางยื่นคำขอ / หน่วยงาน": {
            "meta_keys": ["service_channel", "department", "หน่วยงาน", "ช่องทางการ ให้บริการ", "channel"],
            "content_keywords": ["ช่องทาง", "ยื่น", "หน่วยงาน", "สำนักงานเขต", "เทศบาล", "ออนไลน์", "เว็บไซต์", "เคาน์เตอร์"],
        },
        "ข้อกำหนดทางกฎหมาย และข้อบังคับ": {
            "meta_keys": ["legal_regulatory", "law", "regulation", "ข้อกำหนดทางกฎหมาย และข้อบังคับ", "บทลงโทษ"],
            "content_keywords": ["กฎหมาย", "ข้อกำหนด", "ประกาศ", "พ.ร.บ", "บทลงโทษ", "ข้อบังคับ"],
        },
        "ฟอร์มเอกสารตัวจริง": {
            "meta_keys": ["restaurant_ai_document", "form", "template", "ฟอร์ม", "เอกสาร AI ร้านอาหาร"],
            "content_keywords": ["แบบฟอร์ม", "ดาวน์โหลด", "ฟอร์ม", "ตัวอย่างเอกสาร", "คำขอ"],
        },
    }

    _PHASE3_MIN_SECTIONS = 2
    _PHASE3_ANSWER_CHAR_THRESHOLD = 520
    _PHASE3_ANSWER_LINE_THRESHOLD = 10

    # --------------------------
    # Retrieval reuse/new-topic heuristic (unchanged)
    # --------------------------
    _TOKEN_SPLIT_RE = re.compile(r"[\s/,\-–—|]+", re.UNICODE)
    _FOLLOWUP_SHORT_RE = re.compile(
        r"^(แล้ว(ไง|ล่ะ)?|แล้ว(เอกสาร|ขั้นตอน|ค่าธรรมเนียม)?|ต่อไปล่ะ|มีอะไรบ้าง|ขอ(เอกสาร|ขั้นตอน|ค่าธรรมเนียม|ระยะเวลา|ช่องทาง))\s*$"
    )
    # Continuation/follow-up questions — should always get a direct answer, never Phase 3 menu
    _CONTINUATION_RE = re.compile(
        r"(อีกไหม|อีกบ้าง|อีกมั้ย|ควรทำอะไรอีก|ต้องทำอะไรอีก|อะไรอีก|มีอะไรอีก"
        r"|แล้วต้อง|แล้วควร|แล้วล่ะ|เพิ่มเติม|ยังมีอะไร|ต้องทำด้วย|อีกด้วย"
        r"|ต่อไปต้อง|ขั้นต่อไป|หลังจากนั้น|นอกจากนี้|อื่นๆ.*ต้อง|ควรมี)",
        re.IGNORECASE,
    )

    def __init__(self, retriever):
        self.retriever = retriever
        self._topic_menu_cache: Optional[List[str]] = None
        self.llm_greet_call = self._default_greet_llm_call()
        self._init_llm()

    def _init_llm(self):
        model_name = getattr(conf, "OPENROUTER_MODEL_PRACTICAL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=getattr(conf, "TEMPERATURE_PRACTICAL", 0.2),
            max_tokens=getattr(conf, "MAX_TOKENS_PRACTICAL", 650),
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    # --------------------------
    # Normalization / detectors
    # --------------------------
    def _normalize_for_intent(self, s: str) -> str:
        t = (s or "").strip().lower()
        t = re.sub(r"[!！?？。,，]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
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

    def _tokenize_loose(self, s: str) -> List[str]:
        t = self._normalize_for_intent(s)
        toks = [x.strip() for x in self._TOKEN_SPLIT_RE.split(t) if x and x.strip()]
        return [x for x in toks if len(x) >= 2]

    def _topic_overlap_ratio(self, a: str, b: str) -> float:
        sa = set(self._tokenize_loose(a))
        sb = set(self._tokenize_loose(b))
        if not sa or not sb:
            return 0.0
        inter = len(sa.intersection(sb))
        union = len(sa.union(sb))
        return (inter / union) if union else 0.0

    def _is_short_followup(self, user_text: str) -> bool:
        t = (user_text or "").strip()
        if not t:
            return True
        n = self._normalize_for_intent(t)
        toks = self._tokenize_loose(n)
        if len(toks) <= 3 and len(n) <= 18:
            return True
        if self._FOLLOWUP_SHORT_RE.match(n):
            return True
        return False

    def _should_retrieve_new_topic(self, state: ConversationState, user_text: str) -> bool:
        q = (user_text or "").strip()
        if not q:
            return False

        has_docs = bool(getattr(state, "current_docs", None))
        if not has_docs:
            return True

        last_q = (getattr(state, "last_retrieval_query", None) or "").strip()
        if not last_q:
            return True

        if self._is_short_followup(q):
            return False

        overlap = self._topic_overlap_ratio(last_q, q)
        if self._looks_like_legal_question(q) and overlap < 0.22:
            return True

        return False

    # --------------------------
    # Slot + choices helpers (unchanged)
    # --------------------------
    def _format_numbered_options(self, options: List[str], max_items: int = 9) -> str:
        opts = [str(x).strip() for x in (options or []) if str(x).strip()]
        opts = opts[:max_items]
        return "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(opts)])

    def _parse_selection_numbers(self, user_text: str, options_count: int) -> List[int]:
        t = (user_text or "").strip().lower()
        if not t:
            return []

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
        if self._PHASE3_MENU_HEADER[:10] in question:
            return self._PHASE3_SLOT_KEY
        return "choice"

    # --------------------------
    # Pending-slot recovery (OPT-IN + OWNER-GATED)  ✅ FIX
    # --------------------------
    def _maybe_recover_pending_slot_from_last_bot(self, state: ConversationState, user_text: str) -> None:
        """
        Recovery is disabled by default to prevent hijack.
        Enable only if explicitly requested:
          state.context["allow_practical_pending_recover"] = True

        PLUS owner-gate:
          - only recover if state.context["last_bot_owner"] == "practical"
        """
        ctx = state.context or {}
        if not bool(ctx.get("allow_practical_pending_recover", False)):
            return

        if (ctx.get("last_bot_owner") or "").strip().lower() != "practical":
            return

        pending = ctx.get("pending_slot")
        if isinstance(pending, dict) and pending.get("options"):
            return

        if not user_text or not self._LIKELY_SELECTION_RE.match(user_text.strip()):
            return

        last_bot = next((m.get("content", "") for m in reversed(state.messages or []) if m.get("role") == "assistant"), "")
        opts = self._extract_numbered_options(last_bot)
        if not opts:
            return

        slot_key = (
            self._PHASE3_SLOT_KEY
            if self._PHASE3_MENU_HEADER in (last_bot or "")
            else ("topic" if "เกี่ยวกับร้านอาหาร" in (last_bot or "") else self._infer_slot_key_from_question(last_bot))
        )
        allow_multi = True if slot_key == self._PHASE3_SLOT_KEY else False
        ctx["pending_slot"] = {"key": slot_key, "options": opts, "allow_multi": allow_multi}
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
                slots[key] = [str(x) for x in options if str(x).strip() and str(x).strip() != self._PHASE3_ALL]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

        if isinstance(options, list) and options:
            nums = self._parse_selection_numbers(user_text, options_count=len(options))
            chosen = [str(options[n - 1]) for n in nums if 1 <= n <= len(options)]

            if chosen:
                if self._PHASE3_ALL in chosen and key == self._PHASE3_SLOT_KEY:
                    slots[key] = [str(x) for x in options if str(x).strip() and str(x).strip() != self._PHASE3_ALL]
                else:
                    slots[key] = chosen if allow_multi else chosen[0]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            matched = [str(opt) for opt in options if str(opt) and str(opt) in user_text]
            if matched:
                if self._PHASE3_ALL in matched and key == self._PHASE3_SLOT_KEY:
                    slots[key] = [str(x) for x in options if str(x).strip() and str(x).strip() != self._PHASE3_ALL]
                else:
                    slots[key] = matched if allow_multi else matched[0]
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            if key == "topic" and user_text.strip() and not self._LIKELY_SELECTION_RE.match(user_text.strip()):
                if self._looks_like_legal_question(user_text):
                    ctx.pop("pending_slot", None)
                    state.context = ctx
                    return "BYPASS"
                return "INVALID"

            if key == self._PHASE3_SLOT_KEY:
                return "INVALID"

            if user_text.strip() and not self._LIKELY_SELECTION_RE.match(user_text.strip()):
                slots[key] = user_text.strip()
                ctx.pop("pending_slot", None)
                state.context = ctx
                return "FILLED"

            return "INVALID"

        if user_text.strip():
            slots[key] = user_text.strip()
            ctx.pop("pending_slot", None)
            state.context = ctx
            return "FILLED"

        return "INVALID"

    # --------------------------
    # Topic menu from metadata (NO hallucination) + sanitation
    # --------------------------
    # H2: Multi-query pool for richer topic menu (4 queries = broader coverage)
    _TOPIC_POOL_QUERIES = [
        "ใบอนุญาต เปิดร้านอาหาร",
        "ภาษี VAT จดทะเบียนพาณิชย์",
        "สุขาภิบาลอาหาร ประกันสังคม",
        "เอกสาร ค่าธรรมเนียม ขั้นตอน",
    ]

    def _build_topic_menu_from_corpus(self) -> List[str]:
        freq: Dict[str, int] = {}

        def _add(v: Any) -> None:
            s = self._sanitize_topic_label(str(v) if v is not None else "")
            if len(s) < self._TOPIC_MIN_LEN:
                return
            freq[s] = freq.get(s, 0) + 1

        seen_doc_ids: set = set()
        for q in self._TOPIC_POOL_QUERIES:
            try:
                docs = self._retrieve_docs(q)
            except Exception:
                continue
            for d in docs:
                md = d.get("metadata", {}) or {}
                doc_id = md.get("doc_id") or md.get("row_id") or id(d)
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                _add(md.get("license_type"))
                _add(md.get("department"))
                _add(md.get("operation_topic"))

        items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        menu = [k for k, _ in items][:6]
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

    def _reply_greeting_with_choices(self, state: ConversationState, kind: str = "greet") -> str:
        """
        Practical greeting/menu renderer.

        ✅ FIX: If Supervisor owns menu, do NOT show menu / do NOT set pending_slot.
        (Still returns a short prefix so the system can respond naturally.)
        """
        state.context = state.context or {}

        menu = self._get_topic_menu(state)
        streak = int(state.context.get("greet_streak", 0) or 0) + 1
        state.context["greet_streak"] = streak
        prefix = self._pick_greet_prefix(kind=kind, menu=menu, greet_streak=streak).strip()

        if self._supervisor_owns_menu(state):
            return self._apply_practical_lint(prefix, kind="greet")

        # ✅ menu once per session
        if state.context.get("main_menu_shown"):
            return self._apply_practical_lint(prefix, kind="greet")

        state.context["pending_slot"] = {"key": "topic", "options": menu, "allow_multi": False}
        state.context["main_menu_shown"] = True

        msg = (prefix.rstrip() + "\n" + self._format_numbered_options(menu)).strip()
        return self._apply_practical_lint(msg, kind="greet")

    def _reply_satisfaction(self, state: ConversationState) -> str:
        # Uses same guard in _reply_greeting_with_choices()
        return self._reply_greeting_with_choices(state, kind="thanks")

    # --------------------------
    # LLM + retrieval
    # --------------------------
    def _call_llm_json(self, prompt: str, max_retries: int = 2, state: Optional[ConversationState] = None) -> dict:
        last_err = None
        for _ in range(max_retries):
            try:
                resp = llm_invoke(self.llm, [HumanMessage(content=prompt)], logger=_LOG, label="Practical/json", state=state)
                text = (resp.content or "").strip()

                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception as e:
                last_err = e
                continue

        if last_err:
            _LOG.warning("[Practical] LLM JSON parse failed: %s", last_err)

        return {
            "input_type": "new_question",
            "analysis": "Parse error",
            "action": "ask",
            "execution": {"question": "อยากให้ช่วยเรื่องไหนเกี่ยวกับร้านอาหารครับ?", "context_update": {}},
        }

    def _retrieve_docs(self, query: str) -> List[Dict[str, Any]]:
        docs = self.retriever.invoke(query)
        results: List[Dict[str, Any]] = []
        max_docs = getattr(conf, "LLM_DOCS_MAX_PRACTICAL", 8)
        max_chars = getattr(conf, "LLM_DOC_CHARS_PRACTICAL", 250)
        for d in docs[:max_docs]:
            results.append(
                {"content": (getattr(d, "page_content", "") or "")[:max_chars], "metadata": getattr(d, "metadata", {}) or {}}
            )
        return results

    def _debug_log(self, stage: str, query: str, docs_json: List[Dict[str, Any]]):
        if not _LOG.isEnabledFor(logging.DEBUG):
            return
        try:
            n = len(docs_json)
            top1 = docs_json[0] if n else {}
            top_meta = top1.get("metadata", {}) if isinstance(top1, dict) else {}
            top_content = (top1.get("content", "") if isinstance(top1, dict) else "")[:120]
            _LOG.debug("[DEBUG:%s] query=%r docs_count=%d", stage, query, n)
            if n:
                _LOG.debug("[DEBUG:%s] top1_metadata_keys=%s", stage, list(top_meta.keys())[:8])
                _LOG.debug("[DEBUG:%s] top1_content_120=%r", stage, top_content)
        except Exception:
            pass

    # --------------------------
    # Phase 3 helpers
    # --------------------------
    def _extract_available_phase3_sections(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Return list of canonical section labels that exist in the retrieved docs."""
        available: List[str] = []
        for label in self._PHASE3_CANONICAL:
            if label == "ทั้งหมด":
                continue
            sig = self._SECTION_SIGNALS.get(label)
            if not sig:
                continue
            meta_keys = sig.get("meta_keys") or []
            content_kws = sig.get("content_keywords") or []
            found = False
            for d in docs:
                md = d.get("metadata", {}) or {}
                for k in meta_keys:
                    v = md.get(k)
                    if v is not None and str(v).strip() and str(v).lower() != "nan":
                        found = True
                        break
                if found:
                    break
                content = d.get("content", "") or ""
                for kw in content_kws:
                    if kw and kw in content:
                        found = True
                        break
                if found:
                    break
            if found:
                available.append(label)
        return available

    def _user_requests_specific_sections(self, user_text: str) -> bool:
        """Return True if the user explicitly asked for a named section (skip phase3 menu)."""
        t = self._normalize_for_intent(user_text)
        for label in self._PHASE3_CANONICAL:
            if label == "ทั้งหมด":
                continue
            # Check shortened keywords from canonical labels
            for word in label.split():
                if len(word) >= 4 and word in t:
                    return True
        return False

    def _is_continuation_question(self, user_text: str) -> bool:
        """Return True if user is asking 'what else / what next' — should bypass Phase 3 and answer directly."""
        return bool(self._CONTINUATION_RE.search(user_text or ""))

    def _should_trigger_phase3(self, ans: str, available: List[str]) -> bool:
        """Return True if the answer is long enough and there are enough sections to offer a menu."""
        if len(available) < self._PHASE3_MIN_SECTIONS:
            return False
        if len(ans) >= self._PHASE3_ANSWER_CHAR_THRESHOLD:
            return True
        lines = [ln for ln in ans.splitlines() if ln.strip()]
        if len(lines) >= self._PHASE3_ANSWER_LINE_THRESHOLD:
            return True
        return False

    def _render_phase3_menu(self, available: List[str]) -> Tuple[str, List[str]]:
        """Render a numbered section menu. Returns (menu_text, options_list)."""
        options = list(available) + [self._PHASE3_ALL]
        lines = [self._PHASE3_MENU_HEADER]
        for i, opt in enumerate(options, 1):
            lines.append(f"{i}) {opt}")
        return "\n".join(lines), options

    # --------------------------
    # ENTRYPOINT
    # --------------------------
    def handle(self, state: ConversationState, user_input: str, _internal: bool = False) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        state.persona_id = self.persona_id

        user_text = (user_input or "").strip()
        norm = self._normalize_for_intent(user_text)

        # ✅ recovery only in non-internal, owner-gated (already inside the function)
        if not _internal:
            self._maybe_recover_pending_slot_from_last_bot(state, user_text)

        filled_topic_value = None
        bypassed_menu = False

        pending_key_before = None
        if (not _internal) and isinstance((state.context or {}).get("pending_slot"), dict):
            pending_key_before = (state.context.get("pending_slot") or {}).get("key")

        # ✅ ALWAYS allow pending_slot consumption (this is the core requirement)
        if (not _internal) and user_text:
            pending_status = self._consume_pending_slot_from_user(state, user_text)

            if pending_status == "BYPASS":
                bypassed_menu = True

            if pending_status == "FILLED":
                slots = (state.context or {}).get("slots", {}) or {}

                if isinstance(slots, dict) and "topic" in slots and slots.get("topic"):
                    filled_topic_value = str(slots.get("topic")).strip()
                    state.context["topic"] = filled_topic_value

                if pending_key_before == self._PHASE3_SLOT_KEY:
                    sel = slots.get(self._PHASE3_SLOT_KEY)
                    if isinstance(sel, str) and sel.strip():
                        state.context[self._PHASE3_SLOT_KEY] = [sel.strip()]
                    elif isinstance(sel, list) and sel:
                        state.context[self._PHASE3_SLOT_KEY] = [str(x).strip() for x in sel if str(x).strip()]

                    self._append_user_once(state, user_input)

                    forced = f"ขอข้อมูลเฉพาะหัวข้อ: {', '.join(state.context.get(self._PHASE3_SLOT_KEY, []))}"
                    return self.handle(state, forced, _internal=True)

            if pending_status == "INVALID":
                pending = state.context.get("pending_slot") or {}
                options = pending.get("options") if isinstance(pending, dict) else None
                if isinstance(options, list) and options:
                    msg = "ตอบเป็นตัวเลขได้ครับ\n" + self._format_numbered_options(options)

                    self._append_user_once(state, user_input)
                    msg = self._apply_practical_lint(msg, kind="menu")
                    self._append_assistant(state, msg)

                    state.round = int(getattr(state, "round", 0) or 0) + 1
                    return state, msg

        # --------------------------
        # Supervisor-owned menu: do not render greeting/menu here
        # --------------------------
        if (not _internal) and self._supervisor_owns_menu(state):
            # Still allow satisfaction to be treated as normal text flow; no menu injection.
            pass
        else:
            if (not _internal) and self._looks_like_satisfaction(user_text):
                self._append_user_once(state, user_input)
                msg = self._reply_satisfaction(state)
                self._append_assistant(state, msg)
                state.round = int(getattr(state, "round", 0) or 0) + 1
                return state, msg

            if (not _internal) and self._looks_like_greeting(user_text) and not filled_topic_value and not bypassed_menu:
                self._append_user_once(state, user_input)
                msg = self._reply_greeting_with_choices(state, kind="greet")
                self._append_assistant(state, msg)
                state.round = int(getattr(state, "round", 0) or 0) + 1
                return state, msg

        # ✅ DEDUPE: only append once
        if not _internal:
            self._append_user_once(state, user_input)

        last_bot = next((m["content"] for m in reversed(state.messages[:-1]) if m["role"] == "assistant"), "")

        if (not _internal) and ("ประเภท" in (last_bot or "")) and (
            self._DONT_KNOW_RE.match(norm) or self._ASK_TYPES_RE.search(norm)
        ):
            # If supervisor owns menu, don't inject topic menu here either.
            if not self._supervisor_owns_menu(state):
                menu = self._get_topic_menu(state)
                state.context["pending_slot"] = {"key": "topic", "options": menu, "allow_multi": False}
                state.context["main_menu_shown"] = True
                msg = self._format_numbered_options(menu)
                msg = self._apply_practical_lint(msg, kind="menu")
                self._append_assistant(state, msg)
                state.round = int(getattr(state, "round", 0) or 0) + 1
                return state, msg

        # If user picked topic -> always retrieve for that topic
        if (not _internal) and filled_topic_value:
            q = filled_topic_value
            state.current_docs = self._retrieve_docs(q)
            state.last_retrieval_query = q
            tmp = [
                {"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})}
                for d in state.current_docs[:1]
            ]
            self._debug_log("post_retrieve(topic)", query=q, docs_json=tmp)
            return self.handle(state, "__auto_post_retrieve__", _internal=True)

        # Practical retrieval: new-topic aware
        if (not _internal) and self._looks_like_legal_question(user_text):
            if self._should_retrieve_new_topic(state, user_text):
                state.current_docs = self._retrieve_docs(user_text)
                state.last_retrieval_query = user_text
                tmp = [
                    {"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})}
                    for d in state.current_docs[:1]
                ]
                self._debug_log("post_retrieve", query=user_text, docs_json=tmp)
                return self.handle(state, "__auto_post_retrieve__", _internal=True)

        recent_msgs = state.messages[-12:]

        docs_json = []
        for d in (state.current_docs or [])[:12]:
            md = d.get("metadata", {}) or {}
            docs_json.append(
                {
                    "metadata": {k: ("" if v is None else str(v)) for k, v in md.items()},
                    "content": (d.get("content", "") or "")[:250],
                }
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

        decision = self._call_llm_json(prompt, state=state)
        action = (decision.get("action") or "ask").strip()
        exec_ = decision.get("execution", {}) or {}

        if action == "retrieve":
            q = exec_.get("query") or user_text or user_input
            state.current_docs = self._retrieve_docs(q)
            state.last_retrieval_query = q
            tmp = [
                {"content": d.get("content", "")[:120], "metadata": d.get("metadata", {})}
                for d in state.current_docs[:1]
            ]
            self._debug_log("post_retrieve", query=q, docs_json=tmp)
            return self.handle(state, "__auto_post_retrieve__", _internal=True)

        if action == "ask":
            question = (exec_.get("question") or "อยากให้ช่วยเรื่องอะไรเกี่ยวกับร้านอาหารครับ?").strip()

            if isinstance(exec_.get("context_update", {}), dict):
                state.context.update(exec_.get("context_update", {}))

            pending = state.context.get("pending_slot")
            if not isinstance(pending, dict):
                # Prefer LLM-provided slot_options over regex extraction
                llm_opts = exec_.get("slot_options")
                if isinstance(llm_opts, list):
                    llm_opts = [str(o).strip() for o in llm_opts if str(o).strip()]
                else:
                    llm_opts = []
                parsed_opts = llm_opts or self._extract_numbered_options(question)
                if parsed_opts:
                    slot_key = self._infer_slot_key_from_question(question)
                    allow_multi = True if slot_key == self._PHASE3_SLOT_KEY else False
                    state.context["pending_slot"] = {"key": slot_key, "options": parsed_opts, "allow_multi": allow_multi}

            pending2 = state.context.get("pending_slot")
            if isinstance(pending2, dict):
                options = pending2.get("options")
                if isinstance(options, list) and options:
                    # Sanitize question: strip inline option text the LLM may have embedded
                    # e.g. "(บริษัทจำกัด (ห้างหุ้นส่วน" or "กรุงเทพฯ หรืออยู่ต่างจังหวัด"
                    q_clean = question
                    for opt in options:
                        # Remove (opt) or (opt<no-closing-paren> patterns
                        q_clean = re.sub(r'\s*\(' + re.escape(str(opt)) + r'\)?', ' ', q_clean)
                        # Remove trailing " หรือ opt" or " หรืออยู่ opt"
                        q_clean = re.sub(
                            r'\s+หรือ(?:อยู่|ไปที่|ว่า)?\s*' + re.escape(str(opt)) + r'\s*(?:ครับ|คะ|คะ)?$',
                            'ครับ', q_clean
                        )
                    q_clean = re.sub(r'\s+', ' ', q_clean).strip()
                    # Ensure ends with ครับ
                    if q_clean and not any(q_clean.endswith(e) for e in ('ครับ', 'คะ', 'คะ', '?', 'ไหม')):
                        q_clean = q_clean.rstrip('?').rstrip() + 'ครับ'
                    question = q_clean

                    # Always append numbered menu unless already numbered (Issues 2, 5)
                    if "1)" not in question and "1." not in question:
                        menu = self._format_numbered_options(options)
                        question = question.rstrip() + "\n" + menu

            question = self._apply_practical_lint(question, kind="ask")

            self._append_assistant(state, question)
            state.round = int(getattr(state, "round", 0) or 0) + 1
            return state, question

        if action == "answer":
            ans = (exec_.get("answer") or "").strip()
            if not ans:
                ans = "ตอนนี้ยังไม่พบข้อมูลที่ยืนยันได้ในเอกสารครับ"

            if isinstance(exec_.get("context_update", {}), dict):
                state.context.update(exec_.get("context_update", {}))

            ans = self._apply_practical_lint(ans, kind="answer")

            # P1: intent-aware phase3 trigger
            if not _internal:
                # These functions must exist in your full file (see note above).
                available = self._extract_available_phase3_sections(state.current_docs or [])
                requested_sections = self._user_requests_specific_sections(user_text)

                if not requested_sections and not self._is_continuation_question(user_text):
                    if self._should_trigger_phase3(ans, available):
                        menu_text, options = self._render_phase3_menu(available)
                        state.context["pending_slot"] = {
                            "key": self._PHASE3_SLOT_KEY,
                            "options": options,
                            "allow_multi": True,
                        }
                        state.context["phase3_draft_len"] = len(ans)

                        menu_text = self._apply_practical_lint(menu_text, kind="menu")
                        self._append_assistant(state, menu_text)
                        state.round = int(getattr(state, "round", 0) or 0) + 1
                        return state, menu_text

            self._append_assistant(state, ans)
            state.context["phase"] = None
            state.round = 0
            return state, ans

        fallback = "ผมยังไม่เข้าใจครับ บอกหัวข้อที่อยากรู้เกี่ยวกับร้านอาหารหน่อยครับ"
        fallback = self._apply_practical_lint(fallback, kind="ask")
        self._append_assistant(state, fallback)
        return state, fallback