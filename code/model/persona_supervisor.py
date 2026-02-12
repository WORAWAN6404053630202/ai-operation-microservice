# code/model/persona_supervisor.py
"""
Persona Supervisor (Final - Patched)

Production entrypoint for persona-based conversational AI.

Key improvements:
- Deterministic-first routing for:
  - "ขอสั้น/สรุป/กระชับ" => propose switch to practical
  - "ขอละเอียด/ตามกฎหมาย/เชิงลึก" => propose switch to academic
  - "โหมดไหน/ตอนนี้โหมดอะไร" => answer status immediately (robust to typos)
  - "เปลี่ยนโหมด" without target => ask one choice question (academic/practical)
- Robust confirmation classifier (yes/no) with fuzzy + LLM fallback
- Fix: prevent duplicated assistant messages when replaying after switch confirmation

PATCH (Greeting hardening):
- Add supervisor-level robust greeting/noise gate to prevent greeting from leaking into LLM pipeline
- Support Thai typo/elongation greetings: "สวสัสดี", "สวสัดี", "ดีค่าาา", "หวัดดีจ้า", "ดีงับบบ"
- Support English greetings: hi/hello/hey + good morning/afternoon/evening (with minor typos after normalization)

PATCH (Ambiguous greeting LLM fallback):
- If deterministic greeting gate does NOT match but input looks "ambiguous greeting/noise"
  -> call lightweight LLM classifier to decide is_greeting with confidence threshold.
- Designed to be cheap/rare: only fires on short, low-signal messages.

PATCH (Supervisor greeting reply):
- If greeting/noise is detected => supervisor replies directly (always "สวัสดีครับ ..."),
  no menu, no "เลือกข้อไหน", no routing to persona, no retrieval.
- Escalate on repeated greetings by giving inline examples only (not numbered choices).
"""

from __future__ import annotations

from typing import Tuple, Callable, Optional, Dict, Any
import re
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import conf
from model.conversation_state import ConversationState
from utils.persona_switcher import resolve_persona_switch
from utils.persona_profile import (
    normalize_persona_id,
    build_strict_profile,
    apply_persona_profile,
)

from model.persona_academic import AcademicPersonaService
from model.persona_practical import PracticalPersonaService


class PersonaSupervisor:
    """
    Central orchestrator for persona-based conversation.
    """

    # --------------------------
    # Male ending guardrail (supervisor-level)
    # --------------------------
    _DUAL_ENDING_RE = re.compile(r"(ครับ\s*/\s*ค่ะ|ค่ะ\s*/\s*ครับ)")
    _FEMALE_ENDING_TOKEN_RE = re.compile(r"(?<![ก-๙])ค่ะ(?![ก-๙])")

    def _normalize_male(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return t
        t = self._DUAL_ENDING_RE.sub("ครับ", t)
        t = self._FEMALE_ENDING_TOKEN_RE.sub("ครับ", t)
        return t

    # --------------------------
    # Deterministic detectors
    # --------------------------
    _MODE_STATUS_HINTS = ("โหมด", "mode", "persona", "บุคลิก")
    _MODE_STATUS_Q = re.compile(
        r"(ตอนนี้|ตอนนี้เรา|ตอนนี้บอท|บอทตอนนี้|อยู่|เป็น)\s*.*(โหมด|mode|persona|บุคลิก).*"
        r"|^(โหมด|mode|persona|บุคลิก)\s*(อะไร|ไหน|ไร|หยัง|ไหนอะ|ไหนครับ|ไหนคะ)?\s*\??$",
        re.IGNORECASE,
    )

    _SWITCH_VERBS = ("เปลี่ยน", "สลับ", "ปรับ", "ขอเปลี่ยน", "ขอสลับ", "ขอปรับ")
    _SWITCH_MARKERS = ("โหมด", "mode", "persona", "บุคลิก")
    _TARGET_ACADEMIC_HINTS = (
        "ละเอียด",
        "เชิงลึก",
        "วิชาการ",
        "ตามกฎหมาย",
        "อ้างอิงข้อกฎหมาย",
        "อธิบายละเอียด",
        "ขอแบบละเอียด",
        "ขอแบบทางการ",
    )
    _TARGET_PRACTICAL_HINTS = (
        "สั้น",
        "สั้นๆ",
        "กระชับ",
        "สรุป",
        "สรุปสั้น",
        "เอาแบบสั้น",
        "เอาแบบสรุป",
        "เช็คลิสต์",
        "เป็นข้อๆ",
        "เร็วๆ",
    )

    # --------------------------
    # Greeting / noise detector (SUPERVISOR-LEVEL HARD GATE)
    # --------------------------
    # NOTE: We intentionally over-capture "greeting-only" inputs to keep them out of LLM.
    # PATCH: removed \b so strings like "helloooooolololoooo" still match as greeting prefix
    _EN_GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo+)", re.IGNORECASE)
    _EN_GOOD_TIME_RE = re.compile(
        r"^\s*good\s+(morning|afternoon|evening|night)\b", re.IGNORECASE
    )

    # Thai greeting robust patterns:
    # - "สวัสดี" with typos like "สวสัสดี", "สวสัดี", "สวสัี" (common omissions/swap)
    # - "หวัดดี", "หวัดดีจ้า", elongated
    _TH_SAWASDEE_FUZZY_RE = re.compile(r"^\s*สว[^\s]{0,6}ดี", re.IGNORECASE)
    _TH_WATDEE_RE = re.compile(r"^\s*หวัดดี", re.IGNORECASE)

    # "ดีค่ะ/ดีคับ/ดีงับ/ดีจ้า/ดีค่า" (incl. elongated)
    _TH_DEE_RE = re.compile(
        r"^\s*ดี(?:ครับ|คับ|ค่ะ|คะ|งับ|จ้า|จ้ะ|ค่า)?", re.IGNORECASE
    )

    # numeric selection patterns like "1", "1 2", "1,2", "1-3", "12"
    _LIKELY_SELECTION_RE = re.compile(r"^\s*[\d\s,/-]+\s*$")

    # ---- Ambiguous greeting/noise heuristics + LLM fallback knobs (NEW) ----
    # PATCH: allow longer single-token greetings like "helloooooolololoooo"
    _AMBIG_MAX_CHARS = 80
    _AMBIG_MAX_WORDS = 4
    _GREET_LLM_CONF_TH = 0.70

    _QUESTION_MARKERS_RE = re.compile(
        r"(\?|\bไหม\b|หรือไม่|หรือเปล่า|ยังไง|ทำไง|อย่างไร|ได้ไหม|ควร|ต้อง|คืออะไร)",
        re.IGNORECASE,
    )
    _LEGAL_SIGNAL_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ทะเบียนพาณิชย์|ภาษี|vat|ภพ\.?20|สรรพากร|เทศบาล|สำนักงานเขต|สุขาภิบาล|กรม|ค่าธรรมเนียม|เอกสาร|ขั้นตอน|บทลงโทษ|ประกาศ|พ\.ร\.บ)",
        re.IGNORECASE,
    )
    _MODE_SWITCH_SIGNAL_RE = re.compile(
        r"(โหมด|mode|persona|บุคลิก|เปลี่ยน|สลับ|ปรับ|academic|practical)",
        re.IGNORECASE,
    )
    # single token/noise-like text
    _NOISE_ONLY_RE = re.compile(r"^(?:[อ-ฮะ-์]+|[a-z]+|[0-9]+|[!?.]+)$", re.IGNORECASE)

    # ---- Supervisor greeting reply policy (NEW) ----
    _GREET_EXAMPLES = ("ใบอนุญาตเปิดร้าน", "ภาษี/VAT", "จดทะเบียนพาณิชย์", "สุขาภิบาลอาหาร")

    def _handle_greeting_reply(
        self, state: ConversationState, user_input: str
    ) -> Tuple[ConversationState, str]:
        """
        Supervisor-level greeting reply:
        - Always greet back ("สวัสดีครับ ...")
        - No menu, no "เลือกข้อไหน", no routing to persona, no retrieval.
        - Escalate on repeated greetings by giving inline examples only (not numbered).
        """
        state.context = state.context or {}
        streak = int(state.context.get("greet_streak", 0) or 0) + 1
        state.context["greet_streak"] = streak

        if streak <= 1:
            msg = "สวัสดีครับ ต้องการให้ช่วยเรื่องอะไรเกี่ยวกับร้านอาหารครับ"
        elif streak <= 3:
            ex = " / ".join(self._GREET_EXAMPLES)
            msg = f"สวัสดีครับ อยากคุยเรื่องไหนเป็นหลักครับ เช่น {ex}"
        else:
            ex = " / ".join(self._GREET_EXAMPLES)
            msg = f"สวัสดีครับ บอกหัวข้อได้เลยครับ (เช่น {ex})"

        msg = self._normalize_male(msg)
        state.add_assistant_message(msg)
        state.last_action = "greeting_reply"
        return state, msg
    
    def _handle_greeting_persona_aware(
        self, state: ConversationState, user_input: str
    ) -> Tuple[ConversationState, str]:
        """
        Greeting is handled deterministically (NO LLM).
        - academic: greet + 1 short question (no menu)
        - practical: greet + numbered menu (reuse practical deterministic method)
        """
        # reset/track streak if you want (optional)
        state.context = state.context or {}
        state.context["greet_streak"] = int(state.context.get("greet_streak", 0) or 0) + 1

        # Ensure persona/profile synced before replying
        self._sync_persona_and_profile(state)

        if state.persona_id == "practical":
            # Use practical deterministic menu (NO LLM)
            msg = self._practical._reply_greeting_with_choices(state)
            msg = self._normalize_male(msg)
            state.add_assistant_message(msg)
            state.last_action = "greeting_practical_menu"
            return state, msg

        # academic (default)
        msg = "สวัสดีครับ ต้องการปรึกษาเรื่องใดเกี่ยวกับร้านอาหารครับ"
        msg = self._normalize_male(msg)
        state.add_assistant_message(msg)
        state.last_action = "greeting_academic"
        return state, msg    

    def _is_blank(self, s: str) -> bool:
        return not (s or "").strip()

    def _normalize_for_intent(self, s: str) -> str:
        t = (s or "").strip().lower()
        # collapse spaces and remove some punctuation/elongation
        t = re.sub(r"[!！?？。,，]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        # compress repeated chars (ไทย/อังกฤษ) for robustness
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        return t.strip()

    def _looks_like_greeting_noise_ambiguous(self, s: str) -> bool:
        """
        Ambiguous greeting/noise heuristic.
        IMPORTANT PATCH:
        - Do NOT treat numeric selection like "1", "1,2", "1-3" as greeting/noise.
        """
        raw = (s or "").strip()
        if not raw:
            return True

        # If it's numeric selection, it's NOT greeting/noise (let persona handle pending_slot)
        if self._LIKELY_SELECTION_RE.match(raw):
            return False

        t = self._normalize_for_intent(raw)

        if len(t) > self._AMBIG_MAX_CHARS:
            return False

        if self._QUESTION_MARKERS_RE.search(t):
            return False
        if self._LEGAL_SIGNAL_RE.search(t):
            return False
        if self._MODE_SWITCH_SIGNAL_RE.search(t):
            return False

        words = [w for w in t.split(" ") if w]
        if len(words) > self._AMBIG_MAX_WORDS:
            return False

        # IMPORTANT PATCH: remove digits-only from noise bucket
        # (เดิม _NOISE_ONLY_RE จับ [0-9]+ ด้วย ทำให้ "1" โดน)
        if re.fullmatch(r"[0-9]+", t):
            return False

        if self._NOISE_ONLY_RE.match(t):
            return True

        return len(t) <= 14

    def _default_greet_llm_call(self) -> Callable[[str], dict]:
        """
        NEW: Lightweight LLM classifier for greeting/noise.
        Called ONLY when deterministic greeting gate does not match but input is ambiguous.
        Returns: {"is_greeting": bool, "confidence": 0..1}
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=48,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = (
                "จัดประเภทข้อความผู้ใช้ว่าเป็น 'greeting/noise' หรือไม่\n"
                "เกณฑ์: greeting/noise คือทักทาย/คุยเล่น/เสียงอุทาน/คำสั้นๆที่ไม่ใช่คำถามกฎหมาย\n"
                "ถ้าไม่แน่ใจให้ is_greeting=false และ confidence=0.0\n"
                "ตอบ JSON เท่านั้น:\n"
                '{ "is_greeting": true/false, "confidence": 0.0 }\n'
                f"ข้อความผู้ใช้: {user_text}"
            )
            try:
                text = (llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
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

    def _looks_like_greeting_noise(self, s: str) -> bool:
        """
        Greeting/noise gate to prevent greeting from leaking into persona LLM pipeline.

        IMPORTANT PATCH:
        - Do NOT classify numeric selections like "1", "1,2", "1-3" as greeting/noise.
        """
        raw = (s or "").strip()
        if not raw:
            return True

        # If it's numeric selection, it's NOT greeting/noise (let persona handle pending_slot)
        if self._LIKELY_SELECTION_RE.match(raw):
            return False

        # keep very short NON-numeric as greeting/noise
        if len(raw) <= 2:
            return True

        t = self._normalize_for_intent(raw)

        # English (det)
        if self._EN_GREETING_RE.match(t) or self._EN_GOOD_TIME_RE.match(t):
            return True

        # Thai (det)
        if self._TH_WATDEE_RE.match(t):
            return True
        if self._TH_SAWASDEE_FUZZY_RE.match(t):
            return True

        # Prevent false positive for "ดีไหม/ดีหรือไม่/ดีไหมครับ" etc. (question markers)
        if self._TH_DEE_RE.match(t) and not self._QUESTION_MARKERS_RE.search(t):
            return True

        # ---- LLM fallback only for ambiguous cases ----
        if self._looks_like_greeting_noise_ambiguous(t) and getattr(self, "llm_greet_call", None):
            res = self.llm_greet_call(raw)
            try:
                conf_llm = float(res.get("confidence", 0.0) or 0.0)
            except Exception:
                conf_llm = 0.0
            if conf_llm >= self._GREET_LLM_CONF_TH:
                return bool(res.get("is_greeting", False))

        return False

    def _looks_like_mode_status_query(self, s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if (
            "โหมด" not in t
            and "mode" not in t.lower()
            and "persona" not in t.lower()
            and "บุคลิก" not in t
        ):
            return False
        return bool(self._MODE_STATUS_Q.search(t))

    def _infer_style_switch_target(self, s: str) -> Optional[str]:
        """
        Policy requested:
        - If user asks "short/summary" => treat as request to switch persona to practical.
        - If user asks "detailed/legal-deep" => treat as request to switch persona to academic.
        Returns target persona_id or None.
        """
        t = (s or "").strip().lower()
        if not t:
            return None

        hit_ac = any(h in t for h in self._TARGET_ACADEMIC_HINTS)
        hit_pr = any(h in t for h in self._TARGET_PRACTICAL_HINTS)

        # If both appear, ambiguous -> None (ask user explicitly)
        if hit_ac and hit_pr:
            return None
        if hit_ac:
            return "academic"
        if hit_pr:
            return "practical"
        return None

    def _looks_like_switch_without_target(self, s: str) -> bool:
        t = (s or "").strip().lower()
        if not t:
            return False
        if any(v in t for v in self._SWITCH_VERBS) and any(m in t for m in self._SWITCH_MARKERS):
            # but no explicit target token
            if self._infer_style_switch_target(t) is None and not re.search(
                r"\b(academic|practical)\b|วิชาการ|ละเอียด|สั้น|กระชับ", t
            ):
                return True
        return False

    # --------------------------
    # Confirmation classifier
    # --------------------------
    _YES_TOKENS = (
        "ใช่",
        "ใช่ครับ",
        "ใช่คับ",
        "ใช่ค่ะ",
        "ใข่",  # common typo
        "ไช่",  # common typo
        "ช่าย",  # slang/typo
        "ใช",  # truncated
        "ครับ",
        "คับ",
        "ค่ะ",
        "โอเค",
        "โอเคครับ",
        "โอเคค่ะ",
        "ตกลง",
        "ได้",
        "ได้เลย",
        "เอาเลย",
        "จัดมา",
        "จัดไป",
        "เปลี่ยน",
        "เปลี่ยนเลย",
        "สลับ",
        "yes",
        "yeah",
        "yep",
        "y",
        "ok",
        "okay",
    )
    _NO_TOKENS = (
        "ไม่",
        "ไม่ครับ",
        "ไม่คับ",
        "ไม่ค่ะ",
        "ไม่เอา",
        "ไม่ต้อง",
        "ยังไม่",
        "ไม่เปลี่ยน",
        "ไม่สลับ",
        "ยกเลิก",
        "ช่างมัน",
        "no",
        "nope",
        "n",
        "cancel",
    )

    def _classify_yes_no_det(self, user_text: str) -> Dict[str, Any]:
        """
        Robust deterministic classifier:
        - supports extra words: "ใช่ เปลี่ยนเลย", "ไม่เอาอะ", "โอเคครับ เปลี่ยน"
        - supports slight typos/elongation via normalization
        """
        t = self._normalize_for_intent(user_text)

        # hard blocks
        if not t:
            return {"yes": False, "no": False, "confidence": 0.0, "method": "empty"}

        def _has_any(tokens) -> bool:
            for tok in tokens:
                if not tok:
                    continue
                if tok in t:
                    return True
            return False

        yes = _has_any(self._YES_TOKENS)
        no = _has_any(self._NO_TOKENS)

        # conflict -> unclear
        if yes and no:
            return {"yes": False, "no": False, "confidence": 0.0, "method": "conflict"}

        if yes:
            return {"yes": True, "no": False, "confidence": 0.86, "method": "det_contains"}
        if no:
            return {"yes": False, "no": True, "confidence": 0.86, "method": "det_contains"}

        # weak patterns
        if re.search(r"\b(เอา|ได้|โอเค|ตกลง)\b", t):
            return {"yes": True, "no": False, "confidence": 0.72, "method": "weak_yes"}
        if re.search(r"\b(ไม่เอา|ไม่ต้อง|ยกเลิก|ช่างมัน)\b", t):
            return {"yes": False, "no": True, "confidence": 0.72, "method": "weak_no"}

        return {"yes": False, "no": False, "confidence": 0.0, "method": "unclear"}

    def __init__(
        self,
        retriever,
        llm_switch_call: Optional[Callable[[str], dict]] = None,
    ):
        self.retriever = retriever

        # persona services
        self._academic = AcademicPersonaService(retriever=retriever)
        self._practical = PracticalPersonaService(retriever=retriever)

        # LLM fallback for switch intent + confirmation (optional but recommended)
        self.llm_switch_call = llm_switch_call or self._default_switch_llm_call()
        self.llm_confirm_call = self._default_confirm_llm_call()

        # NEW: LLM fallback for ambiguous greeting/noise only
        self.llm_greet_call = self._default_greet_llm_call()

    def _default_switch_llm_call(self) -> Callable[[str], dict]:
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=256,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(prompt: str) -> dict:
            text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
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

    def _default_confirm_llm_call(self) -> Callable[[str], dict]:
        """
        LLM confirmation classifier to avoid strict yes/no loops.
        Returns: {"yes": bool, "no": bool, "confidence": 0..1}

        Patch:
        - reduce max_tokens to lower chance of "length limit reached" on some models
        - shorten prompt
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=64,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = (
                "ตัดสินว่า ข้อความนี้คือยืนยันการเปลี่ยนโหมด (yes) หรือปฏิเสธ (no) หรือไม่ชัดเจน\n"
                "ตอบ JSON เท่านั้น:\n"
                '{ "yes": true/false, "no": true/false, "confidence": 0.0 }\n'
                f"ข้อความผู้ใช้: {user_text}"
            )
            try:
                text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
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

    # ------------------------------------------------------------------
    # Public API (ENTRYPOINT)
    # ------------------------------------------------------------------

    def handle(
        self,
        state: ConversationState,
        user_input: str,
    ) -> Tuple[ConversationState, str]:

        state.context = state.context or {}

        # 0) one-time greeting
        if not state.context.get("did_greet"):
            greet = "สวัสดีครับ ผมคือ Restbiz ผู้ช่วยด้านกฎหมายร้านอาหาร ต้องการให้ช่วยอะไรครับ"
            greet = self._normalize_male(greet)
            state.add_assistant_message(greet)
            state.context["did_greet"] = True

            # CLI uses empty input to just greet; stop here.
            if self._is_blank(user_input):
                state.last_action = "greet"
                return state, greet

        # 0.5) if blank input after greeting, keep clean
        if self._is_blank(user_input):
            msg = "ต้องการให้ช่วยเรื่องอะไรครับ?"
            msg = self._normalize_male(msg)
            state.add_assistant_message(msg)
            state.last_action = "blank"
            return state, msg

        # 1) Handle pending persona pick (no target specified)
        if state.context.get("awaiting_persona_pick"):
            return self._handle_persona_pick(state, user_input)

        # 2) Handle pending persona switch confirmation
        if state.context.get("awaiting_persona_confirmation"):
            return self._handle_persona_confirmation(state, user_input)

        # 3) Ensure persona + strict profile are synced
        self._sync_persona_and_profile(state)

        # 3.5) HARD GATE: greeting/noise should never enter mode/switch/LLM intent flows
        if self._looks_like_greeting_noise(user_input):
            return self._handle_greeting_persona_aware(state, user_input)

        # 4) Deterministic: mode status query
        if self._looks_like_mode_status_query(user_input):
            msg = f"ตอนนี้อยู่โหมด {state.persona_id} ครับ"
            msg = self._normalize_male(msg)
            state.add_assistant_message(msg)
            state.last_action = "mode_status"
            return state, msg

        '''# 5) Deterministic: style-switch policy (your requested behavior)
        inferred = self._infer_style_switch_target(user_input)
        if inferred and inferred != state.persona_id:
            # ask confirmation for switch
            target_pid = normalize_persona_id(inferred)
            state.context["pending_persona"] = target_pid
            state.context["pending_replay_user_input"] = user_input
            state.context["awaiting_persona_confirmation"] = True

            msg = f"ต้องการเปลี่ยนเป็นโหมด {target_pid} ใช่ไหมครับ"
            msg = self._normalize_male(msg)
            state.add_assistant_message(msg)
            state.last_action = "persona_switch_confirm_style"
            return state, msg'''

        # 6) Deterministic: explicit "switch" but no target
        if self._looks_like_switch_without_target(user_input):
            state.context["awaiting_persona_pick"] = True
            msg = "ต้องการเปลี่ยนเป็นโหมดไหนครับ 1) practical 2) academic"
            msg = self._normalize_male(msg)
            state.add_assistant_message(msg)
            state.last_action = "persona_pick_ask"
            return state, msg

        # 7) Detect persona switch intent (existing resolver: slash command, explicit tokens, etc.)
        if user_input != "__auto_post_retrieve__":
            new_pid, cleaned, meta = resolve_persona_switch(
                user_text=user_input,
                llm_call_json=self.llm_switch_call,
                llm_conf_threshold=0.85,
            )

            if new_pid:
                target_pid = normalize_persona_id(new_pid)
                state.context["pending_persona"] = target_pid
                state.context["pending_replay_user_input"] = cleaned or user_input
                state.context["awaiting_persona_confirmation"] = True

                msg = f"ต้องการเปลี่ยนเป็นโหมด {target_pid} ใช่ไหมครับ"
                msg = self._normalize_male(msg)
                state.add_assistant_message(msg)
                state.last_action = "persona_switch_confirm"
                return state, msg

        # 8) Route
        if state.persona_id == "academic":
            return self._academic.handle(state, user_input)
        return self._practical.handle(state, user_input)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_persona_pick(
        self,
        state: ConversationState,
        user_input: str,
    ) -> Tuple[ConversationState, str]:
        """
        Handle when user asked "เปลี่ยนโหมด" but didn't specify target.
        We ask one choice question and parse reply robustly.
        """
        t = self._normalize_for_intent(user_input)

        # Accept numeric picks
        if re.search(r"\b1\b", t):
            target_pid = "practical"
        elif re.search(r"\b2\b", t):
            target_pid = "academic"
        else:
            # Accept text hints
            target_pid = None
            if any(h in t for h in self._TARGET_PRACTICAL_HINTS) or "practical" in t:
                target_pid = "practical"
            if any(h in t for h in self._TARGET_ACADEMIC_HINTS) or "academic" in t:
                # if both matched, ambiguous -> ask again
                if target_pid and target_pid != "academic":
                    target_pid = None
                else:
                    target_pid = "academic"

        if not target_pid:
            msg = "เลือกโหมด 1) practical 2) academic ครับ"
            msg = self._normalize_male(msg)
            state.add_assistant_message(msg)
            return state, msg

        state.context.pop("awaiting_persona_pick", None)

        # ask confirmation (consistent policy)
        state.context["pending_persona"] = normalize_persona_id(target_pid)
        state.context["pending_replay_user_input"] = ""
        state.context["awaiting_persona_confirmation"] = True

        msg = f"ต้องการเปลี่ยนเป็นโหมด {target_pid} ใช่ไหมครับ"
        msg = self._normalize_male(msg)
        state.add_assistant_message(msg)
        return state, msg

    def _handle_persona_confirmation(
        self,
        state: ConversationState,
        user_input: str,
    ) -> Tuple[ConversationState, str]:

        target_pid = state.context.get("pending_persona")
        if not target_pid:
            state.context.pop("awaiting_persona_confirmation", None)
            msg = "ระบบไม่สามารถเปลี่ยนโหมดได้ในขณะนี้ครับ"
            msg = self._normalize_male(msg)
            state.add_assistant_message(msg)
            return state, msg

        # 1) deterministic robust classify
        det = self._classify_yes_no_det(user_input)
        confirm_yes = bool(det.get("yes", False))
        confirm_no = bool(det.get("no", False))

        # 2) LLM fallback if unclear
        if not confirm_yes and not confirm_no and self.llm_confirm_call:
            res = self.llm_confirm_call(user_input)
            try:
                conf_llm = float(res.get("confidence", 0.0) or 0.0)
            except Exception:
                conf_llm = 0.0
            if conf_llm >= 0.70:
                confirm_yes = bool(res.get("yes", False))
                confirm_no = bool(res.get("no", False))

        if confirm_yes:
            state.persona_id = normalize_persona_id(target_pid)
            self._sync_persona_and_profile(state)

            replay_input = state.context.get("pending_replay_user_input") or ""

            # cleanup
            state.context.pop("pending_persona", None)
            state.context.pop("pending_replay_user_input", None)
            state.context.pop("awaiting_persona_confirmation", None)

            prefix = f"ตอนนี้เป็นโหมด {state.persona_id} แล้วครับ"
            prefix = self._normalize_male(prefix)

            # replay immediately, but FIX duplication:
            # persona.handle already appends assistant message; we will replace it with a single combined message.
            if replay_input.strip():
                before = len(state.messages)
                if state.persona_id == "academic":
                    _, reply = self._academic.handle(state, replay_input)
                else:
                    _, reply = self._practical.handle(state, replay_input)

                # remove last assistant message produced by persona, then add combined once
                if len(state.messages) > before and state.messages[-1].get("role") == "assistant":
                    state.messages.pop()

                final_reply = prefix + "\n" + (reply or "").strip()
                final_reply = self._normalize_male(final_reply)
                state.add_assistant_message(final_reply)
                state.last_action = "persona_switch_applied"
                return state, final_reply

            # no replay
            state.add_assistant_message(prefix)
            state.last_action = "persona_switch_applied"
            return state, prefix

        if confirm_no:
            msg = "เข้าใจแล้วครับ ใช้โหมดเดิมต่อ"
            msg = self._normalize_male(msg)

            state.context.pop("pending_persona", None)
            state.context.pop("pending_replay_user_input", None)
            state.context.pop("awaiting_persona_confirmation", None)

            state.add_assistant_message(msg)
            state.last_action = "persona_switch_cancelled"
            return state, msg

        msg = f"ยืนยันอีกครั้งครับ จะเปลี่ยนเป็นโหมด {target_pid} ใช่ไหม (ตอบ: ใช่/ไม่)"
        msg = self._normalize_male(msg)
        state.add_assistant_message(msg)
        return state, msg

    def _sync_persona_and_profile(self, state: ConversationState) -> None:
        pid = normalize_persona_id(state.persona_id)
        state.persona_id = pid

        # ensure context exists + set derived persona_id here (single source sync point)
        state.context = state.context or {}
        state.context["persona_id"] = state.persona_id

        current_profile = state.strict_profile or {}
        state.strict_profile = build_strict_profile(
            persona_id=pid,
            current=current_profile,
        )

        state.context = apply_persona_profile(
            state.context,
            state.strict_profile,
        )