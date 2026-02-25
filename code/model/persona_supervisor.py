# code/model/persona_supervisor.py
"""
Persona Supervisor (Hybrid Routing + FSM) — Option A (Supervisor owns ALL visible messages)
========================================================================================

✅ Fixes per your feedback:
1) Menu choices MUST be 5 items (target exactly 5; backfill if pool small).
2) Choices must be sampled from REAL data (metadata) and diversified (not top-only).
   - Build topic_pool once per session using MULTI broad queries (more coverage than 1 query).
3) Subsequent greeting/noise/thanks MUST show choices every time
   - but intro (name/role) only ONCE (first greeting of session).
4) Greeting should NOT repeat the same 2 options over and over
   - use per-session seed + per-greeting counter to rotate randomness deterministically.
5) Keep pending_slot active across greeting turns (do not clear on greeting).
6) Priority preserved:
   confirm/switch/intake lock > pending_slot > greeting/noise > legal routing

✅ NEW (this change request):
7) Make menu topics look like “หัวข้อ” more, by prioritizing metadata fields that correspond to:
   - ใบอนุญาต
   - การดำเนินการตามหน่วยงาน
   - หัวข้อการดำเนินการย่อย
   and de-prioritizing pure “หน่วยงาน/department” labels unless needed as backfill.
"""

from __future__ import annotations

from typing import Tuple, Callable, Optional, Dict, Any, List
import re
import json
import random
import hashlib

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import conf
from model.conversation_state import ConversationState
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
    Contract: handle(state, user_input) -> (state, reply_text)
    """

    # --------------------------
    # Thai ending normalization
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
    # Intent / Priority Matrix
    # --------------------------
    INTENT_CONFIRM_YESNO = "CONFIRM_YESNO"
    INTENT_EXPLICIT_SWITCH = "EXPLICIT_SWITCH"
    INTENT_MODE_STATUS = "MODE_STATUS"
    INTENT_ACAD_INTAKE_REPLY = "ACADEMIC_INTAKE_REPLY"
    INTENT_LEGAL_NEW = "LEGAL_NEW_QUESTION"
    INTENT_GREETING = "GREETING_SMALLTALK"
    INTENT_NOISE = "NOISE"
    INTENT_UNKNOWN = "UNKNOWN"

    # FSM states (kept for compatibility)
    S_IDLE = "S_IDLE"
    S_SWITCH_CONFIRM = "S_SWITCH_CONFIRM"
    S_PRACTICAL_ANSWER = "S_PRACTICAL_ANSWER"
    S_ACAD_INTAKE = "S_ACAD_INTAKE"
    S_ACAD_ANSWER = "S_ACAD_ANSWER"
    S_AUTO_RETURN = "S_AUTO_RETURN"

    # --------------------------
    # Deterministic detectors
    # --------------------------
    _MODE_STATUS_Q = re.compile(
        r"(ตอนนี้|ตอนนี้เรา|ตอนนี้บอท|บอทตอนนี้|อยู่|เป็น)\s*.*(โหมด|mode|persona|บุคลิก).*"
        r"|^(โหมด|mode|persona|บุคลิก)\s*(อะไร|ไหน|ไร|หยัง|ไหนอะ|ไหนครับ|ไหนคะ)?\s*\??$",
        re.IGNORECASE,
    )

    _SWITCH_VERBS = ("เปลี่ยน", "สลับ", "ปรับ", "ขอเปลี่ยน", "ขอสลับ", "ขอปรับ", "change", "ไป")
    _SWITCH_MARKERS = ("โหมด", "mode", "persona", "บุคลิก", "บอท", "bot", "ตัว")

    _TARGET_ACADEMIC_HINTS = (
        "ละเอียด",
        "เชิงลึก",
        "วิชาการ",
        "ตามกฎหมาย",
        "อ้างอิงข้อกฎหมาย",
        "อธิบายละเอียด",
        "ขอแบบละเอียด",
        "ละเอียดทั้งหมด",
        "ขอแบบละเอียดทั้งหมด",
        "เอาแบบละเอียดทั้งหมด",
        "ขยายความ",
        "ลงรายละเอียด",
        "ละเอียดขึ้น",
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

    _STYLE_LIKELY_RE = re.compile(
        r"(ขอ|ช่วย|รบกวน|เอา|อยากได้|ขอให้|ช่วยอธิบาย|ขยายความ|ลงรายละเอียด|ละเอียดขึ้น|เชิงลึก|สรุป|สั้นๆ|กระชับ)",
        re.IGNORECASE,
    )

    _SMALLTALK_RE = re.compile(
        r"(ทำอะไรอยู่|ทำไรอยู่|ว่างไหม|อยู่ไหม|เป็นไงบ้าง|เป็นไง|กินข้าวยัง|สบายดีไหม|สบายดีปะ|โอเคไหม|เหนื่อยไหม)",
        re.IGNORECASE,
    )
    _THANKS_RE = re.compile(r"(ขอบคุณ|ขอบใจ|thx|thanks)\b", re.IGNORECASE)

    _LIKELY_SELECTION_RE = re.compile(r"^\s*[\d\s,/-]+\s*$")

    _QUESTION_MARKERS_RE = re.compile(
        r"(\?|\bไหม\b|หรือไม่|หรือเปล่า|ยังไง|ทำไง|อย่างไร|ได้ไหม|ควร|ต้อง|คืออะไร)",
        re.IGNORECASE,
    )

    _LEGAL_SIGNAL_RE = re.compile(
        r"(ใบอนุญาต|จดทะเบียน|ทะเบียนพาณิชย์|ภาษี|vat|ภพ\.?20|สรรพากร|เทศบาล|สำนักงานเขต|สุขาภิบาล|กรม|ค่าธรรมเนียม|เอกสาร|ขั้นตอน|บทลงโทษ|ประกาศ|พ\.ร\.บ|ประกันสังคม|กองทุน|เปิดร้าน|ขึ้นทะเบียน)",
        re.IGNORECASE,
    )

    _NOISE_ONLY_RE = re.compile(r"^(?:[อ-ฮะ-์]+|[a-z]+|[!?.]+)$", re.IGNORECASE)
    _TH_LAUGH_5_RE = re.compile(r"^\s*5{3,}\s*$")

    # --------------------------
    # Confirmation classifier
    # --------------------------
    _YES_CORE = (
        "ใช่", "ช่าย", "ไช่", "ใข่",
        "ยืนยัน", "คอนเฟิร์ม", "confirm",
        "ถูกต้อง", "โอเค", "ตกลง",
        "ได้", "ได้เลย",
        "เอา", "เอาเลย", "จัดไป", "ไปเลย",
        "yes", "yeah", "yep", "yup", "ok", "okay",
        "เยส", "เยป", "เย้ป",
        "งับ", "ค้าบ", "คั้บ", "เออ", "อือ", "อืม",
    )
    _NO_CORE = (
        "ไม่", "ไม่เอา", "ไม่ต้อง", "ยังไม่", "ยกเลิก", "ช่างมัน",
        "no", "nope", "cancel",
        "ไม่เปลี่ยน", "ไม่สลับ",
    )

    # --------------------------
    # Helpers: message append (Supervisor ONLY)
    # --------------------------
    def _add_user(self, state: ConversationState, text: str) -> None:
        state.messages = state.messages or []
        if not text:
            return
        if state.messages and state.messages[-1].get("role") == "user" and (state.messages[-1].get("content") or "") == text:
            return
        state.messages.append({"role": "user", "content": text})

    def _add_assistant(self, state: ConversationState, text: str) -> None:
        state.messages = state.messages or []
        if not text:
            return
        if state.messages and state.messages[-1].get("role") == "assistant" and (state.messages[-1].get("content") or "") == text:
            return
        state.messages.append({"role": "assistant", "content": text})

    def _normalize_for_intent(self, s: str) -> str:
        t = (s or "").strip().lower()
        t = re.sub(r"[!！?？。,，]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        return t.strip()

    def _normalize_confirm_text(self, s: str) -> str:
        t = self._normalize_for_intent(s)
        t = re.sub(r"[^\w\u0E00-\u0E7F\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _classify_yes_no_det(self, user_text: str) -> Dict[str, Any]:
        t = self._normalize_confirm_text(user_text)
        if not t:
            return {"yes": False, "no": False, "confidence": 0.0, "method": "empty"}

        if re.fullmatch(r"1", t):
            return {"yes": True, "no": False, "confidence": 0.95, "method": "num_yes"}
        if re.fullmatch(r"2", t):
            return {"yes": False, "no": True, "confidence": 0.95, "method": "num_no"}

        if t in {"ครับ", "คับ", "ค่ะ", "คะ"}:
            return {"yes": False, "no": False, "confidence": 0.0, "method": "filler_only"}

        def _has_any(tokens) -> bool:
            for tok in tokens:
                if tok and tok in t:
                    return True
            return False

        yes = _has_any(self._YES_CORE)
        no = _has_any(self._NO_CORE)

        if yes and no:
            return {"yes": False, "no": False, "confidence": 0.0, "method": "conflict"}

        if yes:
            return {"yes": True, "no": False, "confidence": 0.86, "method": "det_contains"}
        if no:
            return {"yes": False, "no": True, "confidence": 0.86, "method": "det_contains"}

        return {"yes": False, "no": False, "confidence": 0.0, "method": "unclear"}

    # --------------------------
    # LLM helpers (confirm/style + greet prefix)
    # --------------------------
    def _default_confirm_llm_call(self) -> Callable[[str], dict]:
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=96,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = (
                "หน้าที่: ตีความว่า 'ข้อความผู้ใช้' เป็นการยืนยัน (yes) หรือปฏิเสธ (no) หรือยังไม่ชัดเจน\n"
                "ให้ดูโทน/เจตนา ไม่ต้องยึดแค่คำว่า 'ใช่/ไม่'\n"
                "ตัวอย่าง yes: งับ, ได้เลย, โอเค, ถูกต้อง, ยืนยัน, เอาเลย, จัดไป, ไปเลย\n"
                "ตัวอย่าง no: ไม่เอา, ยกเลิก, ช่างมัน, ไม่ต้อง, ยังไม่\n"
                "ถ้ากำกวมจริงๆ ให้ confidence ต่ำ\n"
                "ตอบเป็น JSON เท่านั้น:\n"
                '{ "yes": true/false, "no": true/false, "confidence": 0.0 }\n'
                f"ข้อความผู้ใช้: {user_text}"
            )
            try:
                text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_style_llm_call(self) -> Callable[[str], dict]:
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=96,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str) -> dict:
            prompt = (
                "หน้าที่: วิเคราะห์ว่า 'ข้อความผู้ใช้' ต้องการให้คำตอบ\n"
                "1) สั้น/กระชับ (practical) หรือ 2) ละเอียด/เชิงลึก (academic)\n"
                "ห้ามเดาสุ่ม ถ้าไม่ชัดให้ confidence ต่ำ\n"
                "ตอบเป็น JSON เท่านั้น:\n"
                '{ "wants_long": true/false, "wants_short": true/false, "confidence": 0.0 }\n'
                f"ข้อความผู้ใช้: {user_text}"
            )
            try:
                text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_greet_prefix_llm_call(self) -> Callable[[str, str, str, bool], dict]:
        """
        LLM returns ONLY the prefix (no numbered menu).
        include_intro:
          - True  => include name/role exactly once (first greeting only)
          - False => do NOT mention name/role anymore
        Return JSON: {"prefix": "..."}
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.35,
            max_tokens=120,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(kind: str, persona_id: str, last_topic_hint: str, include_intro: bool) -> dict:
            prompt = (
                "หน้าที่: เขียนข้อความทักทาย/ตอบรับภาษาไทยแบบมนุษย์\n"
                "ข้อกำหนดร่วม:\n"
                "- 1 ประโยคสั้นๆ + ปิดท้ายด้วยคำถามสั้นๆ 1 ข้อ\n"
                "- ห้ามใส่รายการหัวข้อ/เลขข้อ/เมนู\n"
                "- ห้ามสั่ง user ว่า ‘เลือก/พิมพ์/กด’\n"
                "- ต้องลงท้ายด้วย 'ครับ'\n"
                "โทนตาม persona:\n"
                "  - practical: ตรง กระชับ\n"
                "  - academic: สุภาพมืออาชีพ แต่ไม่ยาว\n"
                "กฎ include_intro:\n"
                "- ถ้า include_intro=true: ต้องแนะนำตัวว่า Restbiz ช่วยเรื่องกฎหมาย/ใบอนุญาต/ภาษีร้านอาหาร\n"
                "- ถ้า include_intro=false: ห้ามพูดชื่อ Restbiz และห้ามบอกหน้าที่บอทซ้ำ\n"
                "ตอบเป็น JSON เท่านั้น: {\"prefix\": \"...\"}\n"
                f"kind: {kind}\n"
                f"persona: {persona_id}\n"
                f"include_intro: {str(bool(include_intro)).lower()}\n"
                f"last_topic_hint: {last_topic_hint}\n"
            )
            try:
                text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _strip_code_fences(self, text: str) -> str:
        t = (text or "").strip()
        if "```json" in t:
            return t.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in t:
            parts = t.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        return t

    # --------------------------
    # Greeting/noise classification
    # --------------------------
    _EN_GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo+)\b", re.IGNORECASE)
    _EN_GOOD_TIME_RE = re.compile(r"^\s*good\s+(morning|afternoon|evening|night)\b", re.IGNORECASE)
    _TH_SAWASDEE_FUZZY_RE = re.compile(r"^\s*สว[^\s]{0,6}ดี", re.IGNORECASE)
    _TH_WATDEE_RE = re.compile(r"^\s*หวัดดี", re.IGNORECASE)
    _TH_DEE_RE = re.compile(r"^\s*ดี(?:ครับ|คับ|ค่ะ|คะ|งับ|จ้า|จ้ะ|ค่า)?", re.IGNORECASE)

    def _looks_like_greeting_or_thanks(self, s: str) -> bool:
        raw = (s or "").strip()
        if not raw:
            return True

        if self._TH_LAUGH_5_RE.match(raw):
            return True

        # numeric-only should NOT be greeting (likely a selection)
        if self._LIKELY_SELECTION_RE.match(raw):
            return False

        t = self._normalize_for_intent(raw)

        if self._THANKS_RE.search(t):
            return True

        if len(t) <= 2:
            return True

        if self._QUESTION_MARKERS_RE.search(t):
            return False
        if self._LEGAL_SIGNAL_RE.search(t):
            return False

        if self._EN_GREETING_RE.match(t) or self._EN_GOOD_TIME_RE.match(t):
            return True
        if self._TH_WATDEE_RE.match(t) or self._TH_SAWASDEE_FUZZY_RE.match(t):
            return True
        if self._TH_DEE_RE.match(t) and not self._QUESTION_MARKERS_RE.search(t):
            return True

        if len(t) <= 14 and self._NOISE_ONLY_RE.match(t):
            return True

        return False

    def _looks_like_legal_question(self, s: str) -> bool:
        t = self._normalize_for_intent(s)
        if not t:
            return False
        if self._LEGAL_SIGNAL_RE.search(t):
            return True
        if self._QUESTION_MARKERS_RE.search(t) and len(t) >= 6:
            return True
        return False

    def _is_noise(self, s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if self._TH_LAUGH_5_RE.match(t):
            return True
        # numeric-only is not noise (selection)
        if self._LIKELY_SELECTION_RE.match(t):
            return False
        if len(t) <= 2:
            return True
        if self._NOISE_ONLY_RE.match(t.lower()) and not self._looks_like_legal_question(t):
            return True
        return False

    # --------------------------
    # Academic intake lock only when truly active
    # --------------------------
    _ACADEMIC_LOCK_STAGES = {"awaiting_slots", "awaiting_sections"}

    def _is_academic_intake_active(self, state: ConversationState) -> bool:
        flow = (state.context or {}).get("academic_flow")
        if not isinstance(flow, dict):
            return False
        stage = str(flow.get("stage") or "").strip()
        if not stage:
            return False
        return stage in self._ACADEMIC_LOCK_STAGES

    # --------------------------
    # Intent classification
    # --------------------------
    def _looks_like_mode_status_query(self, s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if ("โหมด" not in t and "mode" not in t.lower() and "persona" not in t.lower() and "บุคลิก" not in t):
            return False
        return bool(self._MODE_STATUS_Q.search(t))

    def _looks_like_switch_without_target(self, s: str) -> bool:
        t = (s or "").strip().lower()
        if not t:
            return False
        if any(v in t for v in self._SWITCH_VERBS) and any(m in t for m in self._SWITCH_MARKERS):
            if not re.search(r"\b(academic|practical)\b|วิชาการ|ละเอียด|สั้น|กระชับ", t):
                return True
        if re.fullmatch(r"(เปลี่ยน|สลับ|ปรับ)\s*", t):
            return True
        return False

    def _infer_user_style_request_det(self, s: str) -> Dict[str, bool]:
        t = self._normalize_for_intent(s)
        if not t:
            return {"wants_short": False, "wants_long": False}

        wants_short = any(h in t for h in self._TARGET_PRACTICAL_HINTS)
        wants_long = any(h in t for h in self._TARGET_ACADEMIC_HINTS)

        if wants_short and wants_long:
            return {"wants_short": False, "wants_long": False}

        return {"wants_short": wants_short, "wants_long": wants_long}

    def _infer_user_style_request_hybrid(self, s: str) -> Dict[str, Any]:
        text = s or ""
        if not self._STYLE_LIKELY_RE.search(text):
            det = self._infer_user_style_request_det(text)
            if det["wants_short"] or det["wants_long"]:
                return {"wants_short": det["wants_short"], "wants_long": det["wants_long"], "method": "det", "confidence": 0.9}
            return {"wants_short": False, "wants_long": False, "method": "none", "confidence": 0.0}

        res: Dict[str, Any] = {}
        try:
            res = self.llm_style_call(text) or {}
        except Exception:
            res = {}

        try:
            confv = float(res.get("confidence", 0.0) or 0.0)
        except Exception:
            confv = 0.0

        wants_long = bool(res.get("wants_long", False)) if confv >= 0.55 else False
        wants_short = bool(res.get("wants_short", False)) if confv >= 0.55 else False

        if wants_long and wants_short:
            wants_long = False
            wants_short = False

        if wants_long or wants_short:
            return {"wants_short": wants_short, "wants_long": wants_long, "method": "llm", "confidence": confv}

        det = self._infer_user_style_request_det(text)
        if det["wants_short"] or det["wants_long"]:
            return {"wants_short": det["wants_short"], "wants_long": det["wants_long"], "method": "det_fallback", "confidence": 0.7}

        return {"wants_short": False, "wants_long": False, "method": "llm_low", "confidence": confv}

    def _classify_intent(self, state: ConversationState, user_input: str) -> Dict[str, Any]:
        state.context = state.context or {}
        text = (user_input or "")

        if self._is_academic_intake_active(state):
            return {"intent": self.INTENT_ACAD_INTAKE_REPLY, "meta": {}}

        if state.context.get("awaiting_persona_confirmation"):
            return {"intent": self.INTENT_CONFIRM_YESNO, "meta": {}}

        if self._looks_like_switch_without_target(text):
            return {"intent": self.INTENT_EXPLICIT_SWITCH, "meta": {"kind": "no_target"}}

        if self._looks_like_mode_status_query(text):
            return {"intent": self.INTENT_MODE_STATUS, "meta": {}}

        style = self._infer_user_style_request_hybrid(text)
        if style.get("wants_long") or style.get("wants_short"):
            return {"intent": self.INTENT_EXPLICIT_SWITCH, "meta": {"kind": "style", **style}}

        if self._SMALLTALK_RE.search(text) or self._looks_like_greeting_or_thanks(text):
            return {"intent": self.INTENT_GREETING, "meta": {}}

        if self._looks_like_legal_question(text):
            return {"intent": self.INTENT_LEGAL_NEW, "meta": {}}

        if self._is_noise(text):
            return {"intent": self.INTENT_NOISE, "meta": {}}

        return {"intent": self.INTENT_UNKNOWN, "meta": {}}

    # --------------------------
    # Core router
    # --------------------------
    def __init__(
        self,
        retriever,
        llm_confirm_call: Optional[Callable[[str], dict]] = None,
        llm_style_call: Optional[Callable[[str], dict]] = None,
        llm_greet_prefix_call: Optional[Callable[[str, str, str, bool], dict]] = None,
    ):
        self.retriever = retriever
        self._academic = AcademicPersonaService(retriever=retriever)
        self._practical = PracticalPersonaService(retriever=retriever)

        self.llm_confirm_call = llm_confirm_call or self._default_confirm_llm_call()
        self.llm_style_call = llm_style_call or self._default_style_llm_call()
        self.llm_greet_prefix_call = llm_greet_prefix_call or self._default_greet_prefix_llm_call()

        # seeded per session in _get_rng()
        self._rng = random.Random()

    # --------------------------
    # RNG: stable per session + moves each greeting
    # --------------------------
    def _get_session_seed(self, state: ConversationState) -> int:
        """
        Use session_id if exists; fallback to a stable hash of current process + first timestamp stored in context.
        Stored in state.context["rng_seed"] once.
        """
        state.context = state.context or {}
        if isinstance(state.context.get("rng_seed"), int):
            return int(state.context["rng_seed"])

        sid = ""
        if hasattr(state, "session_id"):
            sid = str(getattr(state, "session_id") or "")
        if not sid:
            sid = str(state.context.get("session_id") or "")

        if not sid:
            sid = str(id(state))

        h = hashlib.sha256(sid.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16)
        state.context["rng_seed"] = seed
        return seed

    def _get_rng(self, state: ConversationState) -> random.Random:
        seed = self._get_session_seed(state)
        turns = int((state.context or {}).get("greet_menu_turns") or 0)
        mixed = seed ^ ((turns + 1) * 2654435761 & 0xFFFFFFFF)
        return random.Random(mixed)

    # --------------------------
    # Retrieval gate (Practical legal MUST retrieve)
    # --------------------------
    _TOKEN_SPLIT_RE = re.compile(r"[\s/,\-–—|]+", re.UNICODE)

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

    def _should_retrieve_new_for_practical(self, state: ConversationState, user_input: str) -> bool:
        q = (user_input or "").strip()
        if not q:
            return False

        has_docs = bool(getattr(state, "current_docs", None))
        if not has_docs:
            return True

        last_q = (
            (getattr(state, "last_retrieval_query", None) or "")
            or str((state.context or {}).get("last_retrieval_query") or "")
        ).strip()

        if not last_q:
            return True

        if last_q == q:
            return False

        overlap = self._topic_overlap_ratio(last_q, q)
        return overlap < 0.22

    def _ensure_practical_retrieval_for_legal(self, state: ConversationState, user_input: str) -> None:
        state.context = state.context or {}
        q = (user_input or "").strip()
        if not q:
            return

        if not self._should_retrieve_new_for_practical(state, q):
            return

        docs = self.retriever.invoke(q)
        results: List[Dict[str, Any]] = []
        top_k = int(getattr(conf, "RETRIEVAL_TOP_K", 20) or 20)
        for d in (docs or [])[:top_k]:
            results.append({"content": (getattr(d, "page_content", "") or "")[:600], "metadata": getattr(d, "metadata", {}) or {}})

        state.current_docs = results
        state.last_retrieval_query = q
        state.context["last_retrieval_query"] = q

    # --------------------------
    # Pending slot routing (keep + mixed-input support)
    # --------------------------
    _RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
    _ANY_NUMBER_RE = re.compile(r"\b(\d{1,2})\b")

    def _has_pending_slot(self, state: ConversationState) -> bool:
        p = (state.context or {}).get("pending_slot")
        return isinstance(p, dict) and isinstance(p.get("options"), list) and len(p.get("options")) > 0

    def _looks_like_pending_slot_reply(self, user_input: str) -> bool:
        raw = (user_input or "").strip()
        if not raw:
            return False
        if self._TH_LAUGH_5_RE.match(raw):
            return False

        low = raw.lower()

        if self._LIKELY_SELECTION_RE.match(raw):
            return True

        if self._ANY_NUMBER_RE.search(low):
            return True

        if re.search(r"(ทั้งหมด|all\b|ทุกข้อ|ทุกอย่าง)", low):
            return True

        return False

    def _parse_indices(self, text: str) -> List[int]:
        t = (text or "").strip()
        if not t:
            return []

        nums: List[int] = []

        for m in self._RANGE_RE.finditer(t):
            try:
                a = int(m.group(1))
                b = int(m.group(2))
                if a <= 0 or b <= 0:
                    continue
                lo, hi = (a, b) if a <= b else (b, a)
                nums.extend(list(range(lo, hi + 1)))
            except Exception:
                continue

        for m in self._ANY_NUMBER_RE.finditer(t):
            try:
                n = int(m.group(1))
                nums.append(n)
            except Exception:
                continue

        seen = set()
        out: List[int] = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _map_pending_slot_reply(self, pending: Dict[str, Any], user_input: str) -> Tuple[Optional[str], Optional[str]]:
        options = pending.get("options") or []
        allow_multi = bool(pending.get("allow_multi", False))

        raw = (user_input or "").strip()
        if not raw:
            return None, None

        low = raw.lower()

        if re.search(r"(ทั้งหมด|all\b|ทุกข้อ|ทุกอย่าง)", low):
            if allow_multi and options:
                return ", ".join([str(x).strip() for x in options if str(x).strip()]), None
            return None, "ตัวเลือกนี้ใช้ได้เฉพาะกรณีที่เลือกได้หลายข้อครับ"

        idxs = self._parse_indices(raw)

        if not idxs and re.fullmatch(r"\d{2,}", raw) and len(options) <= 9:
            idxs = [int(ch) for ch in raw if ch.isdigit()]

        if not idxs:
            return None, "กรุณาตอบเป็นตัวเลขตามตัวเลือกครับ"

        valid = [i for i in idxs if 1 <= i <= len(options)]
        if not valid:
            return None, "เลขที่เลือกไม่อยู่ในช่วงตัวเลือกครับ"

        if not allow_multi:
            chosen = options[valid[0] - 1]
            return str(chosen).strip(), None

        texts = [str(options[i - 1]).strip() for i in valid]
        texts = [x for x in texts if x]
        if not texts:
            return None, "เลขที่เลือกไม่อยู่ในช่วงตัวเลือกครับ"
        return ", ".join(texts), None

    def _should_route_pending_slot_now(self, state: ConversationState, user_input: str) -> bool:
        if not self._has_pending_slot(state):
            return False
        if not (user_input or "").strip():
            return False
        if self._TH_LAUGH_5_RE.match((user_input or "").strip()):
            return False

        ctx = state.context or {}

        if self._is_academic_intake_active(state):
            return False
        if ctx.get("awaiting_persona_confirmation"):
            return False
        if ctx.get("awaiting_persona_pick"):
            return False

        if self._looks_like_switch_without_target(user_input):
            return False
        if self._looks_like_mode_status_query(user_input):
            return False

        if self._STYLE_LIKELY_RE.search(user_input or "") and not self._LIKELY_SELECTION_RE.match((user_input or "").strip()):
            if not self._ANY_NUMBER_RE.search(user_input or ""):
                return False

        return self._looks_like_pending_slot_reply(user_input)

    def _route_pending_slot_to_persona(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        pending = (state.context or {}).get("pending_slot") or {}
        mapped, err = self._map_pending_slot_reply(pending, user_input)

        if err:
            msg = self._normalize_male(err)
            self._add_assistant(state, msg)
            state.last_action = "pending_slot_invalid_reply"
            return state, msg

        if not mapped:
            msg = self._normalize_male("กรุณาตอบเป็นตัวเลขตามตัวเลือกครับ")
            self._add_assistant(state, msg)
            state.last_action = "pending_slot_invalid_reply"
            return state, msg

        # when user picks a topic from greeting menu, remember it for "related" sampling later
        state.context = state.context or {}
        if isinstance(pending, dict) and pending.get("key") == "topic":
            state.context["last_topic"] = str(mapped).strip()

        # consume pending_slot now
        state.context.pop("pending_slot", None)

        pid = normalize_persona_id(state.persona_id)
        if pid == "academic":
            st2, reply = self._academic.handle(state, mapped, _internal=True)
            st2, reply = self._post_route_academic_auto_return(st2, reply)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            return st2, reply

        st2, reply = self._practical.handle(state, mapped, _internal=True)
        reply = self._normalize_male(reply)
        self._add_assistant(st2, reply)
        return st2, reply

    # --------------------------
    # Greeting/Menu (data-driven topics, intro only once)
    # --------------------------
    _MENU_SIZE = 5
    _POOL_MAX = 80  # bigger pool => less repeating

    def _format_numbered_options(self, options: List[str], max_items: int = 9) -> str:
        opts = [str(x).strip() for x in (options or []) if str(x).strip()]
        opts = opts[:max_items]
        return "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(opts)])

    def _sanitize_topic_label(self, s: str) -> str:
        raw = (s or "")
        t = raw.strip()
        if not t:
            return ""
        if "\n" in raw or "\r" in raw:
            return ""
        t = re.sub(r"\s+", " ", t).strip()

        # drop obvious garbage / placeholders
        if t in {"-", "—", "–", "N/A", "n/a", "NA", "na"}:
            return ""
        if len(t) < 3:
            return ""
        if len(t) > 64:
            return ""

        # reduce “หน่วยงานล้วนๆ” ให้มีโอกาสติดเมนูน้อยลง (แต่ไม่ห้าม)
        t = re.sub(r"^\s*หน่วยงาน\s*[:：]\s*", "", t).strip()
        return t

    _ORGISH_RE = re.compile(r"^(กรม|สำนักงาน|สำนัก|เทศบาล|อบต\.?|อบจ\.?|สำนักงานเขต)\b")

    def _topic_kind_weight(self, label: str, source_key: str) -> int:
        """
        Weighting to make menu look like “หัวข้อ”:
        - Prefer: license_type / operation_topic / operation_subtopic / operation_by_department
        - Deprioritize: department (pure org name) unless needed as backfill
        """
        k = (source_key or "").strip().lower()
        l = (label or "").strip()

        # Strong preference to “หัวข้อ/การดำเนินการ/ใบอนุญาต”
        if k in {"license_type", "ใบอนุญาต"}:
            return 5
        if k in {
            "operation_topic",
            "หัวข้อการดำเนินการย่อย",
            "operation_subtopic",
            "sub_operation_topic",
            "operation_topic_sub",
            "subtopic",
        }:
            return 4
        if k in {
            "operation_by_department",
            "operation_action",
            "การดำเนินการ ตามหน่วยงาน",
            "การดำเนินการตามหน่วยงาน",
            "action_by_department",
            "operation_process",
        }:
            return 4

        # department: allow but lower; even lower if org-ish
        if k in {"department", "หน่วยงาน"}:
            return 1 if self._ORGISH_RE.search(l) else 2

        return 2

    def _collect_topic_freq_from_docs(self, docs: List[Any]) -> Dict[str, int]:
        """
        Extract real labels from metadata.
        NEW: prioritize “ใบอนุญาต/การดำเนินการตามหน่วยงาน/หัวข้อการดำเนินการย่อย”
        by weighting their contribution higher than department.
        """
        freq: Dict[str, int] = {}

        def _add(v: Any, source_key: str) -> None:
            s = self._sanitize_topic_label(str(v) if v is not None else "")
            if not s:
                return
            w = self._topic_kind_weight(s, source_key=source_key)
            freq[s] = freq.get(s, 0) + int(w)

        # Candidate keys (support both your canonical keys + possible Thai/raw sheet keys)
        # NOTE: we DO NOT assume exact keys exist; we just try.
        LICENSE_KEYS = ["license_type", "ใบอนุญาต"]
        OP_BY_DEPT_KEYS = ["operation_by_department", "operation_action", "action_by_department", "operation_process", "การดำเนินการ ตามหน่วยงาน", "การดำเนินการตามหน่วยงาน"]
        OP_SUB_KEYS = ["operation_subtopic", "sub_operation_topic", "operation_topic_sub", "subtopic", "หัวข้อการดำเนินการย่อย"]
        OP_TOPIC_KEYS = ["operation_topic", "หัวข้อการดำเนินการย่อย"]  # keep both (some ingestors map subtopic into topic)
        DEPT_KEYS = ["department", "หน่วยงาน"]

        for d in (docs or []):
            md = getattr(d, "metadata", {}) or {}

            # 1) ใบอนุญาต
            for k in LICENSE_KEYS:
                _add(md.get(k), source_key=k)

            # 2) การดำเนินการตามหน่วยงาน
            for k in OP_BY_DEPT_KEYS:
                _add(md.get(k), source_key=k)

            # 3) หัวข้อการดำเนินการย่อย
            for k in OP_SUB_KEYS:
                _add(md.get(k), source_key=k)

            # 4) หัวข้อการดำเนินการ (ถ้ามี)
            for k in OP_TOPIC_KEYS:
                _add(md.get(k), source_key=k)

            # 5) หน่วยงาน (เป็น backfill)
            for k in DEPT_KEYS:
                _add(md.get(k), source_key=k)

        return freq

    def _build_topic_pool_from_corpus(self, state: ConversationState) -> List[Tuple[str, int]]:
        """
        Build (topic, freq) pool from REAL data using multiple broad queries.
        """
        queries = [
            # เปิดร้าน/ใบอนุญาต/สุขาภิบาล
            "ใบอนุญาต เปิดร้านอาหาร เทศบาล สำนักงานเขต สุขาภิบาลอาหาร",
            # ภาษี/VAT
            "ภาษี VAT ภพ.20 ใบกำกับภาษี กรมสรรพากร จด VAT",
            # จดทะเบียนธุรกิจ/พาณิชย์
            "จดทะเบียนพาณิชย์ นิติบุคคล DBD กรมพัฒนาธุรกิจการค้า หนังสือรับรอง",
            # แรงงาน/ประกันสังคม
            "ประกันสังคม ขึ้นทะเบียนนายจ้าง ลูกจ้าง กองทุนเงินทดแทน",
            # การดำเนินการตามหน่วยงาน (ให้ retriever ดึงเอกสารที่มี action/procedure fields)
            "ขั้นตอนการดำเนินการ เอกสารที่ต้องใช้ ค่าธรรมเนียม ระยะเวลา ช่องทางยื่นคำขอ",
        ]

        merged: Dict[str, int] = {}
        for q in queries:
            try:
                docs = self.retriever.invoke(q) or []
            except Exception:
                docs = []
            freq = self._collect_topic_freq_from_docs(docs)
            for k, v in freq.items():
                merged[k] = merged.get(k, 0) + int(v)

        # Extra: downrank pure-org labels globally (still keep for backfill)
        items = sorted(merged.items(), key=lambda x: (-x[1], x[0]))

        # Keep a larger pool to reduce repetition
        pool = items[: self._POOL_MAX]

        # hard fallback if metadata is empty (should be rare)
        if len(pool) < 10:
            base = [
                ("จด VAT / ขอ ภพ.20", 10),
                ("ใบกำกับภาษี / ออกใบเสร็จ", 9),
                ("ขอใบอนุญาตเปิดร้านอาหาร", 9),
                ("สุขาภิบาลอาหาร / อาหารสะอาด", 8),
                ("จดทะเบียนพาณิชย์ / DBD", 7),
                ("ขึ้นทะเบียนประกันสังคมนายจ้าง", 7),
                ("กองทุนเงินทดแทน", 6),
                ("เอกสารที่ต้องใช้", 6),
                ("ค่าธรรมเนียม", 5),
                ("ขั้นตอนการดำเนินการ", 5),
            ]
            existing = {k for k, _ in pool}
            for k, w in base:
                if k not in existing:
                    pool.append((k, w))

        state.context = state.context or {}
        state.context["topic_pool"] = pool
        return pool

    def _get_topic_pool(self, state: ConversationState) -> List[Tuple[str, int]]:
        state.context = state.context or {}
        cached = state.context.get("topic_pool")
        if isinstance(cached, list) and cached and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in cached):
            out: List[Tuple[str, int]] = []
            for t, w in cached:
                try:
                    out.append((str(t), int(w)))
                except Exception:
                    continue
            if out:
                return out
        return self._build_topic_pool_from_corpus(state)

    def _get_last_topic_hint(self, state: ConversationState) -> str:
        ctx = state.context or {}
        last = str(ctx.get("last_topic") or "").strip()
        if last:
            return last
        last_q = str(getattr(state, "last_retrieval_query", "") or "").strip()
        return last_q[:60] if last_q else ""

    def _weighted_sample_no_replace(self, pool: List[Tuple[str, int]], k: int, rng: random.Random) -> List[str]:
        """
        Weighted sample without replacement + flatten weights => diversity.
        """
        if not pool or k <= 0:
            return []

        topics = [t for t, _ in pool]
        weights = [max(1, int(w)) for _, w in pool]

        max_w = max(weights) if weights else 1
        weights = [max(1, int((w / max_w) * 7) + 1) for w in weights]

        chosen: List[str] = []
        local_topics = topics[:]
        local_weights = weights[:]

        while local_topics and len(chosen) < k:
            pick = rng.choices(local_topics, weights=local_weights, k=1)[0]
            chosen.append(pick)
            idx = local_topics.index(pick)
            local_topics.pop(idx)
            local_weights.pop(idx)

        return chosen

    def _related_topics_from_last(self, state: ConversationState, need: int) -> List[str]:
        """
        Pull docs near last topic/query and extract metadata topics (data-driven).
        """
        if need <= 0:
            return []
        last_q = self._get_last_topic_hint(state).strip()
        if not last_q:
            return []
        try:
            docs = self.retriever.invoke(last_q) or []
        except Exception:
            docs = []
        freq = self._collect_topic_freq_from_docs(docs[:16])
        items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        out: List[str] = []
        for k, _ in items:
            if k and k not in out:
                out.append(k)
            if len(out) >= need:
                break
        return out

    def _compose_menu_topics(self, state: ConversationState, size: int) -> List[str]:
        """
        Always aim for exactly `size` topics.
        Mix: related (<=2) + fresh random from corpus.
        Also avoid repeating the previous menu too much.
        """
        pool = self._get_topic_pool(state)
        size = max(1, int(size))

        rng = self._get_rng(state)

        related_target = 2 if size >= 5 else 1
        related = self._related_topics_from_last(state, need=related_target)

        related_set = set(related)
        pool_fresh = [(t, w) for (t, w) in pool if t not in related_set]

        fresh_need = max(0, size - len(related))
        fresh = self._weighted_sample_no_replace(pool_fresh, k=fresh_need, rng=rng)

        combined: List[str] = []
        for t in related + fresh:
            t2 = self._sanitize_topic_label(t)
            if t2 and t2 not in combined:
                combined.append(t2)
            if len(combined) >= size:
                break

        if len(combined) < size:
            for t, _ in pool:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in combined:
                    combined.append(t2)
                if len(combined) >= size:
                    break

        # anti-repeat with last_menu_topics
        last_menu = (state.context or {}).get("last_menu_topics")
        if isinstance(last_menu, list) and len(last_menu) >= 3:
            overlap = len(set(last_menu).intersection(set(combined)))
            if overlap >= 4 and len(pool_fresh) >= size:
                fresh2 = self._weighted_sample_no_replace(pool_fresh, k=size, rng=rng)
                combined2: List[str] = []
                for t in fresh2:
                    t2 = self._sanitize_topic_label(t)
                    if t2 and t2 not in combined2:
                        combined2.append(t2)
                    if len(combined2) >= size:
                        break
                if len(combined2) == size:
                    combined = combined2

        return combined[:size]

    def _get_prefix_llm(self, kind: str, state: ConversationState, include_intro: bool) -> str:
        pid = normalize_persona_id(state.persona_id)
        last_hint = self._get_last_topic_hint(state)

        res: Dict[str, Any] = {}
        try:
            res = self.llm_greet_prefix_call(kind, pid, last_hint, bool(include_intro)) or {}
        except Exception:
            res = {}

        prefix = ""
        if isinstance(res, dict):
            prefix = str(res.get("prefix") or "").strip()

        if not prefix:
            if include_intro:
                prefix = "สวัสดีครับ ผมคือ Restbiz ผู้ช่วยเรื่องกฎหมาย ใบอนุญาต และภาษีสำหรับร้านอาหารครับ อยากให้ช่วยเรื่องไหนครับ"
            else:
                prefix = "ได้ครับ ตอนนี้อยากให้ช่วยเรื่องไหนครับ"

        prefix = re.sub(r"\s+", " ", prefix).strip()
        return self._normalize_male(prefix)

    def _render_greeting_with_menu(self, state: ConversationState, kind: str, menu_topics: List[str], include_intro: bool) -> str:
        prefix = self._get_prefix_llm(kind, state, include_intro=include_intro)
        menu = self._format_numbered_options(menu_topics, max_items=9)
        msg = (prefix.rstrip() + "\n" + menu).strip()
        return self._normalize_male(msg)

    def _handle_greeting(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        """
        Policy:
        - ALWAYS show menu (5 topics) on greeting/noise/thanks turns.
        - Intro (name/role) only once: first greeting menu turn.
        - pending_slot(topic) is always (re)written with current menu options.
        """
        state.context = state.context or {}
        raw = (user_input or "").strip()
        t = self._normalize_for_intent(raw)

        kind = "greet"
        if not raw:
            kind = "blank"
        elif self._THANKS_RE.search(t):
            kind = "thanks"
        elif self._SMALLTALK_RE.search(raw) or self._TH_LAUGH_5_RE.match(raw):
            kind = "smalltalk"

        turns = int(state.context.get("greet_menu_turns") or 0)
        include_intro = turns == 0

        topics = self._compose_menu_topics(state, size=self._MENU_SIZE)

        state.context["pending_slot"] = {"key": "topic", "options": topics, "allow_multi": False}
        state.context["main_menu_shown"] = True
        state.context["last_menu_topics"] = topics

        msg = self._render_greeting_with_menu(state, kind=kind, menu_topics=topics, include_intro=include_intro)
        self._add_assistant(state, msg)

        state.context["greet_menu_turns"] = turns + 1
        state.last_action = "greeting_first_menu" if include_intro else "greeting_with_menu_refresh"

        return state, msg

    # --------------------------
    # Switch helpers
    # --------------------------
    def _mark_auto_return_if_practical_to_academic(self, state: ConversationState, target_pid: str) -> None:
        origin = normalize_persona_id(state.persona_id)
        state.context = state.context or {}
        state.context["switch_origin_persona"] = origin

        if origin == "practical" and normalize_persona_id(target_pid) == "academic":
            state.context["auto_return_after_academic_done"] = True
        else:
            state.context.pop("auto_return_after_academic_done", None)

    def _enter_switch_confirmation(self, state: ConversationState, target_pid: str, replay_user_input: str = "") -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        state.context["pending_persona"] = normalize_persona_id(target_pid)
        state.context["pending_replay_user_input"] = replay_user_input or ""
        state.context["awaiting_persona_confirmation"] = True
        state.context["confirm_tries"] = 0

        self._mark_auto_return_if_practical_to_academic(state, target_pid)

        msg = self._normalize_male(f"ต้องการเปลี่ยนเป็นโหมด {normalize_persona_id(target_pid)} ใช่ไหมครับ")
        self._add_assistant(state, msg)
        state.last_action = "persona_switch_confirm"
        return state, msg

    # --------------------------
    # Main handle
    # --------------------------
    def handle(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        self._sync_persona_and_profile(state)

        # First-touch: if no user input -> greeting
        if not state.context.get("did_greet") and not (user_input or "").strip() and not self._is_academic_intake_active(state):
            state.context["did_greet"] = True
            return self._handle_greeting(state, user_input="")

        if not state.context.get("did_greet"):
            state.context["did_greet"] = True

        if (user_input or "").strip():
            self._add_user(state, user_input)

        # Academic intake lock
        if self._is_academic_intake_active(state) and not state.context.get("awaiting_persona_confirmation"):
            state.persona_id = "academic"
            self._sync_persona_and_profile(state)
            state.context["did_greet"] = True

            st2, reply = self._academic.handle(
                state,
                user_input,
                force_intake=True,
                _internal=True,
            )
            st2, reply = self._post_route_academic_auto_return(st2, reply)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            return st2, reply

        # Pending slot has priority (but not over switch/confirm)
        if self._should_route_pending_slot_now(state, user_input):
            return self._route_pending_slot_to_persona(state, user_input)

        intent_obj = self._classify_intent(state, user_input)
        intent = intent_obj.get("intent", self.INTENT_UNKNOWN)
        meta = intent_obj.get("meta", {}) or {}

        if intent == self.INTENT_CONFIRM_YESNO:
            return self._handle_persona_confirmation(state, user_input)

        if intent == self.INTENT_EXPLICIT_SWITCH:
            if meta.get("kind") == "no_target":
                state.context["awaiting_persona_pick"] = True
                msg = self._normalize_male("ต้องการเปลี่ยนเป็นโหมดไหนครับ 1) practical 2) academic")
                self._add_assistant(state, msg)
                state.last_action = "persona_pick_ask"
                return state, msg

            cur = normalize_persona_id(state.persona_id)
            wants_long = bool(meta.get("wants_long", False))
            wants_short = bool(meta.get("wants_short", False))

            if cur == "practical" and wants_long:
                return self._enter_switch_confirmation(state, "academic", replay_user_input=user_input)
            if cur == "academic" and wants_short:
                return self._enter_switch_confirmation(state, "practical", replay_user_input=user_input)

        if state.context.get("awaiting_persona_pick"):
            return self._handle_persona_pick(state, user_input)

        if intent == self.INTENT_MODE_STATUS:
            msg = self._normalize_male(f"ตอนนี้อยู่โหมด {normalize_persona_id(state.persona_id)} ครับ")
            self._add_assistant(state, msg)
            state.last_action = "mode_status"
            return state, msg

        # Greeting/noise always returns menu (with intro only once)
        if intent == self.INTENT_GREETING:
            return self._handle_greeting(state, user_input)
        if intent == self.INTENT_NOISE:
            return self._handle_greeting(state, user_input="")

        # Route by persona
        if normalize_persona_id(state.persona_id) == "academic":
            st2, reply = self._academic.handle(state, user_input, _internal=True)
            st2, reply = self._post_route_academic_auto_return(st2, reply)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            return st2, reply

        # Practical legal must retrieve (supervisor-level safety)
        if intent == self.INTENT_LEGAL_NEW:
            self._ensure_practical_retrieval_for_legal(state, user_input)

        st2, reply = self._practical.handle(state, user_input, _internal=True)
        reply = self._normalize_male(reply)
        self._add_assistant(st2, reply)
        return st2, reply

    # --------------------------
    # Post-route: academic auto return
    # --------------------------
    def _post_route_academic_auto_return(self, state: ConversationState, reply: str) -> Tuple[ConversationState, str]:
        flow = (state.context or {}).get("academic_flow") or {}
        auto_ret = bool((state.context or {}).get("auto_return_after_academic_done"))

        if auto_ret and isinstance(flow, dict) and flow.get("stage") == "done":
            state.persona_id = "practical"
            self._sync_persona_and_profile(state)

            state.context.pop("auto_return_after_academic_done", None)
            state.context.pop("switch_origin_persona", None)
            state.context.pop("auto_return_to_practical", None)

            # After auto-return, show a refreshed menu (no intro)
            topics = self._compose_menu_topics(state, size=self._MENU_SIZE)
            state.context["pending_slot"] = {"key": "topic", "options": topics, "allow_multi": False}
            state.context["main_menu_shown"] = True
            state.context["last_menu_topics"] = topics

            follow = self._render_greeting_with_menu(state, kind="smalltalk", menu_topics=topics, include_intro=False)

            final = (reply or "").rstrip() + "\n\n" + follow
            final = self._normalize_male(final)

            state.last_action = "academic_done_auto_return_to_practical"
            return state, final

        return state, reply

    # --------------------------
    # Persona pick/confirmation
    # --------------------------
    def _handle_persona_pick(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        t = self._normalize_for_intent(user_input)

        if re.search(r"\b1\b", t):
            target_pid = "practical"
        elif re.search(r"\b2\b", t):
            target_pid = "academic"
        else:
            target_pid = None
            if any(h in t for h in self._TARGET_PRACTICAL_HINTS) or "practical" in t:
                target_pid = "practical"
            if any(h in t for h in self._TARGET_ACADEMIC_HINTS) or "academic" in t:
                if target_pid and target_pid != "academic":
                    target_pid = None
                else:
                    target_pid = "academic"

        if not target_pid:
            msg = self._normalize_male("เลือกโหมด 1) practical 2) academic ครับ")
            self._add_assistant(state, msg)
            return state, msg

        state.context.pop("awaiting_persona_pick", None)
        return self._enter_switch_confirmation(state, target_pid, replay_user_input="")

    def _handle_persona_confirmation(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        target_pid = state.context.get("pending_persona")
        if not target_pid:
            state.context.pop("awaiting_persona_confirmation", None)
            msg = self._normalize_male("ระบบไม่สามารถเปลี่ยนโหมดได้ในขณะนี้ครับ")
            self._add_assistant(state, msg)
            return state, msg

        tries = int(state.context.get("confirm_tries", 0) or 0)

        det = self._classify_yes_no_det(user_input)
        confirm_yes = bool(det.get("yes", False))
        confirm_no = bool(det.get("no", False))
        det_method = str(det.get("method") or "")

        if (
            not confirm_yes
            and not confirm_no
            and self.llm_confirm_call
            and det_method not in {"filler_only", "empty"}
        ):
            res = self.llm_confirm_call(user_input)
            try:
                conf_llm = float(res.get("confidence", 0.0) or 0.0)
            except Exception:
                conf_llm = 0.0

            if conf_llm >= 0.55:
                confirm_yes = bool(res.get("yes", False))
                confirm_no = bool(res.get("no", False))

        if confirm_yes:
            state.persona_id = normalize_persona_id(target_pid)
            self._sync_persona_and_profile(state)

            replay_input = state.context.get("pending_replay_user_input") or ""

            state.context.pop("pending_persona", None)
            state.context.pop("pending_replay_user_input", None)
            state.context.pop("awaiting_persona_confirmation", None)
            state.context.pop("confirm_tries", None)

            prefix = self._normalize_male(f"ตอนนี้เป็นโหมด {state.persona_id} แล้วครับ")
            self._add_assistant(state, prefix)

            if replay_input.strip():
                if state.persona_id == "academic":
                    st2, reply = self._academic.handle(state, replay_input, _internal=True)
                    st2, reply = self._post_route_academic_auto_return(st2, reply)
                else:
                    st2, reply = self._practical.handle(state, replay_input, _internal=True)

                combined = self._normalize_male(prefix + "\n" + (reply or "").strip())
                self._add_assistant(st2, combined)
                state.last_action = "persona_switch_applied_replayed"
                return st2, combined

            state.last_action = "persona_switch_applied"
            return state, prefix

        if confirm_no:
            msg = self._normalize_male("โอเคครับ ใช้โหมดเดิมต่อ")
            state.context.pop("pending_persona", None)
            state.context.pop("pending_replay_user_input", None)
            state.context.pop("awaiting_persona_confirmation", None)
            state.context.pop("confirm_tries", None)
            state.context.pop("auto_return_after_academic_done", None)
            state.context.pop("switch_origin_persona", None)

            self._add_assistant(state, msg)
            state.last_action = "persona_switch_cancelled"
            return state, msg

        tries += 1
        state.context["confirm_tries"] = tries

        if tries >= 2:
            msg = self._normalize_male(
                f"ยังไม่ชัดว่าต้องการเปลี่ยนเป็นโหมด {target_pid} ไหมครับ "
                "ถ้าจะเปลี่ยนพิมพ์ “ได้เลย/โอเค/ยืนยัน/จัดไป” "
                "ถ้าไม่เปลี่ยนพิมพ์ “ไม่เอา/ยกเลิก” (ตอนนี้ขอใช้โหมดเดิมไว้ก่อนครับ)"
            )
            state.context.pop("pending_persona", None)
            state.context.pop("pending_replay_user_input", None)
            state.context.pop("awaiting_persona_confirmation", None)
            state.context.pop("confirm_tries", None)
            state.context.pop("auto_return_after_academic_done", None)
            state.context.pop("switch_origin_persona", None)

            self._add_assistant(state, msg)
            state.last_action = "persona_switch_unclear_cancelled"
            return state, msg

        msg = self._normalize_male(
            f"จะเปลี่ยนเป็นโหมด {target_pid} ไหมครับ (ตอบได้เลย เช่น “ได้เลย/ยืนยัน” หรือ “ไม่เอา/ยกเลิก”)"
        )
        self._add_assistant(state, msg)
        return state, msg

    # --------------------------
    # Persona/profile sync
    # --------------------------
    def _sync_persona_and_profile(self, state: ConversationState) -> None:
        pid = normalize_persona_id(state.persona_id)
        state.persona_id = pid

        state.context = state.context or {}
        state.context["persona_id"] = state.persona_id

        current_profile = getattr(state, "strict_profile", None) or {}
        state.strict_profile = build_strict_profile(
            persona_id=pid,
            current=current_profile,
        )

        state.context = apply_persona_profile(
            state.context,
            state.strict_profile,
        )