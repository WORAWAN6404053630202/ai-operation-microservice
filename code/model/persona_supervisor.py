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

✅ NEW (this change request): Option A production-ready menu
7) Make menu topics look like “หัวข้อ” more, by prioritizing metadata fields that correspond to:
   - ใบอนุญาต
   - การดำเนินการตามหน่วยงาน
   - หัวข้อการดำเนินการย่อย
   and de-prioritizing pure “หน่วยงาน/department” labels unless needed as backfill.

✅ NEW (your latest request): "quality gate + level separation + safe fallback"
- Quality gate:
  - reject obvious noise/placeholder/org-only labels aggressively (but still allow as last-resort backfill)
  - require "menu-worthy" signals for main menu (permit/procedure/tax/docs/fees/time/channel/etc.)
- Level separation:
  - keep "case-specific / detail-ish" labels OUT of main menu (put them only in drill-down / Phase 3 later)
- Safe fallback:
  - always guarantee exactly 5 menu items even if data is sparse

✅ NEWEST (your requirement):
- If user asks for "ละเอียด/เชิงลึก/วิชาการ" OR hints Academic persona:
  - DO NOT switch immediately
  - MUST say current persona + ask confirmation to switch to the other persona
- If user says "change/switch persona/mode" (with/without target):
  - MUST say current persona + ask confirmation to switch to the other persona
- If user requests persona that is already active:
  - MUST say "already in mode X" and ask if they want to switch back to the other persona
- Academic final_answer 3-branch logic stays in AcademicPersonaService (no changes here)

✅ BUGFIX (this request):
- pending_slot ที่เป็นตัวเลือก 2-5 ข้อ (เช่น location: กรุงเทพฯ/ต่างจังหวัด)
  ต้อง “รับคำตอบแบบข้อความ” ได้ด้วย (เช่น "กทม", "กรุงเทพ", "ต่างจังหวัด", "จังหวัด")
  โดย *ไม่ hardcode* — ใช้ LLM map user_text -> option ที่ใกล้ที่สุดแบบ deterministic (temp=0)
"""

from __future__ import annotations

from typing import Tuple, Callable, Optional, Dict, Any, List
import logging
import re
import json
import random
import hashlib

_LOG = logging.getLogger("restbiz.supervisor")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import conf
from model.conversation_state import ConversationState
from utils.llm_call import llm_invoke
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

    _SWITCH_VERBS = ("เปลี่ยน", "สลับ", "ปรับ", "ขอเปลี่ยน", "ขอสลับ", "ขอปรับ", "change", "switch", "ไป")
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
        "อธิบายมากกว่า",
        "อธิบายเพิ่มเติม",
        "รายละเอียดมากกว่า",
        "รายละเอียดเพิ่มเติม",
        "บอกเพิ่มเติม",
        "อธิบายให้มากกว่า",
        "มากกว่านี้",
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

    _NOISE_ONLY_RE = re.compile(r"^(?:[a-z]+|[!?.]+)$", re.IGNORECASE)
    _TH_LAUGH_5_RE = re.compile(r"^\s*5{3,}\s*$")

    # Follow-up patterns (handled before fallback_safe_return)
    _ELABORATE_RE = re.compile(
        r"(อธิบาย(มากกว่า|เพิ่ม|เพิ่มเติม|ขยาย|ให้ละเอียด|ต่อ)|ขยายความ|เพิ่มเติมอีก|รายละเอียดมากกว่า|บอกเพิ่ม|เล่าให้ฟัง|อธิบายต่อ|รายละเอียดเพิ่ม"
        r"|กลับไปเรื่องเดิม|กลับเรื่องเก่า|กลับเรื่องเดิม|กลับเรื่องที่คุย|คุยต่อเรื่องเดิม|ขอกลับไปเรื่อง|อยากคุยต่อ)",
        re.IGNORECASE,
    )

    # Patterns for resuming an Academic session after auto-return to Practical
    _ACADEMIC_RESUME_RE = re.compile(
        r"(ทั้งหมด|ขอทั้งหมด|ดูทั้งหมด|ส่วนที่เหลือ|ขอส่วนอื่น|อยากรู้ส่วนอื่น"
        r"|อยากรู้ต่อ|อยากดูต่อ|ขอต่อจากเดิม|อยากรู้เพิ่มเรื่องนี้|ยังอยากรู้"
        r"|อยากได้ส่วน|อยากถามต่อ|ต้องการทราบเพิ่ม|ต้องการเพิ่มเติม)",
        re.IGNORECASE,
    )
    _NEW_TOPIC_RE = re.compile(
        r"(ขอหัวข้อ(ใหม่|อื่น)|แนะนำหัวข้อ|มีเรื่องอื่น(อีก)?|เรื่องอื่น(อีก)?|หัวข้ออื่น|เรื่องไหนอีก|อยากรู้เรื่องอื่น|ต้(อง|แ)การรู้เรื่องอื่น|มีอะไรอีก(มั้ย|ไหม)?|แนะนำเรื่องอื่น)",
        re.IGNORECASE,
    )
    _FOLLOWUP_CONTEXTUAL_RE = re.compile(
        r"^(อันไหน|แบบไหน|กรณีไหน|ของ(ฉัน|ผม|หนู)|แบบ(ฉัน|ผม|หนู)|กรณีของ|ที่เหมาะกับ|ที่ใช้ได้กับ|สำหรับ(ฉัน|ผม|หนู|กรณีนี้|ประเภทนี้))",
        re.IGNORECASE,
    )
    # Short Thai interjections that are not legal questions → re-show menu
    _TH_INTERJECTION_RE = re.compile(
        r"^\s*(เอ้|เฮ้|เฮ|โอ้|โอ้โห|อ้าว|อ้าว|ว้าว|เออ|เอ่อ|อ่า|อ้า|อืม|อ๋อ|อ๋อ|เออนะ|เอ้าๆ|งั้นหรอ|งั้นเหรอ|จริงดิ|จริงเหรอ|ไม่ใช่เหรอ|เหรอ)\s*(ครับ|คับ|ค่ะ|คะ|นะ|นะครับ|นะคะ)?\s*$",
        re.IGNORECASE,
    )

    # --------------------------
    # Confirmation classifier
    # --------------------------
    _YES_CORE = (
        "ใช่", "ช่าย", "ไช่", "ใข่", "ชั่ย", "ชัย", "ชัยๆ",
        "ยืนยัน", "คอนเฟิร์ม", "confirm",
        "ถูกต้อง", "ถูกแล้ว", "โอเค", "ตกลง", "ใช่เลย", "ใช่แล้ว",
        "ได้", "ได้เลย", "ต้องการ",
        "เอา", "เอาเลย", "จัดไป", "ไปเลย",
        "yes", "yeah", "yep", "yup", "ok", "okay",
        "เยส", "เยป", "เย้ป",
        "งับ", "ค้าบ", "คั้บ", "เออ", "อือ", "อืม",
        "แน่นอน", "เลย", "เปลี่ยน",
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
        if state.messages and state.messages[-1].get("role") == "user" and (state.messages[-1].get("content") or "").strip() == text.strip():
            return
        state.messages.append({"role": "user", "content": text})

    def _add_assistant(self, state: ConversationState, text: str) -> None:
        state.messages = state.messages or []
        if not text:
            return
        if state.messages and state.messages[-1].get("role") == "assistant" and (state.messages[-1].get("content") or "").strip() == text.strip():
            return
        state.messages.append({"role": "assistant", "content": text})
        _LOG.debug("[Supervisor] assistant_msg_len=%d", len(text))

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

        # "ครับผม" = strong polite yes
        if re.search(r"ครับผม|ครับ\s*ผม", t):
            return {"yes": True, "no": False, "confidence": 0.92, "method": "det_krabphom"}

        # Extended/repeated particles like "จ้าา", "ค่ะๆ", "ครับๆ" are usually yes
        if re.fullmatch(r"(จ้า+|ครับ+ๆ*|ค่ะ+ๆ*|ใช่ๆ*)", t):
            return {"yes": True, "no": False, "confidence": 0.85, "method": "det_particle_yes"}

        # Pure filler without other signals → unclear (needs LLM)
        if re.fullmatch(r"(ครับ|คับ|ค่ะ|คะ)", t):
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
    # LLM helpers (confirm/style + greet prefix + topic picker + slot mapper)
    # --------------------------
    def _default_topic_picker_llm_call(self) -> Callable[[str, List[str], int, List[str]], dict]:
        """
        Pick k topics from candidates, given context hint.
        Return JSON: {"topics": ["...", ...], "confidence": 0.0-1.0}
        """
        # Use dedicated fast model + short timeout for topic_picker (non-critical, fail fast)
        topic_model = getattr(conf, "OPENROUTER_MODEL_TOPIC_PICKER", getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=topic_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=220,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(last_hint: str, candidates: List[str], k: int, banned: List[str]) -> dict:
            cand = [str(x).strip() for x in (candidates or []) if str(x).strip()]
            cand = cand[:40]
            banned2 = [str(x).strip() for x in (banned or []) if str(x).strip()]
            banned2 = banned2[:60]

            prompt = (
                "หน้าที่: เลือกหัวข้อเมนูจำนวน k ข้อ จากรายการ candidates\n"
                "เป้าหมาย:\n"
                "1) เกี่ยวข้องกับบริบท last_topic_hint ให้มากที่สุด\n"
                "2) หลากหลาย (อย่าเลือกหัวข้อที่ความหมายซ้ำกัน)\n"
                "3) เป็น 'หัวข้อการทำงาน/ใบอนุญาต/ขั้นตอน' ไม่ใช่เคสเฉพาะหรือชื่อหน่วยงานล้วนๆ\n"
                "ข้อห้าม:\n"
                "- ห้ามเลือกคำ generic/placeholder ใน banned\n"
                "- ห้ามเลือกเคสเฉพาะ เช่น 'กรณี...', 'ถ้า...', 'สำหรับ...' (ยกเว้นจำเป็นจริงๆ)\n"
                "- ห้ามเลือกชื่อหน่วยงานล้วนๆ เช่น กรม..., สำนักงาน..., เทศบาล..., อบต., อบจ., สำนักงานเขต (ยกเว้นจำเป็นจริงๆ)\n"
                "- ห้ามเลือกซ้ำ\n"
                "ให้ตอบเป็น JSON เท่านั้น:\n"
                '{ "topics": ["..."], "confidence": 0.0 }\n'
                f"last_topic_hint: {last_hint}\n"
                f"k: {int(k)}\n"
                f"banned: {banned2}\n"
                f"candidates: {cand}\n"
            )

            try:
                text = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/topic_picker").content.strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_confirm_llm_call(self) -> Callable[[str], dict]:
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=96,
            request_timeout=timeout,
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
                text = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm").content.strip()
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
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=96,
            request_timeout=timeout,
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
                text = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm").content.strip()
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
                text = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm").content.strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_slot_mapper_llm_call(self) -> Callable[[str, str, List[str]], dict]:
        """
        Map a free-text reply into one of pending_slot options.
        Return JSON: {"choice_index": 1..N or 0, "choice_text": "...", "confidence": 0.0-1.0}
        - deterministic (temperature=0)
        - NO hardcode: lets LLM interpret abbreviations like "กทม"
        """
        switch_model = getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL)
        timeout = int(getattr(conf, "LLM_REQUEST_TIMEOUT", 30))
        llm = ChatOpenAI(
            model=switch_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=120,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(slot_key: str, user_text: str, options: List[str]) -> dict:
            opts = [str(x).strip() for x in (options or []) if str(x).strip()][:12]
            prompt = (
                "หน้าที่: จับคู่ข้อความผู้ใช้ให้เข้ากับตัวเลือกที่ใกล้ที่สุด (เลือกได้ 1 ข้อ)\n"
                "กติกา:\n"
                "- ถ้าแมพได้ชัดเจน ให้คืน choice_index เป็นเลข 1..N และ choice_text เป็นข้อความของตัวเลือกนั้น\n"
                "- ถ้าไม่ชัดเจนจริงๆ ให้คืน choice_index=0 และ confidence ต่ำ\n"
                "- ห้ามเดาแบบสุ่ม\n"
                "ตอบเป็น JSON เท่านั้น:\n"
                '{"choice_index": 0, "choice_text": "", "confidence": 0.0}\n'
                f"slot_key: {slot_key}\n"
                f"user_text: {user_text}\n"
                f"options: {opts}\n"
            )
            try:
                text = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/llm").content.strip()
            except Exception:
                return {}
            text = self._strip_code_fences(text)
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        return _call

    def _default_fallback_intent_llm_call(self) -> Callable[[str, str, str], dict]:
        """
        LLM classifier for inputs that didn't match any deterministic rule.
        Return JSON: {"intent": "new_topic|elaborate|legal_question|greeting|unknown", "query": "", "confidence": 0.0}
        - new_topic:       user wants a new/different topic → show topic menu
        - elaborate:       user wants more detail on last answer
        - legal_question:  user is asking about restaurant business law/permit/tax
        - greeting:        greeting, thanks, farewell
        - unknown:         cannot determine → fallback to topic menu
        Uses fast topic-picker model (low latency).
        """
        topic_model = getattr(conf, "OPENROUTER_MODEL_TOPIC_PICKER", getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL))
        timeout = int(getattr(conf, "LLM_TOPIC_PICKER_TIMEOUT", 8))
        llm = ChatOpenAI(
            model=topic_model,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=150,
            request_timeout=timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        def _call(user_text: str, last_query: str, persona: str) -> dict:
            prompt = (
                "คุณคือ routing classifier สำหรับ AI ผู้ช่วยธุรกิจร้านอาหารไทย\n"
                "จงจำแนก intent จากข้อความผู้ใช้ด้านล่าง\n\n"
                f"user_text: {user_text}\n"
                f"last_legal_query: {last_query or '(none)'}\n"
                f"current_persona: {persona}\n\n"
                "Intent categories:\n"
                "- new_topic: ต้องการเปลี่ยนหัวข้อ / ขอหัวข้อแนะนำใหม่ / อยากรู้เรื่องอื่น\n"
                "- elaborate: ต้องการให้อธิบายเพิ่มเติมจากคำตอบหรือหัวข้อล่าสุด\n"
                "- legal_question: ถามเรื่องกฎหมาย/ใบอนุญาต/ภาษี/จดทะเบียน/ธุรกิจร้านอาหาร\n"
                "- greeting: ทักทาย/ขอบคุณ/ปิดบทสนทนา\n"
                "- unknown: ไม่เกี่ยวกับธุรกิจร้านอาหารและไม่สามารถระบุได้\n\n"
                "ตอบ JSON เท่านั้น:\n"
                '{"intent": "new_topic", "query": "", "confidence": 0.9}\n'
                "- query: ถ้า intent=legal_question ให้ใส่คำถามที่ชัดเจนขึ้น, ไม่งั้นเว้นว่าง"
            )
            try:
                text = llm_invoke(llm, [HumanMessage(content=prompt)], logger=_LOG, label="Supervisor/fallback_intent").content.strip()
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
    # Intent classification helpers
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
            if re.search(r"\b(academic|practical)\b|วิชาการ|ละเอียด|สั้น|กระชับ", t):
                return False
            return True
        if re.fullmatch(r"(เปลี่ยน|สลับ|ปรับ)\s*", t):
            return True
        return False

    def _infer_target_persona_from_text(self, s: str) -> Optional[str]:
        t = self._normalize_for_intent(s)
        if not t:
            return None

        if re.search(r"\bacademic\b", t):
            return "academic"
        if re.search(r"\bpractical\b", t):
            return "practical"

        if "วิชาการ" in t:
            return "academic"
        if any(h in t for h in self._TARGET_ACADEMIC_HINTS):
            return "academic"
        if any(h in t for h in self._TARGET_PRACTICAL_HINTS):
            return "practical"

        return None

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
        llm_topic_picker_call: Optional[Callable[[str, List[str], int, List[str]], dict]] = None,
        llm_slot_mapper_call: Optional[Callable[[str, str, List[str]], dict]] = None,
    ):
        self.retriever = retriever
        self._academic = AcademicPersonaService(retriever=retriever)
        self._practical = PracticalPersonaService(retriever=retriever)

        self.llm_confirm_call = llm_confirm_call or self._default_confirm_llm_call()
        self.llm_style_call = llm_style_call or self._default_style_llm_call()
        self.llm_greet_prefix_call = llm_greet_prefix_call or self._default_greet_prefix_llm_call()
        self.llm_topic_picker_call = llm_topic_picker_call or self._default_topic_picker_llm_call()

        # ✅ NEW: pending_slot free-text mapper (no hardcode)
        self.llm_slot_mapper_call = llm_slot_mapper_call or self._default_slot_mapper_llm_call()

        # ✅ NEW: LLM fallback intent classifier — replaces hardcoded error message
        self.llm_fallback_intent_call = self._default_fallback_intent_llm_call()

        self._rng = random.Random()

    # --------------------------
    # RNG: stable per session + moves each greeting
    # --------------------------
    def _get_session_seed(self, state: ConversationState) -> int:
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

        t = self._normalize_for_intent(raw)
        if len(t) <= 2:
            return False
        if self._looks_like_greeting_or_thanks(raw):
            return False

        return True

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
        key = str(pending.get("key") or "").strip()

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

        if idxs:
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

        # 1) exact/contains match against options (fast deterministic)
        raw_norm = self._normalize_for_intent(raw)
        for opt in options:
            s = str(opt).strip()
            if not s:
                continue
            s_norm = self._normalize_for_intent(s)
            if raw_norm == s_norm:
                return s, None
            if raw_norm and s_norm and (raw_norm in s_norm or s_norm in raw_norm):
                return s, None

        # 2) topic: accept any free text as topic
        if key == "topic":
            return raw.strip(), None

        # ✅ 3) non-topic: allow free-text mapping via LLM (no hardcode)
        # Example: location slot with options ["กรุงเทพฯ", "ต่างจังหวัด"] and user says "กทม"
        if isinstance(options, list) and len(options) >= 2 and self.llm_slot_mapper_call:
            try:
                res = self.llm_slot_mapper_call(key, raw.strip(), [str(x).strip() for x in options]) or {}
            except Exception:
                res = {}
            try:
                confv = float(res.get("confidence", 0.0) or 0.0)
            except Exception:
                confv = 0.0

            # accept only when confident
            if confv >= 0.60:
                idx = 0
                try:
                    idx = int(res.get("choice_index", 0) or 0)
                except Exception:
                    idx = 0
                if 1 <= idx <= len(options):
                    return str(options[idx - 1]).strip(), None

                choice_text = str(res.get("choice_text") or "").strip()
                if choice_text:
                    # map back to options if same/contains
                    ct = self._normalize_for_intent(choice_text)
                    for opt in options:
                        s = str(opt).strip()
                        if not s:
                            continue
                        s_norm = self._normalize_for_intent(s)
                        if ct == s_norm or (ct in s_norm) or (s_norm in ct):
                            return s, None

        # otherwise require numeric for non-topic pending slots
        return None, "กรุณาตอบเป็นตัวเลขตามตัวเลือกครับ"

    # --------------------------
    # Pending slot routing guards
    # --------------------------
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

        if self._looks_like_switch_without_target(user_input):
            return False
        if self._infer_target_persona_from_text(user_input) in {"academic", "practical"}:
            return False

        if self._looks_like_mode_status_query(user_input):
            return False

        if self._STYLE_LIKELY_RE.search(user_input or "") and not self._LIKELY_SELECTION_RE.match((user_input or "").strip()):
            if not self._ANY_NUMBER_RE.search(user_input or ""):
                return False

        if self._looks_like_greeting_or_thanks(user_input) or self._is_noise(user_input):
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

        state.context = state.context or {}
        if isinstance(pending, dict) and pending.get("key") == "topic":
            state.context["last_topic"] = str(mapped).strip()
            # Also set last_user_legal_query so Academic auto-replay works after mode switch
            state.context["last_user_legal_query"] = str(mapped).strip()
            # Save menu for one-turn recovery (lets user pick another topic by number)
            state.context["last_topic_menu"] = list(pending.get("options") or [])
        elif isinstance(pending, dict) and pending.get("key") and mapped:
            # Non-topic slot (e.g. registration_type=นิติบุคคล) → cross-persona slot memory
            # so Academic mode can read it via state.get_collected_slots()
            try:
                state.save_collected_slot(pending["key"], str(mapped))
            except Exception:
                pass

        state.context.pop("pending_slot", None)

        # Pre-retrieve docs for topic slots so the LLM has documents when called
        # with _internal=True (which skips the retrieval block inside persona.handle)
        if isinstance(pending, dict) and pending.get("key") == "topic" and mapped:
            try:
                state.current_docs = self._practical._retrieve_docs(str(mapped))
                state.last_retrieval_query = str(mapped)
                _LOG.info("[Supervisor] pre-retrieved %d docs for topic=%r", len(state.current_docs), str(mapped)[:40])
            except Exception as e:
                _LOG.warning("[Supervisor] pre-retrieve failed for topic=%r: %s", str(mapped)[:40], e)
                state.current_docs = []

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
    # Auto-return followup (unchanged)
    # --------------------------
    def _build_auto_return_followup(self, state: ConversationState) -> str:
        ctx = state.context or {}
        topic_hint = str(ctx.get("auto_return_topic_context") or "").strip()

        orig_hint = ctx.get("last_user_legal_query")
        if topic_hint:
            ctx["last_user_legal_query"] = topic_hint
            state.context = ctx

        try:
            topics = self._compose_menu_topics(state, size=self._MENU_SIZE)
        except Exception:
            topics = []

        if topic_hint:
            if orig_hint is not None:
                ctx["last_user_legal_query"] = orig_hint
            else:
                ctx.pop("last_user_legal_query", None)
            state.context = ctx

        if not topics:
            return ""

        lines = ["กลับมาโหมด Practical แล้วครับ มีเรื่องอื่นที่อยากสอบถามไหมครับ"]
        for i, topic in enumerate(topics, 1):
            lines.append(f"{i}) {topic}")
        lines.append("หรือพิมพ์คำถามได้เลยครับ")

        state.context["pending_slot"] = {"key": "topic", "options": topics, "allow_multi": False}
        state.context["main_menu_shown"] = True
        state.context["last_menu_topics"] = topics
        ctx.pop("auto_return_topic_context", None)

        _LOG.info("[Supervisor] auto_return_followup topics=%d hint=%r", len(topics), topic_hint[:40] if topic_hint else "")
        return "\n".join(lines)

    def _post_route_academic_auto_return(self, state: ConversationState, reply: str) -> Tuple[ConversationState, str]:
        ctx = state.context or {}
        auto_supervisor = bool(ctx.get("auto_return_after_academic_done", False))
        auto_academic = bool(ctx.get("auto_return_to_practical", False))

        if not (auto_supervisor or auto_academic):
            return state, reply

        if self._is_academic_intake_active(state):
            return state, reply

        origin = normalize_persona_id(ctx.get("switch_origin_persona") or "practical")
        if origin != "practical":
            origin = "practical"

        if normalize_persona_id(getattr(state, "persona_id", "") or "") != "academic":
            return state, reply

        state.persona_id = origin
        ctx["persona_id"] = origin
        ctx.pop("auto_return_after_academic_done", None)
        ctx.pop("auto_return_to_practical", None)
        ctx.pop("switch_origin_persona", None)
        state.context = ctx

        state.last_action = "auto_return_to_practical"

        # Mark that academic session is resumable (section_catalog / academic_question / docs still in context)
        state.context["academic_resume_available"] = True

        follow_up = self._build_auto_return_followup(state)
        if follow_up:
            combined = reply.rstrip() + "\n\n─────────────────\n" + follow_up
            return state, combined

        return state, reply

    # --------------------------
    # Greeting/Menu (unchanged from your current file)
    # --------------------------
    _MENU_SIZE = 5
    _POOL_MAX = 80
    _MENU_CANDIDATE_MAX = 30
    _LLM_PICK_MIN_CONF = 0.55

    _MENU_REQUIRE_KEYWORDS = (
        "ใบอนุญาต", "อนุญาต", "ขั้นตอน", "เอกสาร", "ค่าธรรมเนียม", "ระยะเวลา", "ช่องทาง",
        "ภาษี", "vat", "ภพ", "จดทะเบียน", "ทะเบียนพาณิชย์", "dbd",
        "ประกันสังคม", "กองทุน", "สุขาภิบาล", "เปิดร้าน", "ยื่นคำขอ", "คำขอ",
        "ใบกำกับภาษี", "ใบเสร็จ", "แบบฟอร์ม", "ฟอร์ม",
    )
    # NOTE: org-name fragments (สรรพากร, กรม, สำนักงาน) intentionally excluded —
    # they are caught by _looks_orgish() and must NOT grant menu_worthy status.
    _MENU_DETAILISH_PATTERNS = (
        r"^กรณี", r"^ถ้า", r"^สำหรับ", r"^เมื่อ", r"^หาก", r"^ในกรณี", r"^ข้อยกเว้น",
        r"^หมายเหตุ", r"^เพิ่มเติม", r"^ตัวอย่าง", r"^คำแนะนำ",
        r"ยกเว้น", r"ที่ได้รับ", r"กรณีพิเศษ", r"เฉพาะกรณี",
    )
    _MENU_REJECT_PATTERNS = (
        r"^\W+$",
        r"^https?://",
        r"@",
        r"\b(line|facebook|fb|ig|email)\b",
        r"\b\d{8,}\b",
    )

    _MENU_FALLBACK_TOPICS = [
        "ขอใบอนุญาตเปิดร้านอาหาร",
        "สุขาภิบาลอาหาร / อาหารสะอาด",
        "ภาษี VAT / ขอ ภพ.20",
        "จดทะเบียนพาณิชย์ / DBD",
        "เอกสารที่ต้องใช้ / เช็คลิสต์",
        "ค่าธรรมเนียม",
        "ระยะเวลาดำเนินการ",
        "ช่องทางยื่นคำขอ / หน่วยงาน",
        "ประกันสังคม (ขึ้นทะเบียนนายจ้าง)",
        "กองทุนเงินทดแทน",
    ]

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

        if t in {"-", "—", "–", "N/A", "n/a", "NA", "na"}:
            return ""
        if len(t) < 3:
            return ""
        if len(t) > 64:
            return ""

        t = re.sub(r"^\s*หน่วยงาน\s*[:：]\s*", "", t).strip()
        # Collapse consecutive Thai/ASCII word repetitions (e.g. "ทะเบียนทะเบียน" → "ทะเบียน")
        t = re.sub(r"([ก-๙a-zA-Z]{2,})\1+", r"\1", t)
        return t

    # NOTE: no \b — Thai has no spaces between words so \b never fires mid-string.
    # Prefix match alone is sufficient: any label starting with these IS an org name.
    _ORGISH_RE = re.compile(
        r"^(กรม|สำนักงาน|สำนัก|เทศบาล|อบต\.?|อบจ\.?|สำนักงานเขต"
        r"|กรุงเทพมหานคร|กทม\.?)"
    )

    def _looks_orgish(self, label: str) -> bool:
        l = (label or "").strip()
        if not l:
            return False
        return bool(self._ORGISH_RE.search(l))

    def _is_detailish_label(self, label: str) -> bool:
        l = (label or "").strip()
        if not l:
            return True
        low = self._normalize_for_intent(l)
        for pat in self._MENU_DETAILISH_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                return True
        return False

    def _passes_reject_patterns(self, label: str) -> bool:
        low = self._normalize_for_intent(label)
        for pat in self._MENU_REJECT_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                return False
        return True

    def _menu_keyword_score(self, label: str) -> int:
        low = self._normalize_for_intent(label)
        score = 0
        for kw in self._MENU_REQUIRE_KEYWORDS:
            if kw and kw.lower() in low:
                score += 1
        return score

    def _is_menu_worthy(self, label: str) -> bool:
        l = self._sanitize_topic_label(label)
        if not l:
            return False
        if not self._passes_reject_patterns(l):
            return False
        if self._is_detailish_label(l):
            return False
        if self._looks_orgish(l):
            return False
        if self._menu_keyword_score(l) >= 1:
            return True
        low = self._normalize_for_intent(l)
        if re.search(r"(ขั้นตอน|เอกสาร|ค่าธรรมเนียม|ระยะเวลา|ช่องทาง|ใบอนุญาต|ภาษี|จดทะเบียน)", low):
            return True
        return False

    def _topic_kind_weight(self, label: str, source_key: str) -> int:
        k = (source_key or "").strip().lower()
        l = (label or "").strip()

        if k in {"license_type", "ใบอนุญาต"}:
            return 5
        if k in {
            "operation_topic",
            "หัวข้อการดำเนินการย่อย",
            "operation_subtopic",
            "sub_operation_topic",
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

        if k in {"department", "หน่วยงาน"}:
            return 1 if self._ORGISH_RE.search(l) else 2

        return 2

    def _get_banned_topic_labels(self) -> List[str]:
        banned = set()
        for x in getattr(self, "_STOP_LABELS", set()) or set():
            if x:
                banned.add(str(x).strip())

        extra = {
            "หัวข้อหลัก", "หัวข้อ", "หัวข้อย่อย", "หมวด", "หมวดหมู่",
            "อื่นๆ", "อื่น ๆ", "ไม่ระบุ", "ทั่วไป",
        }
        for x in extra:
            banned.add(x)

        return sorted([b for b in banned if b])

    def _collect_topic_freq_from_docs(self, docs: List[Any]) -> Dict[str, int]:
        freq: Dict[str, int] = {}

        def _add(v: Any, source_key: str) -> None:
            s = self._sanitize_topic_label(str(v) if v is not None else "")
            if not s:
                return

            if self._is_detailish_label(s):
                return

            if not self._is_menu_worthy(s):
                if self._looks_orgish(s):
                    return
                if len(s) <= 10:
                    w = 1
                else:
                    return
            else:
                w = self._topic_kind_weight(s, source_key=source_key)

            freq[s] = freq.get(s, 0) + int(w)

        LICENSE_KEYS = ["license_type", "ใบอนุญาต"]
        OP_BY_DEPT_KEYS = ["operation_by_department", "operation_action", "action_by_department", "operation_process", "การดำเนินการ ตามหน่วยงาน", "การดำเนินการตามหน่วยงาน"]
        OP_SUB_KEYS = ["operation_subtopic", "sub_operation_topic", "subtopic", "หัวข้อการดำเนินการย่อย"]
        OP_TOPIC_KEYS = ["operation_topic"]
        DEPT_KEYS = ["department", "หน่วยงาน"]

        for d in (docs or []):
            md = getattr(d, "metadata", {}) or {}

            for k in LICENSE_KEYS:
                _add(md.get(k), source_key=k)

            for k in OP_BY_DEPT_KEYS:
                _add(md.get(k), source_key=k)

            for k in OP_SUB_KEYS:
                _add(md.get(k), source_key=k)

            for k in OP_TOPIC_KEYS:
                _add(md.get(k), source_key=k)

            for k in DEPT_KEYS:
                _add(md.get(k), source_key=k)

        return freq

    def _build_topic_pool_from_corpus(self, state: ConversationState) -> List[Tuple[str, int]]:
        queries = [
            "ใบอนุญาต เปิดร้านอาหาร เทศบาล สำนักงานเขต สุขาภิบาลอาหาร",
            "ภาษี VAT ภพ.20 ใบกำกับภาษี กรมสรรพากร จด VAT",
            "จดทะเบียนพาณิชย์ นิติบุคคล DBD กรมพัฒนาธุรกิจการค้า หนังสือรับรอง",
            "ประกันสังคม ขึ้นทะเบียนนายจ้าง ลูกจ้าง กองทุนเงินทดแทน",
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

        items = sorted(merged.items(), key=lambda x: (-x[1], x[0]))
        pool = items[: self._POOL_MAX]

        if len(pool) < 12:
            existing = {k for k, _ in pool}
            for i, t in enumerate(self._MENU_FALLBACK_TOPICS):
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in existing and self._is_menu_worthy(t2):
                    pool.append((t2, 10 - (i // 2)))

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
                    ts = str(t)
                    if not self._is_menu_worthy(ts):
                        continue
                    out.append((ts, int(w)))
                except Exception:
                    continue
            if out:
                return out
        return self._build_topic_pool_from_corpus(state)

    def _get_last_topic_hint(self, state: ConversationState) -> str:
        ctx = state.context or {}

        last_legal = str(ctx.get("last_user_legal_query") or "").strip()
        if last_legal:
            return last_legal[:80]

        last = str(ctx.get("last_topic") or "").strip()
        if last:
            return last

        last_q = str(getattr(state, "last_retrieval_query", "") or "").strip()
        return last_q[:60] if last_q else ""

    def _weighted_sample_no_replace(self, pool: List[Tuple[str, int]], k: int, rng: random.Random) -> List[str]:
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

    def _dedupe_semantic_loose(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in items or []:
            s = self._sanitize_topic_label(x)
            if not s:
                continue
            if not self._is_menu_worthy(s):
                continue

            key = self._normalize_for_intent(s)
            key = re.sub(r"(การ|การทำ|การขอ|ขอ|ยื่น|ขึ้นทะเบียน|จดทะเบียน|ทะเบียน|ใบอนุญาต)\s*", "", key).strip()
            key = re.sub(r"\s+", " ", key)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _llm_pick_menu_topics(self, state: ConversationState, last_hint: str, candidates: List[str], k: int) -> List[str]:
        if not self.llm_topic_picker_call:
            return []

        banned = self._get_banned_topic_labels()
        cand = [c for c in self._dedupe_semantic_loose(candidates) if c][: self._MENU_CANDIDATE_MAX]

        if len(cand) <= k:
            return cand[:k]

        res: Dict[str, Any] = {}
        try:
            res = self.llm_topic_picker_call(last_hint or "", cand, int(k), banned) or {}
        except Exception:
            res = {}

        try:
            confv = float(res.get("confidence", 0.0) or 0.0)
        except Exception:
            confv = 0.0

        topics = res.get("topics", [])
        if not isinstance(topics, list):
            topics = []

        picked: List[str] = []
        banned_set = set([b.strip().lower() for b in banned if b and b.strip()])
        for x in topics:
            s = self._sanitize_topic_label(str(x))
            if not s:
                continue
            if s.strip().lower() in banned_set:
                continue
            if not self._is_menu_worthy(s):
                continue
            picked.append(s)

        picked = self._dedupe_semantic_loose(picked)

        non_org = [t for t in picked if not self._looks_orgish(t)]
        org = [t for t in picked if self._looks_orgish(t)]
        final = non_org[:k]
        if len(final) < k:
            final.extend([t for t in org if t not in final][: (k - len(final))])

        if confv < self._LLM_PICK_MIN_CONF:
            return []
        if len(final) < max(3, min(k, 4)):
            return []

        return final[:k]

    def _compose_menu_topics(self, state: ConversationState, size: int) -> List[str]:
        pool = self._get_topic_pool(state)
        size = max(1, int(size))

        rng = self._get_rng(state)
        last_hint = self._get_last_topic_hint(state).strip()

        related_target = 2 if size >= 5 else 1
        related = self._related_topics_from_last(state, need=related_target)

        related = [self._sanitize_topic_label(x) for x in related]
        related = [x for x in related if x and self._is_menu_worthy(x)]
        related = self._dedupe_semantic_loose(related)

        related_set = set(related)
        pool_fresh = [(t, w) for (t, w) in pool if t not in related_set]
        fresh_need = max(0, size - len(related))
        fresh = self._weighted_sample_no_replace(pool_fresh, k=max(fresh_need, size), rng=rng)

        fresh = [self._sanitize_topic_label(x) for x in fresh]
        fresh = [x for x in fresh if x and x not in related_set and self._is_menu_worthy(x)]
        fresh = self._dedupe_semantic_loose(fresh)

        candidates: List[str] = []
        for t in related:
            if t and t not in candidates:
                candidates.append(t)
        for t in fresh:
            if t and t not in candidates:
                candidates.append(t)
            if len(candidates) >= self._MENU_CANDIDATE_MAX:
                break

        if len(candidates) < min(12, self._MENU_CANDIDATE_MAX):
            for t, _ in pool:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in candidates and self._is_menu_worthy(t2):
                    candidates.append(t2)
                if len(candidates) >= self._MENU_CANDIDATE_MAX:
                    break

        picked = self._llm_pick_menu_topics(state, last_hint=last_hint, candidates=candidates, k=size)

        if not picked:
            combined: List[str] = []
            for t in related + fresh:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in combined and self._is_menu_worthy(t2):
                    combined.append(t2)
                if len(combined) >= size:
                    break

            if len(combined) < size:
                for t, _ in pool:
                    t2 = self._sanitize_topic_label(t)
                    if t2 and t2 not in combined and self._is_menu_worthy(t2):
                        combined.append(t2)
                    if len(combined) >= size:
                        break

            picked = combined[:size]

        last_menu = (state.context or {}).get("last_menu_topics")
        if isinstance(last_menu, list) and len(last_menu) >= 3:
            overlap = len(set(last_menu).intersection(set(picked)))
            if overlap >= 4 and len(pool_fresh) >= size:
                fresh2 = self._weighted_sample_no_replace(pool_fresh, k=self._MENU_CANDIDATE_MAX, rng=rng)
                fresh2 = [self._sanitize_topic_label(x) for x in fresh2]
                fresh2 = [x for x in fresh2 if x and self._is_menu_worthy(x)]
                fresh2 = self._dedupe_semantic_loose(fresh2)

                candidates2: List[str] = []
                for t in related:
                    if t and t not in candidates2:
                        candidates2.append(t)
                for t in fresh2:
                    if t and t not in candidates2:
                        candidates2.append(t)
                    if len(candidates2) >= self._MENU_CANDIDATE_MAX:
                        break

                picked2 = self._llm_pick_menu_topics(state, last_hint=last_hint, candidates=candidates2, k=size)
                if picked2 and len(set(picked2).intersection(set(last_menu))) < overlap:
                    picked = picked2

        picked = self._dedupe_semantic_loose(picked)

        if len(picked) < size:
            for t in self._MENU_FALLBACK_TOPICS:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in picked and self._is_menu_worthy(t2):
                    picked.append(t2)
                if len(picked) >= size:
                    break

        if len(picked) < size:
            for t, _ in pool:
                t2 = self._sanitize_topic_label(t)
                if t2 and t2 not in picked and self._is_menu_worthy(t2):
                    picked.append(t2)
                if len(picked) >= size:
                    break

        return picked[:size]

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
                prefix = "ตอนนี้อยากให้ช่วยเรื่องไหนครับ"

        prefix = re.sub(r"\s+", " ", prefix).strip()
        return self._normalize_male(prefix)

    def _render_greeting_with_menu(self, state: ConversationState, kind: str, menu_topics: List[str], include_intro: bool) -> str:
        prefix = self._get_prefix_llm(kind, state, include_intro=include_intro)
        menu = self._format_numbered_options(menu_topics, max_items=9)
        msg = (prefix.rstrip() + "\n" + menu).strip()
        return self._normalize_male(msg)

    def _handle_greeting(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
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
    # Switch helpers (unchanged)
    # --------------------------
    def _mark_auto_return_if_practical_to_academic(self, state: ConversationState, target_pid: str) -> None:
        origin = normalize_persona_id(state.persona_id)
        state.context = state.context or {}
        state.context["switch_origin_persona"] = origin

        if origin == "practical" and normalize_persona_id(target_pid) == "academic":
            state.context["auto_return_after_academic_done"] = True
        else:
            state.context.pop("auto_return_after_academic_done", None)

    def _other_persona(self, pid: str) -> str:
        pid2 = normalize_persona_id(pid)
        return "academic" if pid2 == "practical" else "practical"

    def _enter_switch_confirmation(self, state: ConversationState, target_pid: str, replay_user_input: str = "") -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        state.context["pending_persona"] = normalize_persona_id(target_pid)
        state.context["pending_replay_user_input"] = replay_user_input or ""
        state.context["awaiting_persona_confirmation"] = True
        state.context["confirm_tries"] = 0

        self._mark_auto_return_if_practical_to_academic(state, target_pid)

        current = normalize_persona_id(state.persona_id)
        target = normalize_persona_id(target_pid)

        msg = self._normalize_male(f"ตอนนี้อยู่โหมด {current} ครับ ต้องการเปลี่ยนไปโหมด {target} ใช่ไหมครับ")
        self._add_assistant(state, msg)
        state.last_action = "persona_switch_confirm"
        return state, msg

    def _propose_toggle_switch(self, state: ConversationState, reason: str, replay_user_input: str = "") -> Tuple[ConversationState, str]:
        cur = normalize_persona_id(state.persona_id)
        target = self._other_persona(cur)
        state.context = state.context or {}
        state.context["switch_reason"] = reason
        return self._enter_switch_confirmation(state, target_pid=target, replay_user_input=replay_user_input)

    def _propose_switch_to_target(self, state: ConversationState, target: str, reason: str, replay_user_input: str = "") -> Tuple[ConversationState, str]:
        cur = normalize_persona_id(state.persona_id)
        target2 = normalize_persona_id(target)

        if cur == target2:
            other = self._other_persona(cur)
            msg = self._normalize_male(f"ตอนนี้อยู่โหมด {cur} อยู่แล้วครับ ต้องการสลับไปโหมด {other} ไหมครับ")
            self._add_assistant(state, msg)
            state.last_action = "persona_already_in_mode"
            state.context = state.context or {}
            state.context["pending_persona"] = other
            state.context["pending_replay_user_input"] = replay_user_input or ""
            state.context["awaiting_persona_confirmation"] = True
            state.context["confirm_tries"] = 0
            self._mark_auto_return_if_practical_to_academic(state, other)
            return state, msg

        state.context = state.context or {}
        state.context["switch_reason"] = reason
        return self._enter_switch_confirmation(state, target_pid=target2, replay_user_input=replay_user_input)

    # --------------------------
    # Persona/Profile sync
    # --------------------------
    def _sync_persona_and_profile(self, state: ConversationState) -> None:
        state.context = state.context or {}
        state.context["supervisor_owns_menu"] = True

        raw_pid = ""
        if hasattr(state, "persona_id"):
            raw_pid = str(getattr(state, "persona_id") or "")
        if not raw_pid:
            raw_pid = str(state.context.get("persona_id") or "")
        if not raw_pid:
            raw_pid = "practical"

        pid = normalize_persona_id(raw_pid)

        if hasattr(state, "persona_id"):
            state.persona_id = pid
        state.context["persona_id"] = pid

        prof = state.context.get("persona_profile")
        prof_pid = ""
        if isinstance(prof, dict):
            prof_pid = str(prof.get("persona_id") or "")

        if not isinstance(prof, dict) or normalize_persona_id(prof_pid or pid) != pid:
            profile = build_strict_profile(pid)
            state.context["persona_profile"] = profile
        else:
            profile = prof

        try:
            apply_persona_profile(state, profile)
        except Exception:
            return

    # --------------------------
    # Main handle
    # --------------------------
    def handle(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        st, reply = self._handle_inner(state, user_input)
        if hasattr(st, "trim_messages"):
            st.trim_messages(keep_last=12)
        # Trim large context fields that bloat the prompt (topic_pool can be 100+ items)
        if st.context and len(st.context.get("topic_pool") or []) > 10:
            st.context["topic_pool"] = st.context["topic_pool"][:10]
        return st, reply

    def _handle_inner(self, state: ConversationState, user_input: str) -> Tuple[ConversationState, str]:
        state.context = state.context or {}
        self._sync_persona_and_profile(state)

        raw = (user_input or "")
        raw_stripped = raw.strip()

        if not state.context.get("did_greet") and not raw_stripped and not self._is_academic_intake_active(state):
            state.context["did_greet"] = True
            return self._handle_greeting(state, user_input="")

        if not state.context.get("did_greet"):
            state.context["did_greet"] = True

        if raw_stripped:
            self._add_user(state, raw)

        # 2.1) Academic intake lock
        if self._is_academic_intake_active(state) and not state.context.get("awaiting_persona_confirmation"):
            st2, reply = self._academic.handle(state, raw_stripped, _internal=False)
            st2, reply = self._post_route_academic_auto_return(st2, reply)

            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            st2.last_action = "academic_intake_route"
            return st2, reply

        # 2.2) awaiting persona confirmation
        if state.context.get("awaiting_persona_confirmation"):
            if not raw_stripped:
                msg = self._normalize_male("ขอคำตอบยืนยันอีกครั้งได้ไหมครับ")
                self._add_assistant(state, msg)
                state.last_action = "persona_switch_confirm_unclear"
                return state, msg

            det = self._classify_yes_no_det(raw_stripped)

            yes = bool(det.get("yes"))
            no = bool(det.get("no"))
            confv = float(det.get("confidence", 0.0) or 0.0)

            if (not yes and not no) or confv < 0.55:
                llm_res: Dict[str, Any] = {}
                try:
                    llm_res = self.llm_confirm_call(raw_stripped) or {}
                except Exception:
                    llm_res = {}

                try:
                    llm_conf = float(llm_res.get("confidence", 0.0) or 0.0)
                except Exception:
                    llm_conf = 0.0

                if llm_conf >= 0.55:
                    yes = bool(llm_res.get("yes", False))
                    no = bool(llm_res.get("no", False))

            if yes and not no:
                target = normalize_persona_id(state.context.get("pending_persona") or "practical")

                state.persona_id = target
                state.context["persona_id"] = target

                state.context.pop("awaiting_persona_confirmation", None)
                state.context.pop("pending_persona", None)
                replay = str(state.context.pop("pending_replay_user_input", "") or "").strip()

                msg = self._normalize_male(f"ตอนนี้เป็นโหมด {target} แล้วครับ")
                self._add_assistant(state, msg)
                state.last_action = "persona_switched"

                # Only replay original input if it was a legal/content question —
                # NOT a switch trigger (e.g. "ไปบอทอีกตัว") to prevent infinite switch loop.
                if replay and self._looks_like_legal_question(replay) and not self._looks_like_switch_without_target(replay):
                    st2, reply2 = self._handle_inner(state, replay)
                    return st2, reply2

                # When switching to Academic with no replay, auto-continue with last legal topic
                if target == "academic" and not replay:
                    ctx_now = state.context or {}
                    last_q = ctx_now.get("last_user_legal_query", "").strip() or ctx_now.get("last_topic", "").strip()
                    if last_q:
                        _LOG.info("[Supervisor] academic_switch auto-replay last_q=%r", last_q[:40])
                        st2, reply2 = self._handle_inner(state, last_q)
                        return st2, msg + "\n\n" + reply2

                return state, msg

            if no and not yes:
                state.context.pop("awaiting_persona_confirmation", None)
                state.context.pop("pending_persona", None)
                state.context.pop("pending_replay_user_input", None)

                msg = self._normalize_male("โอเคครับ ไม่เปลี่ยนโหมด")
                self._add_assistant(state, msg)
                state.last_action = "persona_switch_cancelled"
                return state, msg

            msg = self._normalize_male("ยังไม่ชัดครับ ตอบ “ใช่/ไม่” (หรือ 1/2) ได้ไหมครับ")
            self._add_assistant(state, msg)
            state.last_action = "persona_switch_confirm_unclear"
            return state, msg

        # 2.2b) Academic resume: user wants to continue a previous academic session (remaining sections)
        # Triggered only when academic_resume_available=True (set after auto-return) AND input matches
        # continuation intent (e.g. "ทั้งหมด", "ขอส่วนอื่น", "อยากรู้ต่อ").
        if state.context.get("academic_resume_available") and raw_stripped and self._ACADEMIC_RESUME_RE.search(raw_stripped):
            _LOG.info("[Supervisor] academic_resume triggered input=%r", raw_stripped[:40])
            state.context.pop("academic_resume_available", None)
            state.context.pop("pending_slot", None)  # clear practical topics menu
            state.persona_id = "academic"
            state.context["persona_id"] = "academic"
            # Reset FSM to awaiting_sections — section_catalog is still in context
            flow = dict(state.context.get("academic_flow") or {})
            flow["stage"] = "awaiting_sections"
            state.context["academic_flow"] = flow
            st2, reply = self._academic.handle(state, raw_stripped, _internal=False)
            st2, reply = self._post_route_academic_auto_return(st2, reply)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            st2.last_action = "academic_resume"
            return st2, reply

        # 2.3) style request -> propose switch
        style = self._infer_user_style_request_hybrid(raw_stripped)
        if style.get("wants_long") or style.get("wants_short"):
            target = "academic" if style.get("wants_long") else "practical"
            replay = raw_stripped if self._looks_like_legal_question(raw_stripped) else ""
            return self._propose_switch_to_target(
                state,
                target=target,
                reason=f"style:{style.get('method','unknown')}",
                replay_user_input=replay,
            )

        # 2.4) explicit target in text + switch verb -> propose switch
        explicit_target = self._infer_target_persona_from_text(raw_stripped)
        if explicit_target in {"academic", "practical"} and any(v in self._normalize_for_intent(raw_stripped) for v in self._SWITCH_VERBS):
            return self._propose_switch_to_target(
                state,
                target=explicit_target,
                reason="explicit_target",
                replay_user_input=raw_stripped,
            )

        # 2.5) switch without target -> propose toggle
        if self._looks_like_switch_without_target(raw_stripped):
            return self._propose_toggle_switch(
                state,
                reason="switch_without_target",
                replay_user_input=raw_stripped,
            )

        # 2.5.5) Number typed but no pending_slot → try to recover from last_topic_menu
        if not self._has_pending_slot(state) and self._LIKELY_SELECTION_RE.match(raw_stripped):
            last_menu = (state.context or {}).get("last_topic_menu") or []
            if last_menu:
                idxs = self._parse_indices(raw_stripped)
                valid = [i for i in idxs if 1 <= i <= len(last_menu)]
                if valid:
                    # Restore pending_slot temporarily so 2.6 can route it normally
                    state.context["pending_slot"] = {
                        "key": "topic",
                        "options": last_menu,
                        "allow_multi": len(valid) > 1,
                    }
                    _LOG.info("[Supervisor] restored last_topic_menu for number input %r", raw_stripped)

        # 2.6) pending slot route
        if self._should_route_pending_slot_now(state, raw_stripped):
            return self._route_pending_slot_to_persona(state, raw_stripped)

        # 2.7) greeting/noise -> always show refreshed menu
        if self._looks_like_greeting_or_thanks(raw_stripped) or self._is_noise(raw_stripped) or not raw_stripped:
            return self._handle_greeting(state, user_input=raw_stripped)

        # 2.8) mode status
        if self._looks_like_mode_status_query(raw_stripped):
            pid = normalize_persona_id(state.persona_id)
            msg = self._normalize_male(f"ตอนนี้เป็นโหมด {pid} ครับ")
            self._add_assistant(state, msg)
            state.last_action = "mode_status"
            return state, msg

        # 2.9) legal routing
        if self._looks_like_legal_question(raw_stripped):
            state.context["last_user_legal_query"] = raw_stripped

            pid = normalize_persona_id(state.persona_id)
            if pid == "academic":
                st2, reply = self._academic.handle(state, raw_stripped, _internal=False)
                st2, reply = self._post_route_academic_auto_return(st2, reply)

                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "academic_answer"
                return st2, reply

            self._ensure_practical_retrieval_for_legal(state, raw_stripped)
            st2, reply = self._practical.handle(state, raw_stripped, _internal=False)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            st2.last_action = "practical_answer"
            return st2, reply

        # 3) Short interjection → re-show menu gracefully instead of error
        if self._TH_INTERJECTION_RE.match(raw_stripped):
            _LOG.info("[Supervisor] interjection→greeting persona=%s input=%r", getattr(state, "persona_id", "?"), raw_stripped[:30])
            return self._handle_greeting(state, raw_stripped)

        # 3.1) "ขอหัวข้อใหม่", "มีเรื่องอื่นอีกไหม", "แนะนำหัวข้อ" → refresh menu
        if self._NEW_TOPIC_RE.search(raw_stripped):
            _LOG.info("[Supervisor] new_topic_request→greeting input=%r", raw_stripped[:40])
            return self._handle_greeting(state, raw_stripped)

        # 3.2) "อธิบายมากกว่านี้", "ขยายความ", "เพิ่มเติมอีก" → re-route with last legal query
        if self._ELABORATE_RE.search(raw_stripped):
            last_q = (state.context or {}).get("last_user_legal_query", "").strip()
            if last_q:
                _LOG.info("[Supervisor] elaborate→practical last_q=%r", last_q[:40])
                self._ensure_practical_retrieval_for_legal(state, last_q)
                elaborate_input = f"อธิบายรายละเอียดเพิ่มเติม: {last_q}"
                st2, reply = self._practical.handle(state, elaborate_input, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_elaborate"
                return st2, reply

        # 3.3) "อันไหนเหมาะกับฉัน", "สำหรับกรณีฉัน" → contextual follow-up with last legal query
        if self._FOLLOWUP_CONTEXTUAL_RE.search(raw_stripped):
            last_q = (state.context or {}).get("last_user_legal_query", "").strip()
            if last_q:
                _LOG.info("[Supervisor] contextual_followup→practical input=%r last_q=%r", raw_stripped[:30], last_q[:30])
                self._ensure_practical_retrieval_for_legal(state, last_q)
                combined = f"{raw_stripped} (เกี่ยวกับ: {last_q})"
                st2, reply = self._practical.handle(state, combined, _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "practical_contextual_followup"
                return st2, reply

        # 4) LLM fallback intent classifier — no hardcode, no dead-end error message
        _LOG.info("[Supervisor] fallback_intent_llm persona=%s input=%r", getattr(state, "persona_id", "?"), raw_stripped[:60])
        intent_res: Dict[str, Any] = {}
        try:
            last_q_fb = (state.context or {}).get("last_user_legal_query", "")
            persona_fb = normalize_persona_id(getattr(state, "persona_id", "practical"))
            intent_res = self.llm_fallback_intent_call(raw_stripped, last_q_fb, persona_fb) or {}
        except Exception as _e:
            _LOG.warning("[Supervisor] fallback_intent_llm error: %s", _e)
            intent_res = {}

        fallback_intent = (intent_res.get("intent") or "unknown").strip().lower()
        _LOG.info("[Supervisor] fallback_intent=%r confidence=%s", fallback_intent, intent_res.get("confidence"))

        if fallback_intent == "new_topic":
            _LOG.info("[Supervisor] fallback_llm→new_topic input=%r", raw_stripped[:40])
            return self._handle_greeting(state, raw_stripped)

        if fallback_intent == "elaborate":
            last_q_fb2 = (state.context or {}).get("last_user_legal_query", "").strip()
            if last_q_fb2:
                _LOG.info("[Supervisor] fallback_llm→elaborate last_q=%r", last_q_fb2[:40])
                self._ensure_practical_retrieval_for_legal(state, last_q_fb2)
                st2, reply = self._practical.handle(state, f"อธิบายรายละเอียดเพิ่มเติม: {last_q_fb2}", _internal=False)
                reply = self._normalize_male(reply)
                self._add_assistant(st2, reply)
                st2.last_action = "fallback_llm_elaborate"
                return st2, reply

        if fallback_intent == "legal_question":
            q_fb = (intent_res.get("query") or raw_stripped).strip()
            _LOG.info("[Supervisor] fallback_llm→legal_question q=%r", q_fb[:40])
            state.context["last_user_legal_query"] = q_fb
            self._ensure_practical_retrieval_for_legal(state, q_fb)
            st2, reply = self._practical.handle(state, q_fb, _internal=False)
            reply = self._normalize_male(reply)
            self._add_assistant(st2, reply)
            st2.last_action = "fallback_llm_legal"
            return st2, reply

        if fallback_intent == "greeting":
            return self._handle_greeting(state, raw_stripped)

        # Truly unroutable → show topic menu (better than dead-end error)
        _LOG.warning("[Supervisor] fallback_safe_return persona=%s input=%r", getattr(state, "persona_id", "?"), raw_stripped[:60])
        state.last_action = "fallback_safe_return"
        return self._handle_greeting(state, raw_stripped)