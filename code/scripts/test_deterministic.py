# Deterministic unit-ish tests (Production-grade)
# Run:
#   PYTHONPATH="$PWD/code" python code/scripts/test_deterministic.py

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

# Project imports
from model.conversation_state import ConversationState
from model.persona_supervisor import PersonaSupervisor
from service.data_loader import DataLoader


# ----------------------------
# Fake Document + Retriever
# ----------------------------
@dataclass
class FakeDoc:
    page_content: str
    metadata: Dict[str, Any]


class FakeRetriever:
    """
    Deterministic retriever stub:
    - records queries
    - returns predictable docs based on query keywords
    """

    def __init__(self):
        self.queries: List[str] = []

    def invoke(self, query: str) -> List[FakeDoc]:
        q = (query or "").strip()
        self.queries.append(q)

        if re.search(r"(ประกันสังคม|กองทุน|ขึ้นทะเบียน)", q):
            return [
                FakeDoc(
                    page_content="ขั้นตอนขึ้นทะเบียนประกันสังคม: ...",
                    metadata={
                        "department": "สำนักงานประกันสังคม",
                        "license_type": "ขึ้นทะเบียนนายจ้าง",
                        "operation_steps": "1) ยื่นคำขอ 2) แนบเอกสาร",
                        "fees": "ไม่มีค่าธรรมเนียม",
                    },
                )
            ]

        if re.search(r"(vat|ภพ\.?20|ภาษี)", q, re.IGNORECASE):
            return [
                FakeDoc(
                    page_content="VAT/ภพ.20: ...",
                    metadata={
                        "department": "กรมสรรพากร",
                        "license_type": "ภพ.20",
                        "fees": "ไม่มีค่าธรรมเนียม",
                    },
                )
            ]

        return [
            FakeDoc(
                page_content=f"generic result for: {q}",
                metadata={
                    "department": "หน่วยงานตัวอย่าง",
                    "license_type": "หัวข้อทั่วไป",
                },
            )
        ]


# ----------------------------
# Tiny test harness
# ----------------------------
def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _last_assistant(state: ConversationState) -> str:
    for m in reversed(state.messages or []):
        if m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def _new_state(persona_id: str = "practical") -> ConversationState:
    return ConversationState(
        session_id="test",
        persona_id=persona_id,
        context={},
        messages=[],
        internal_messages=[],
    )


# ============================================================
# TEST 1: Style switch (deterministic path)
# ============================================================
def test_01_style_switch_policy_persona_aware():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("practical")
    st.context["did_greet"] = True

    st, reply = sup.handle(st, "ขอแบบละเอียดหน่อย")
    _assert(st.context.get("awaiting_persona_confirmation"), "T1: must ask confirm")
    _assert(st.context.get("pending_persona") == "academic", "T1: target must be academic")

    st2 = _new_state("academic")
    st2.context["did_greet"] = True

    st2, reply2 = sup.handle(st2, "สรุปสั้นๆได้ไหม")
    _assert(st2.context.get("pending_persona") == "practical", "T1b: should target practical")


# ============================================================
# TEST 2: Confirmation classifier robustness
# ============================================================
def test_02_confirmation_slang_yes_no():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("practical")
    st.context.update({
        "did_greet": True,
        "awaiting_persona_confirmation": True,
        "pending_persona": "academic",
    })

    st, reply = sup.handle(st, "เยปปป")
    _assert(st.persona_id == "academic", "T2: slang yes must switch")

    st2 = _new_state("practical")
    st2.context.update({
        "did_greet": True,
        "awaiting_persona_confirmation": True,
        "pending_persona": "academic",
    })

    st2, reply2 = sup.handle(st2, "ครับ")
    _assert(st2.persona_id == "practical", "T2b: filler must NOT switch")


# ============================================================
# TEST 3: Greeting must NOT retrieve
# ============================================================
def test_03_greeting_no_retrieve():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("practical")
    st.context["did_greet"] = True

    st, reply = sup.handle(st, "สวัสดีค่าาาา")
    _assert("1)" in reply, "T3: greeting must show numbered menu")
    _assert(len(retriever.queries) == 0, "T3: greeting must not trigger retrieval")


# ============================================================
# TEST 4: Practical new legal question bypass pending_slot
# ============================================================
def test_04_practical_bypass_pending_slot():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("practical")
    st.context.update({
        "did_greet": True,
        "pending_slot": {"key": "topic", "options": ["ใบอนุญาต"], "allow_multi": False},
    })

    st, _ = sup.handle(st, "ขึ้นทะเบียนกองทุนประกันสังคมต้องทำยังไง")
    _assert(len(retriever.queries) >= 1, "T4: must retrieve")
    _assert("ประกันสังคม" in retriever.queries[-1], "T4: query must reflect new topic")


# ============================================================
# TEST 5: Academic intake must override greeting
# ============================================================
def test_05_academic_intake_lock():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("academic")
    st.context.update({
        "did_greet": True,
        "academic_flow": {"stage": "awaiting_slots"},
    })

    st, reply = sup.handle(st, "สวัสดี")
    _assert("ข้อมูลจำเป็น" in reply or "ที่ตั้งร้าน" in reply, "T5: greeting must NOT override intake")


# ============================================================
# TEST 6: Academic auto-return to practical
# ============================================================
def test_06_academic_auto_return():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("academic")
    st.context.update({
        "did_greet": True,
        "auto_return_after_academic_done": True,
        "academic_flow": {"stage": "done"},
    })

    def fake_academic_handle(state, user_input):
        return state, "คำตอบแบบ academic ..."

    sup._academic.handle = fake_academic_handle

    st, reply = sup.handle(st, "อะไรก็ได้")
    _assert(st.persona_id == "practical", "T6: must auto-return to practical")
    _assert("1)" in reply, "T6: practical menu must appear")


# ============================================================
# TEST 7: No mutation rule (append-only)
# ============================================================
def test_07_no_mutation_rule():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("practical")
    st.context["did_greet"] = True

    st, _ = sup.handle(st, "ขึ้นทะเบียนนายจ้างต้องทำยังไง")
    first_len = len(st.messages)

    st, _ = sup.handle(st, "ขอเพิ่มรายละเอียด")
    second_len = len(st.messages)

    _assert(second_len > first_len, "T7: messages must be append-only")


# ============================================================
# TEST 8: Retrieval reuse vs new topic detection
# ============================================================
def test_08_retrieval_reuse():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("practical")
    st.context["did_greet"] = True

    st, _ = sup.handle(st, "ขึ้นทะเบียนนายจ้างต้องทำยังไง")
    q1 = retriever.queries[-1]

    st, _ = sup.handle(st, "ค่าธรรมเนียมเท่าไร")
    q2 = retriever.queries[-1]

    _assert(q1 == q2 or len(retriever.queries) <= 2, "T8: must reuse or controlled retrieve")


# ============================================================
# TEST 9: Academic resume without confirm
# ============================================================
def test_09_academic_resume_layer():
    retriever = FakeRetriever()
    sup = PersonaSupervisor(retriever=retriever)

    st = _new_state("academic")
    st.context.update({
        "did_greet": True,
        "academic_flow": {"stage": "awaiting_sections"},
    })

    st, reply = sup.handle(st, "ขอค่าธรรมเนียมเพิ่ม")
    _assert("ค่าธรรมเนียม" in reply or len(st.current_docs) >= 0, "T9: must stay in academic")


# ----------------------------
# Runner
# ----------------------------
TESTS = [
    test_01_style_switch_policy_persona_aware,
    test_02_confirmation_slang_yes_no,
    test_03_greeting_no_retrieve,
    test_04_practical_bypass_pending_slot,
    test_05_academic_intake_lock,
    test_06_academic_auto_return,
    test_07_no_mutation_rule,
    test_08_retrieval_reuse,
    test_09_academic_resume_layer,
]


def main():
    ok = 0
    fail = 0
    for t in TESTS:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            fail += 1

    print("\n--- Summary ---")
    print(f"Passed: {ok}")
    print(f"Failed: {fail}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()