#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_40_academic_flow.py
from __future__ import annotations


def test_academic_intake_lock_overrides_greeting(supervisor, retriever, new_state):
    st = new_state("academic")
    st.context.update({
        "did_greet": True,
        "academic_flow": {"stage": "awaiting_slots"},
    })

    st2, reply = supervisor.handle(st, "สวัสดี")
    # must NOT show greeting menu; must stay intake
    assert ("ข้อมูลจำเป็น" in reply) or ("ที่ตั้งร้าน" in reply)
    # intake path should not be greeting-menu; retrieval may happen based on your academic implementation
    # but key is: must not become greeting/menu reply


def test_academic_resume_layer_no_confirm(supervisor, new_state):
    st = new_state("academic")
    st.context.update({
        "did_greet": True,
        "academic_flow": {"stage": "awaiting_sections"},
    })

    st2, reply = supervisor.handle(st, "ขอค่าธรรมเนียมเพิ่ม")
    # must stay academic; no confirm
    assert st2.persona_id == "academic"
    assert st2.context.get("awaiting_persona_confirmation") is not True