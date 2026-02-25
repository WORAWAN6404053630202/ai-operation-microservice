#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_20_switching_rules.py
from __future__ import annotations


def test_style_request_requires_confirm(supervisor, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    st2, reply = supervisor.handle(st, "ขอแบบละเอียดหน่อย")
    assert st2.context.get("awaiting_persona_confirmation") is True
    assert st2.context.get("pending_persona") == "academic"
    assert "ใช่ไหม" in reply


def test_confirm_slang_yes_switches(supervisor, new_state):
    st = new_state("practical")
    st.context.update({
        "did_greet": True,
        "awaiting_persona_confirmation": True,
        "pending_persona": "academic",
    })

    st2, reply = supervisor.handle(st, "เยปปป")
    assert st2.persona_id == "academic"


def test_confirm_filler_does_not_switch(supervisor, new_state):
    st = new_state("practical")
    st.context.update({
        "did_greet": True,
        "awaiting_persona_confirmation": True,
        "pending_persona": "academic",
    })

    st2, reply = supervisor.handle(st, "ครับ")
    assert st2.persona_id == "practical"


def test_switch_without_target_asks_pick(supervisor, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    st2, reply = supervisor.handle(st, "เปลี่ยนโหมด")
    assert st2.context.get("awaiting_persona_pick") is True
    assert "1) practical" in reply and "2) academic" in reply