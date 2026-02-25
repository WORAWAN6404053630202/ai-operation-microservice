#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_10_opening_and_greeting.py
from __future__ import annotations


def test_opening_blank_should_greet_no_retrieve(supervisor, retriever, new_state):
    st = new_state("practical")
    st2, reply = supervisor.handle(st, "")
    assert "สวัสดี" in reply or "อยากให้ช่วย" in reply or "ผู้ช่วย" in reply
    assert len(retriever.queries) == 0


def test_greeting_never_retrieve(supervisor, retriever, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    st2, reply = supervisor.handle(st, "สวัสดีค่าาาา")
    assert len(retriever.queries) == 0
    assert ("1)" in reply) or ("พิมพ์คำถาม" in reply) or ("ต้องการให้ช่วย" in reply)


def test_noise_never_retrieve(supervisor, retriever, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    st2, reply = supervisor.handle(st, "55555")
    assert len(retriever.queries) == 0
    # allow human small-talk prompt
    assert ("พิมพ์คำถาม" in reply) or ("สวัสดี" in reply) or ("ต้องการให้ช่วย" in reply)


def test_thanks_never_retrieve(supervisor, retriever, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    st2, reply = supervisor.handle(st, "ขอบคุณนะ")
    assert len(retriever.queries) == 0