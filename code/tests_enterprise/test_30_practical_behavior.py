#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_30_practical_behavior.py
from __future__ import annotations

import re


def test_practical_legal_question_must_retrieve(supervisor, retriever, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    st2, reply = supervisor.handle(st, "ขึ้นทะเบียนกองทุนประกันสังคมต้องทำยังไง")
    assert len(retriever.queries) >= 1
    assert "ประกันสังคม" in retriever.queries[-1]


def test_practical_followup_reuse_or_controlled(supervisor, retriever, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    st, _ = supervisor.handle(st, "ขึ้นทะเบียนนายจ้างต้องทำยังไง")
    q1 = retriever.queries[-1]

    st, _ = supervisor.handle(st, "ค่าธรรมเนียมเท่าไร")
    q2 = retriever.queries[-1]

    assert (q1 == q2) or (len(retriever.queries) <= 2)