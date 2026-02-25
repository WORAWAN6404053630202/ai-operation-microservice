#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_70_no_mutation.py
from __future__ import annotations


def test_append_only_messages(supervisor, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    st, _ = supervisor.handle(st, "ขึ้นทะเบียนนายจ้างต้องทำยังไง")
    n1 = len(st.messages)

    st, _ = supervisor.handle(st, "ขอเพิ่มรายละเอียด")
    n2 = len(st.messages)

    assert n2 > n1