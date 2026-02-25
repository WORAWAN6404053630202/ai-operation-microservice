#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_00_contracts.py
from __future__ import annotations

def test_contract_supervisor_handle_returns_tuple(supervisor, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True
    st2, reply = supervisor.handle(st, "สวัสดี")
    assert st2 is not None
    assert isinstance(reply, str)
    assert isinstance(st2.context, dict)