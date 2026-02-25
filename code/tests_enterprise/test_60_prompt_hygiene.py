#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_60_prompt_hygiene.py
from __future__ import annotations


def test_llm_prompt_budget_basic(supervisor, llm_stats, new_state):
    """
    Enterprise guard:
    - ensure LLM prompt size doesn't explode in normal flow
    - This is a basic check; tune the threshold to your production target.
    """
    st = new_state("practical")
    st.context["did_greet"] = True

    # trigger style classifier -> confirm
    supervisor.handle(st, "ขอแบบละเอียดหน่อย")

    # we patched LLM, so stats exists
    assert llm_stats.count() >= 1
    worst = max(c.get("prompt_chars", 0) for c in llm_stats.calls)
    assert worst < 20000  # tune based on your cap