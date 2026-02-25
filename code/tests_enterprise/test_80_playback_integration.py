#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_80_playback_integration.py
from __future__ import annotations

from .playback import run_playback


def test_playback_end_to_end_practical_then_style_switch(supervisor, retriever, llm_stats, new_state):
    st = new_state("practical")
    # opening already greeted for test stability
    st.context["did_greet"] = True

    turns = [
        "ขึ้นทะเบียนนายจ้างต้องทำยังไง",
        "ค่าธรรมเนียมเท่าไร",
        "ขอแบบละเอียดตามกฎหมายหน่อย",
        "เยปปป",  # confirm yes -> switch to academic
        "สวัสดี",  # must stay in intake if awaiting slots
    ]

    st2, results = run_playback(
        supervisor,
        st,
        turns,
        retriever_spy=retriever,
        llm_stats=llm_stats,
    )

    # assertions across timeline
    assert results[0].retrieval_calls >= 1
    assert st2.persona_id in {"academic", "practical"}  # depends on flow stage
    # key: after confirm yes, persona must become academic
    assert any(r.persona_id == "academic" for r in results), "must switch to academic in playback"