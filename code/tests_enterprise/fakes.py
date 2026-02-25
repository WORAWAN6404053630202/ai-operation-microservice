from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable


@dataclass
class FakeDoc:
    page_content: str
    metadata: Dict[str, Any]


class SpyRetriever:
    """
    Enterprise Spy Retriever:
    - records queries
    - supports deterministic routing by keywords
    - can simulate "no results"
    """
    def __init__(self):
        self.queries: List[str] = []
        self.force_empty: bool = False

    def invoke(self, query: str) -> List[FakeDoc]:
        q = (query or "").strip()
        self.queries.append(q)

        if self.force_empty:
            return []

        # Social security / fund
        if re.search(r"(ประกันสังคม|กองทุน|ขึ้นทะเบียน|นายจ้าง)", q):
            return [
                FakeDoc(
                    page_content="ขั้นตอนขึ้นทะเบียนประกันสังคม: ...",
                    metadata={
                        "department": "สำนักงานประกันสังคม",
                        "license_type": "ขึ้นทะเบียนนายจ้าง",
                        "operation_steps": "1) ยื่นคำขอ 2) แนบเอกสาร",
                        "fees": "ไม่มีค่าธรรมเนียม",
                        "service_channel": "ออนไลน์/สำนักงานประกันสังคม",
                    },
                )
            ]

        # VAT
        if re.search(r"(vat|ภพ\.?20|ภาษี)", q, re.IGNORECASE):
            return [
                FakeDoc(
                    page_content="VAT/ภพ.20: ...",
                    metadata={
                        "department": "กรมสรรพากร",
                        "license_type": "ภพ.20",
                        "fees": "ไม่มีค่าธรรมเนียม",
                        "service_channel": "สำนักงานสรรพากรพื้นที่/ออนไลน์",
                    },
                )
            ]

        # Default generic
        return [
            FakeDoc(
                page_content=f"generic result for: {q}",
                metadata={
                    "department": "หน่วยงานตัวอย่าง",
                    "license_type": "หัวข้อทั่วไป",
                },
            )
        ]


class LLMCallStats:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def record(self, kind: str, payload: Dict[str, Any]):
        self.calls.append({"kind": kind, **payload})

    def count(self, kind: Optional[str] = None) -> int:
        if kind is None:
            return len(self.calls)
        return sum(1 for c in self.calls if c.get("kind") == kind)


class FakeLLMJSON:
    """
    Enterprise fake for ChatOpenAI.invoke that returns a JSON string (model_kwargs response_format=json_object).
    Use:
      fake = FakeLLMJSON(stats, router=function)
      monkeypatch ChatOpenAI.invoke => fake.invoke
    """
    def __init__(self, stats: LLMCallStats, router: Callable[[str], Dict[str, Any]]):
        self.stats = stats
        self.router = router

    class _Msg:
        def __init__(self, content: str):
            self.content = content

    def invoke(self, messages):
        # messages: list[HumanMessage] etc. We'll just join contents
        text = ""
        try:
            text = "\n".join([(getattr(m, "content", "") or "") for m in (messages or [])])
        except Exception:
            text = str(messages)

        obj = self.router(text)
        self.stats.record("llm_invoke", {"prompt_chars": len(text)})

        import json
        return self._Msg(json.dumps(obj, ensure_ascii=False))