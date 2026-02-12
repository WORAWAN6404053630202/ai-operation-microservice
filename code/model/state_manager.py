# code/model/state_manager.py
"""
State Manager Service
Handles persistence of conversation states

PRODUCTION FIXES:
- Persist directory is now stable (not dependent on current working directory).
- Supports env override via conf.STATE_DIR (if present).
"""

import json
from pathlib import Path
from typing import Optional

from model.conversation_state import ConversationState

try:
    import conf
except Exception:
    conf = None


class StateManager:
    """
    Manages conversation state persistence.

    Responsibilities:
    - Save conversation state to disk
    - Load conversation state from disk
    - Delete state when session ends
    """

    def __init__(self, persist_dir: str | None = None):
        # Prefer explicit arg > conf.STATE_DIR > stable default under code/data/states
        if persist_dir:
            base = Path(persist_dir)
        elif conf is not None and getattr(conf, "STATE_DIR", None):
            base = Path(getattr(conf, "STATE_DIR"))
        else:
            # Stable default relative to this file: code/data/states
            base = Path(__file__).resolve().parent.parent / "data" / "states"

        self.dir = base
        self.dir.mkdir(parents=True, exist_ok=True)

    def _safe_session_id(self, session_id: str) -> str:
        """Sanitize session_id for filesystem usage"""
        return (session_id or "").replace("/", "_").replace("\\", "_").strip()

    def _state_path(self, session_id: str) -> Path:
        safe_id = self._safe_session_id(session_id)
        return self.dir / f"{safe_id}.json"

    def save(self, session_id: str, state: ConversationState) -> None:
        """
        Save conversation state to disk (atomic write)
        """
        if not session_id:
            raise ValueError("session_id is required")

        state.session_id = session_id

        path = self._state_path(session_id)
        tmp_path = path.with_suffix(".tmp")

        payload = state.model_dump()

        payload.setdefault("_meta", {})
        payload["_meta"]["schema_version"] = payload["_meta"].get("schema_version", "v1")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        tmp_path.replace(path)

    def load(self, session_id: str) -> Optional[ConversationState]:
        """
        Load conversation state from disk
        """
        if not session_id:
            return None

        path = self._state_path(session_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.pop("_meta", None)

        return ConversationState(**data)

    def delete(self, session_id: str) -> None:
        """
        Delete a conversation state
        """
        if not session_id:
            return

        path = self._state_path(session_id)
        if path.exists():
            path.unlink()
