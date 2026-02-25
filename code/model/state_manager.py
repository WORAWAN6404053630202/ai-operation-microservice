# code/model/state_manager.py
"""
State Manager Service
Handles persistence of conversation states

PRODUCTION FIXES:
- Persist directory is stable (not dependent on current working directory).
- Supports env override via conf.STATE_DIR (if present).
- ✅ Best-effort cross-process file locking to prevent concurrent write clobber
- ✅ Payload trimming on save to reduce latency/state bloat (messages + internal_messages)
"""

from __future__ import annotations

import json
import os
import time
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

    Notes:
    - For true multi-worker production, prefer Redis/DB.
      This implementation adds a best-effort file lock to reduce collisions.
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

        # Lock behavior (best-effort)
        self._lock_timeout_s = float(getattr(conf, "STATE_LOCK_TIMEOUT_S", 2.0) if conf is not None else 2.0)
        self._lock_poll_s = float(getattr(conf, "STATE_LOCK_POLL_S", 0.05) if conf is not None else 0.05)

        # Trimming defaults (payload size)
        self._default_max_recent = int(getattr(conf, "MAX_RECENT_MESSAGES_SAVE", 18) if conf is not None else 18)
        self._default_max_internal = int(getattr(conf, "MAX_INTERNAL_MESSAGES_SAVE", 40) if conf is not None else 40)

    def _safe_session_id(self, session_id: str) -> str:
        """Sanitize session_id for filesystem usage"""
        return (session_id or "").replace("/", "_").replace("\\", "_").strip()

    def _state_path(self, session_id: str) -> Path:
        safe_id = self._safe_session_id(session_id)
        return self.dir / f"{safe_id}.json"

    def _lock_path(self, session_id: str) -> Path:
        safe_id = self._safe_session_id(session_id)
        return self.dir / f"{safe_id}.lock"

    # --------------------------
    # Best-effort file lock
    # --------------------------
    def _acquire_lock(self, session_id: str) -> None:
        """
        Cross-process best-effort lock using atomic create (O_EXCL).
        Writes owner pid + timestamp for observability and stale detection (best-effort).

        If lock cannot be acquired within timeout, raise TimeoutError.
        """
        lock_path = self._lock_path(session_id)
        deadline = time.time() + self._lock_timeout_s

        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    payload = {"pid": os.getpid(), "ts": time.time()}
                    os.write(fd, json.dumps(payload).encode("utf-8"))
                finally:
                    os.close(fd)
                return
            except FileExistsError:
                # If lock exists, check for staleness (best-effort)
                try:
                    stat = lock_path.stat()
                    age = time.time() - float(stat.st_mtime)
                    stale_after = float(getattr(conf, "STATE_LOCK_STALE_S", 15.0) if conf is not None else 15.0)
                    if age > stale_after:
                        # best-effort break stale lock
                        lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass

                if time.time() >= deadline:
                    raise TimeoutError(f"Could not acquire state lock for session_id={session_id!r}")
                time.sleep(self._lock_poll_s)

    def _release_lock(self, session_id: str) -> None:
        lock_path = self._lock_path(session_id)
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    # --------------------------
    # Payload trimming
    # --------------------------
    def _trim_state_for_save(self, state: ConversationState) -> None:
        """
        Trim payload to reduce on-disk size and downstream latency.

        Policy:
        - messages trimmed to strict_profile.max_recent_messages (fallback to default)
        - internal_messages trimmed to MAX_INTERNAL_MESSAGES_SAVE (fallback to default)
        - keep context/current_docs/retrieval tracking intact
        """
        # messages
        max_recent = None
        try:
            sp = getattr(state, "strict_profile", None) or {}
            if isinstance(sp, dict):
                v = sp.get("max_recent_messages")
                if v is not None:
                    max_recent = int(v)
        except Exception:
            max_recent = None

        if not max_recent or max_recent <= 0:
            max_recent = self._default_max_recent

        if isinstance(state.messages, list) and len(state.messages) > max_recent:
            state.messages = state.messages[-max_recent:]

        # internal_messages
        max_internal = self._default_max_internal
        if isinstance(state.internal_messages, list) and max_internal > 0 and len(state.internal_messages) > max_internal:
            state.internal_messages = state.internal_messages[-max_internal:]

    def save(self, session_id: str, state: ConversationState) -> None:
        """
        Save conversation state to disk (atomic write + best-effort lock)
        """
        if not session_id:
            raise ValueError("session_id is required")

        state.session_id = session_id

        # trim before dumping (reduces IO and state growth)
        self._trim_state_for_save(state)

        path = self._state_path(session_id)
        tmp_path = path.with_suffix(f".{os.getpid()}.tmp")

        self._acquire_lock(session_id)
        try:
            payload = state.model_dump()

            payload.setdefault("_meta", {})
            payload["_meta"]["schema_version"] = payload["_meta"].get("schema_version", "v1")
            payload["_meta"]["saved_at"] = time.time()

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            # atomic replace
            tmp_path.replace(path)
        finally:
            # best-effort cleanup tmp (in case of exception before replace)
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._release_lock(session_id)

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
        Delete a conversation state (best-effort)
        """
        if not session_id:
            return

        path = self._state_path(session_id)
        lock_path = self._lock_path(session_id)

        # Try to lock briefly to avoid deleting while writing (best-effort)
        try:
            self._acquire_lock(session_id)
        except Exception:
            # If can't lock, still best-effort delete
            pass

        try:
            if path.exists():
                path.unlink()
        finally:
            try:
                # cleanup lock file too
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._release_lock(session_id)