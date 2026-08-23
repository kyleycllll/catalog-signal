from __future__ import annotations
import json
import os
from pathlib import Path
from .schemas import SessionState


class SessionStore:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.getenv("SESSION_DB_PATH", "data/sessions.json"))
        self._sessions: dict[str, SessionState] = {}
        if self.path.exists():
            self._sessions = {key: SessionState.model_validate(value) for key, value in json.loads(self.path.read_text()).items()}

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def save(self, state: SessionState) -> None:
        self._sessions[state.id] = state
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({key: value.model_dump(mode="json") for key, value in self._sessions.items()}, indent=2))
