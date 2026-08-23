from __future__ import annotations
from .schemas import Constraints, Intent, SessionState


def update_constraints(state: SessionState, intent: Intent) -> Constraints:
    old = state.constraints
    state.constraints = Constraints(
        category=intent.category or old.category,
        required_attributes=list(dict.fromkeys(intent.required_attributes)),
        preferred_attributes=list(dict.fromkeys(intent.preferred_attributes)),
        excluded_attributes=list(dict.fromkeys(intent.excluded_attributes)),
        max_price=intent.max_price if intent.max_price is not None else old.max_price,
        min_price=intent.min_price if intent.min_price is not None else old.min_price,
    )
    return state.constraints


def append_history(state: SessionState, role: str, content: str, keep_last: int = 12) -> None:
    state.history.append({"role": role, "content": content})
    if len(state.history) > keep_last:
        compacted = state.history[:-keep_last]
        summary = "; ".join(f"{row['role']}: {row['content'][:120]}" for row in compacted)
        state.context_summary = (state.context_summary + " | " + summary).strip(" | ")[-2000:]
        state.history = state.history[-keep_last:]
