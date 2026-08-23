from product_discovery.agent import referenced_product_id, select_tools
from product_discovery.memory import append_history, update_constraints
from product_discovery.schemas import Constraints, Intent, SessionState


def test_model_intent_updates_persistent_constraints():
    state = SessionState(id="s", constraints=Constraints(category="backpack", max_price=100))
    intent = Intent(category="backpack", required_attributes=["black"], max_price=80)
    updated = update_constraints(state, intent)
    assert updated.max_price == 80 and updated.required_attributes == ["black"]


def test_reference_selects_memory_lookup_and_resolves_result():
    state = SessionState(id="s", prior_result_ids=["a", "b"])
    intent = Intent(referenced_result=2)
    assert referenced_product_id(intent, state) == "b"
    assert "conversation_memory_lookup" in select_tools(intent)


def test_history_compacts():
    state = SessionState(id="s")
    for index in range(14):
        append_history(state, "user", f"message {index}")
    assert len(state.history) == 12 and "message 0" in state.context_summary
