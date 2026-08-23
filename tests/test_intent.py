import asyncio

from product_discovery.intent import extract_intent
from product_discovery.schemas import Constraints, Intent


class IntentModel:
    async def parse_intent(self, message, constraints, history):
        assert message == "Find a black backpack under $80"
        return Intent(category="backpack", required_attributes=["black"], max_price=80)


def test_intent_is_supplied_by_model_client_not_regex():
    intent = asyncio.run(extract_intent("Find a black backpack under $80", Constraints(), [], IntentModel()))
    assert intent.category == "backpack" and intent.max_price == 80
