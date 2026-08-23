"""Create an annotation-ready JSONL file for the LoRA structured-intent task."""
import json
from pathlib import Path

examples = [
    {"conversation": ["Find black backpacks under $80", "Only waterproof options"], "target": {"category": "backpack", "required_attributes": ["black", "waterproof"], "preferred_attributes": [], "excluded_attributes": [], "max_price": 80, "use_image_similarity": False, "referenced_result": None, "clarification_required": False, "search_strategy": ["hybrid_text", "metadata_filter"]}},
    {"conversation": ["Show backpacks", "The second result is close, but smaller"], "target": {"category": "backpack", "required_attributes": [], "preferred_attributes": ["compact"], "excluded_attributes": [], "max_price": None, "use_image_similarity": False, "referenced_result": 2, "clarification_required": False, "search_strategy": ["hybrid_text", "metadata_filter"]}},
]
Path("data/intent_train_template.jsonl").write_text("\n".join(json.dumps(row) for row in examples) + "\n")
print("Wrote data/intent_train_template.jsonl. Add real, licensed catalogue-grounded examples before training.")
