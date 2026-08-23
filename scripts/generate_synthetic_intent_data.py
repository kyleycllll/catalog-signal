"""Generate explicitly synthetic, catalog-grounded LoRA intent examples.

These are training scaffolding, not human-labelled evaluation data. Every row points
to its Farfetch product ID so split leakage can be prevented and audited.
"""
from __future__ import annotations
import json
from pathlib import Path


def target(category, required=None, preferred=None, excluded=None, max_price=None, referenced=None):
    return {"category": category, "required_attributes": required or [], "preferred_attributes": preferred or [],
            "excluded_attributes": excluded or [], "max_price": max_price, "min_price": None,
            "use_image_similarity": False, "referenced_result": referenced, "clarification_required": False,
            "search_strategy": ["hybrid_text", "metadata_filter"]}


products = json.loads(Path("data/processed/products.json").read_text(encoding="utf-8"))
examples = []
for product in products:
    category, price, identifier = product["category"], product.get("price"), product["id"]
    ceiling = float(int(price + 25)) if price is not None else None
    examples.append({"source_type": "synthetic_catalog_grounded", "product_ids": [identifier],
                     "conversation": [f"Find {category} options.", f"Show one under ${ceiling:.0f}."],
                     "target": target(category, max_price=ceiling)})
    if product.get("colour"):
        examples.append({"source_type": "synthetic_catalog_grounded", "product_ids": [identifier],
                         "conversation": [f"I need a {product['colour']} {category}."],
                         "target": target(category, required=[product["colour"]])})
    if product.get("material"):
        examples.append({"source_type": "synthetic_catalog_grounded", "product_ids": [identifier],
                         "conversation": [f"Find {category}.", f"Only include {product['material']} options."],
                         "target": target(category, required=[product["material"]])})
    examples.append({"source_type": "synthetic_catalog_grounded", "product_ids": [identifier],
                     "conversation": [f"Show me {category}.", "The second result is close, but I want a smaller version."],
                     "target": target(category, preferred=["compact"], referenced=2)})

destination = Path("data/intent_examples.synthetic.jsonl")
destination.write_text("\n".join(json.dumps(row) for row in examples) + "\n", encoding="utf-8")
manifest = {"source_type": "synthetic_catalog_grounded", "catalog": "data/processed/products.json", "products": len(products), "examples": len(examples), "warning": "Do not use this synthetic corpus as a final evaluation set or claim human-labelled performance."}
Path("data/intent_examples.synthetic.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(manifest)
