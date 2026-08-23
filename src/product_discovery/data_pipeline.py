"""Reproducible import and validation for Amazon Berkeley Objects (ABO) listings."""
from __future__ import annotations
import gzip
import json
import re
from pathlib import Path
from typing import Iterable
from .schemas import Product


def _value(values: object, language: str = "en_US") -> str | None:
    if not isinstance(values, list): return None
    for item in values:
        if isinstance(item, dict) and item.get("language_tag") == language and item.get("value"):
            return str(item["value"])
    for item in values:
        if isinstance(item, dict) and item.get("value"): return str(item["value"])
    return None


def _attribute_values(row: dict, key: str) -> list[str]:
    values = row.get(key, [])
    if not isinstance(values, list): return []
    output = []
    for value in values:
        text = value.get("value") if isinstance(value, dict) else value
        if text: output.append(str(text).lower())
    return output


def normalize_abo_listing(row: dict, image_paths: dict[str, str] | None = None) -> Product | None:
    """Map one official ABO listing to the app schema without inventing absent fields."""
    item_id = row.get("item_id")
    title = _value(row.get("item_name"))
    product_type = _value(row.get("product_type"))
    if not item_id or not title or not product_type: return None
    descriptions = _attribute_values(row, "product_description") + _attribute_values(row, "bullet_point")
    colour = _attribute_values(row, "color")
    material = _attribute_values(row, "material")
    image_id = row.get("main_image_id")
    attributes = list(dict.fromkeys(colour + material + _attribute_values(row, "style") + _attribute_values(row, "pattern")))
    return Product(id=str(item_id), title=title, description=" ".join(descriptions), category=product_type.lower(),
                   colour=colour[0] if colour else None, brand=_value(row.get("brand")),
                   material=material[0] if material else None, attributes=attributes,
                   image_path=(image_paths or {}).get(str(image_id)))


def read_jsonl_or_gz(path: str | Path) -> Iterable[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def import_abo(listings_path: str | Path, output_path: str | Path, limit: int = 1000) -> dict[str, int]:
    accepted, rejected, products = 0, 0, []
    for row in read_jsonl_or_gz(listings_path):
        product = normalize_abo_listing(row)
        if product is None: rejected += 1; continue
        products.append(product.model_dump())
        accepted += 1
        if accepted >= limit: break
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(products, indent=2), encoding="utf-8")
    return {"accepted": accepted, "rejected": rejected}


def validate_catalog(path: str | Path) -> list[Product]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    products = [Product.model_validate(row) for row in rows]
    identifiers = [product.id for product in products]
    if len(set(identifiers)) != len(identifiers): raise ValueError("Product IDs must be unique")
    if not products: raise ValueError("Catalog is empty")
    return products


COLOURS = {"black", "white", "blue", "red", "pink", "green", "yellow", "orange", "purple", "brown", "beige", "grey", "gray", "gold", "silver", "ivory", "cream", "navy"}
MATERIALS = {"wool", "leather", "cotton", "silk", "linen", "cashmere", "nylon", "polyester", "denim", "suede", "rubber", "canvas"}


def _clean_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def normalize_farfetch_product(row: dict, source_root: str | Path) -> Product | None:
    """Normalize the downloaded Kaggle Farfetch schema with an existing local image."""
    product_id, title = str(row.get("item_id", "")).strip(), str(row.get("title", "")).strip()
    if not product_id or not title: return None
    image_names = [name.strip() for name in str(row.get("image_file", "")).split("|") if name.strip()]
    image_path = next((Path(source_root) / name for name in image_names if (Path(source_root) / name).exists()), None)
    if not image_path: return None
    description = str(row.get("description") or "")
    details = _clean_html(str(row.get("details") or ""))
    searchable = f"{title} {description} {details} {row.get('breadcrumbs', '')}".lower()
    attributes = sorted({word for word in COLOURS | MATERIALS if re.search(rf"\b{re.escape(word)}\b", searchable)})
    breadcrumbs = [segment.strip() for segment in str(row.get("breadcrumbs") or "").split(",") if segment.strip()]
    try: price = float(row["price"])
    except (KeyError, TypeError, ValueError): price = None
    return Product(id=product_id, title=title, description=description, category=(breadcrumbs[-1] if breadcrumbs else "fashion").lower(),
                   price=price, colour=next((value for value in attributes if value in COLOURS), None), brand=row.get("brand"),
                   material=next((value for value in attributes if value in MATERIALS), None), attributes=attributes,
                   image_path=str(image_path))


def import_farfetch(metadata_path: str | Path, output_path: str | Path, source_root: str | Path) -> dict[str, int]:
    rows = json.loads(Path(metadata_path).read_text(encoding="utf-8")); products = []
    for row in rows:
        product = normalize_farfetch_product(row, source_root)
        if product: products.append(product.model_dump())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(products, indent=2), encoding="utf-8")
    return {"source_rows": len(rows), "accepted": len(products), "rejected": len(rows) - len(products)}
