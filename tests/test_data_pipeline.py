import gzip
import json
from product_discovery.data_pipeline import import_abo, import_farfetch, normalize_abo_listing, normalize_farfetch_product, validate_catalog


RAW = {"item_id":"x1","item_name":[{"language_tag":"en_US","value":"Blue pack"}],"product_type":[{"language_tag":"en_US","value":"Backpack"}],"item_description":[{"language_tag":"en_US","value":"Daily pack"}],"color":[{"value":"Blue"}],"material":[{"value":"Nylon"}],"main_image_id":"img"}


def test_normalizes_without_inventing_price():
    product = normalize_abo_listing(RAW)
    assert product and product.id == "x1" and product.price is None and product.category == "backpack"


def test_import_and_validate_catalog(tmp_path):
    raw = tmp_path / "listings.json.gz"; output = tmp_path / "products.json"
    with gzip.open(raw, "wt") as handle: handle.write(json.dumps(RAW) + "\n")
    assert import_abo(raw, output, 10)["accepted"] == 1
    assert validate_catalog(output)[0].title == "Blue pack"


def test_normalizes_farfetch_with_local_image(tmp_path):
    image = tmp_path / "images" / "99_0.jpg"; image.parent.mkdir(); image.write_bytes(b"image")
    row = {"item_id":"99", "title":"Black wool scarf", "price":"50.00", "brand":"Brand", "breadcrumbs":"Women, Accessories", "description":"warm", "image_file":"images/99_0.jpg"}
    product = normalize_farfetch_product(row, tmp_path)
    assert product and product.price == 50 and product.colour == "black" and product.material == "wool"
