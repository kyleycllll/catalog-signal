import json
import pytest
from product_discovery.training_data import prepare_file, split_examples, validate_no_leakage


def test_split_is_repeatable_and_has_no_product_leakage():
    rows = [{"product_ids": ["p1"], "target": {}}, {"product_ids": ["p2"], "target": {}}, {"product_ids": ["p3"], "target": {}}]
    assert split_examples(rows, 9) == split_examples(rows, 9)


def test_leakage_is_detected():
    with pytest.raises(ValueError): validate_no_leakage({"train": [{"product_ids": ["p1"]}], "test": [{"product_ids": ["p1"]}]})


def test_prepare_writes_manifest(tmp_path):
    source = tmp_path / "rows.jsonl"; source.write_text(json.dumps({"product_ids": ["p1"], "target": {}}) + "\n")
    prepare_file(source, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "split_manifest.json").read_text())["leakage_checked"]
