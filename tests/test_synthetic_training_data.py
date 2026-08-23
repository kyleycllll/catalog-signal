import json
import runpy
from pathlib import Path


def test_generated_synthetic_examples_are_catalog_grounded(monkeypatch, tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/products.json").write_text(json.dumps([{"id":"p1","category":"scarf","price":10,"colour":"black","material":"wool"}]))
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(Path(__file__).parents[1] / "scripts/generate_synthetic_intent_data.py"), run_name="__main__")
    rows = [json.loads(line) for line in (tmp_path / "data/intent_examples.synthetic.jsonl").read_text().splitlines()]
    assert rows and all(row["source_type"] == "synthetic_catalog_grounded" and row["product_ids"] == ["p1"] for row in rows)
