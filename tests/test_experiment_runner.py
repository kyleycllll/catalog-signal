import asyncio
import importlib.util
import json
from pathlib import Path


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "run_search_experiments.py"
    spec = importlib.util.spec_from_file_location("search_experiment_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_runner_writes_machine_and_human_readable_baseline_evidence(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([
        {"id": "a", "title": "trail backpack", "category": "bag"},
        {"id": "b", "title": "office chair", "category": "furniture"},
    ]))
    labels = tmp_path / "test.jsonl"
    labels.write_text("\n".join([
        json.dumps({"query_id": "q1", "query": "trail backpack", "label": "E", "product": {"id": "a"}}),
        json.dumps({"query_id": "q1", "query": "trail backpack", "label": "I", "product": {"id": "b"}}),
    ]) + "\n")
    output = tmp_path / "report"
    config = {
        "catalog_path": str(catalog),
        "evaluation_jsonl": str(labels),
        "candidate_k": 2,
        "systems": [{"name": "BM25", "strategy": "bm25", "rerank": False}],
    }
    report = asyncio.run(_runner_module().run(config, output))
    assert report["results"][0]["status"] == "completed"
    assert "| BM25 | completed" in (output / "comparison.md").read_text()
    assert json.loads((output / "results.json").read_text())["dataset"]["queries"] == 1
