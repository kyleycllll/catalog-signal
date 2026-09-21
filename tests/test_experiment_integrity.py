"""Integrity checks for the held-out ESCI retrieval/reranking experiment."""
import asyncio
import importlib.util
import json
import re
from pathlib import Path

import pytest

from product_discovery.esci_catalog import stable_sample
from product_discovery.local_reranker import classification_prompt, parse_label, prompt_row
from product_discovery.retrieval import rerank_with_sft
from product_discovery.schemas import ModelPrediction, Product, RelevanceLabel, SessionState

ROOT = Path(__file__).resolve().parents[1]


def _prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_esci", ROOT / "scripts" / "prepare_esci.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook_prompt_function():
    notebook = json.loads((ROOT / "notebooks" / "esci_sft_reranker_colab.ipynb").read_text())
    source = "".join(notebook["cells"][3]["source"])
    start = source.index("LABEL_HELP =")
    end = source.index("def training_text")
    namespace = {"pd": pytest.importorskip("pandas")}
    exec(source[start:end], namespace)  # the notebook's own LABEL_HELP/clean/classification_prompt
    return namespace["classification_prompt"]


def test_local_prompt_is_identical_to_notebook_prompt():
    notebook_prompt = _notebook_prompt_function()
    row = {
        "query": "usb c cable 6ft",
        "product_title": "Braided USB-C Cable",
        "product_brand": "Acme",
        "product_color": "",
        "product_bullet_point": "x" * 2000,
        "product_description": "Fast charging " * 200,
    }
    assert classification_prompt(row) == notebook_prompt(row)
    product = Product(id="p1", title="Braided USB-C Cable", description="Fast", brand=None, colour="Black")
    assert classification_prompt(prompt_row("usb c", product)) == notebook_prompt(prompt_row("usb c", product))


def test_label_parser_matches_notebook_fallback():
    assert parse_label('{"label":"S"}') == ("S", True)
    assert parse_label('{"label" : "C"} trailing') == ("C", True)
    assert parse_label("no json") == ("I", False)


class FixedLabelModel:
    def __init__(self, labels):
        self.labels = labels

    async def rerank(self, query, products):
        return [ModelPrediction(product_id=p.id, label=RelevanceLabel(self.labels[p.id])) for p in products]


def test_reranking_only_permutes_fixed_candidates_and_label_dominates():
    products = [Product(id=f"p{i}", title=f"item {i}") for i in range(5)]
    retrieval = {p.id: {"retrieval": 1.0 / (61 + i)} for i, p in enumerate(products)}
    labels = {"p0": "I", "p1": "C", "p2": "E", "p3": "S", "p4": "E"}
    ranked = asyncio.run(
        rerank_with_sft("q", products, retrieval, SessionState(id="t"), FixedLabelModel(labels))
    )
    ids = [row.product.id for row in ranked]
    assert sorted(ids) == sorted(p.id for p in products)
    # Gains dominate; within a label, original retrieval order is kept.
    assert ids == ["p2", "p4", "p3", "p1", "p0"]


def test_split_integrity_rejects_query_leakage():
    prepare = _prepare_module()
    rows = [
        {"query_id": "1", "prepared_split": "train", "esci_label": "E"},
        {"query_id": "1", "prepared_split": "test", "esci_label": "S"},
    ]
    with pytest.raises(ValueError, match="leaks"):
        prepare.validate_split_integrity(rows)
    assert prepare.prepared_split("test", "42", 42) == "test"
    assert prepare.prepared_split("train", "42", 42) in {"train", "validation"}


def test_query_sample_is_deterministic_and_order_independent():
    queries = [{"query_id": str(i)} for i in range(100)]
    first = stable_sample(queries, 10, 42)
    second = stable_sample(list(reversed(queries)), 10, 42)
    assert first == second and len(first) == 10


def test_heldout_config_preregisters_validation_selection():
    config = json.loads((ROOT / "configs" / "esci_heldout.json").read_text())
    assert config["retrieval"]["selection"]["split"] == "validation"
    assert config["retrieval"]["rrf_k"] == 60
    assert config["retrieval"]["candidate_ks"] == [10, 40]


def test_saved_manifest_split_integrity_if_present():
    path = ROOT / "data" / "manifests" / "esci_split_manifest.json"
    if not path.exists():
        pytest.skip("data manifest not generated")
    manifest = json.loads(path.read_text())
    integrity = manifest["query_split_integrity"]
    assert integrity["query_sets_disjoint"] and integrity["official_test_queries_final_only"]
    assert manifest["source"]["files"]["products"]["matches_git_lfs_pointer"] is True
    assert manifest["source"]["files"]["examples"]["matches_git_lfs_pointer"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["catalog"]["content_fingerprint"])
