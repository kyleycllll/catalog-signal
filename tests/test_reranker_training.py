import json

import pytest

from product_discovery.cross_encoder_reranker import cross_encoder_product_text
from product_discovery.local_reranker import classification_prompt
from product_discovery.reranker_training import (
    expected_gain,
    expected_gain_order,
    fit_prompt_ids,
    length_grouped_batches,
    qwen_training_example,
    target_text,
    validation_summary,
)
from product_discovery.schemas import Product


class CharTokenizer:
    """One token per character; EOS id 0. Enough to test masking and length control."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _row(**overrides):
    row = {
        "query": "cholula hot sauce",
        "product_title": "Cholula Original Hot Sauce",
        "product_brand": "Cholula",
        "product_color": "",
        "product_bullet_point": "b" * 900,
        "product_description": "d" * 900,
        "label": "E",
    }
    row.update(overrides)
    return row


def test_training_example_masks_prompt_and_keeps_label_and_eos():
    tokenizer = CharTokenizer()
    example = qwen_training_example(_row(), tokenizer, max_length=4096)
    target = [ord(c) for c in target_text("E")] + [0]
    assert example["input_ids"][-len(target):] == target
    assert example["labels"][-len(target):] == target
    assert set(example["labels"][: -len(target)]) == {-100}
    assert example["input_ids"][: -len(target)] == [ord(c) for c in classification_prompt(_row())]
    assert not example["shortened"]


def test_label_is_never_truncated_when_prompt_is_too_long():
    tokenizer = CharTokenizer()
    example = qwen_training_example(_row(label="C"), tokenizer, max_length=600)
    assert example["shortened"]
    assert len(example["input_ids"]) <= 600
    decoded = "".join(chr(t) for t in example["input_ids"][:-1])
    assert decoded.endswith('JSON:{"label":"C"}')
    assert "Query: cholula hot sauce" in decoded


def test_fit_prompt_rejects_impossible_budget():
    with pytest.raises(ValueError):
        fit_prompt_ids(_row(), CharTokenizer(), max_length=50, reserved=10)


def test_target_text_rejects_non_esci_label():
    assert json.loads(target_text("S")) == {"label": "S"}
    with pytest.raises(ValueError):
        target_text("UNJUDGED")


def test_length_grouped_batches_cover_each_index_once_and_are_deterministic():
    lengths = [5, 1, 9, 3, 7, 2, 8, 4, 6, 10, 11]
    first = length_grouped_batches(lengths, batch_size=3, mega=6, seed=1)
    assert sorted(i for batch in first for i in batch) == list(range(len(lengths)))
    assert first == length_grouped_batches(lengths, batch_size=3, mega=6, seed=1)
    assert all(len(batch) <= 3 for batch in first)


def test_expected_gain_and_order():
    assert expected_gain([1, 0, 0, 0]) == 3
    assert expected_gain([0, 0, 0, 1]) == 0
    assert expected_gain([0.5, 0.5, 0, 0]) == pytest.approx(2.5)
    products = [Product(id=pid, title=pid, description="") for pid in ("a", "b", "c")]
    retrieval = {"a": {"retrieval": 3.0}, "b": {"retrieval": 2.0}, "c": {"retrieval": 1.0}}
    # c has the highest expected gain, so it overtakes the retrieval order.
    assert expected_gain_order(products, retrieval, {"a": 1.0, "b": 1.0, "c": 3.0}) == ["c", "a", "b"]
    # Equal gains fall back to retrieval order.
    assert expected_gain_order(products, retrieval, {"a": 2.0, "b": 2.0, "c": 2.0}) == ["a", "b", "c"]


def test_validation_summary_separates_unjudged_rows():
    rows = [{"label": "E"}, {"label": "E"}, {"label": "I"}, {"label": "UNJUDGED"}, {"label": "UNJUDGED"}]
    summary = validation_summary(rows, ["E", "C", "I", "E", "I"])
    assert summary["examples"] == 3
    assert summary["e_to_c"] == 1 and summary["e_to_i"] == 0
    assert summary["e_to_c_or_i_rate"] == 0.5
    assert summary["unjudged_predicted_e_or_s_rate"] == 0.5


def test_cross_encoder_text_uses_prompt_fields_and_caps():
    text = cross_encoder_product_text(_row(product_color="red"))
    assert text.startswith("Cholula Original Hot Sauce | Brand: Cholula | Color: red | ")
    assert text.count("b") == 900 and text.count("d") >= 900
    assert "Color:" not in cross_encoder_product_text(_row())
