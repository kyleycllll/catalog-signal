import pytest

from product_discovery.cross_encoder_reranker import cross_encoder_product_text
from product_discovery.reranker_training import expected_gain, validation_summary


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


def test_expected_gain_uses_esci_order():
    assert expected_gain([1, 0, 0, 0]) == 3
    assert expected_gain([0, 0, 0, 1]) == 0
    assert expected_gain([0.5, 0.5, 0, 0]) == pytest.approx(2.5)


def test_validation_summary_separates_unjudged_rows():
    rows = [
        {"label": "E"},
        {"label": "E"},
        {"label": "I"},
        {"label": "UNJUDGED"},
        {"label": "UNJUDGED"},
    ]
    summary = validation_summary(rows, ["E", "C", "I", "E", "I"])
    assert summary["examples"] == 3
    assert summary["e_to_c"] == 1 and summary["e_to_i"] == 0
    assert summary["e_to_c_or_i_rate"] == 0.5
    assert summary["unjudged_predicted_e_or_s_rate"] == 0.5


def test_cross_encoder_text_uses_product_fields_and_caps():
    text = cross_encoder_product_text(_row(product_color="red"))
    assert text.startswith("Cholula Original Hot Sauce | Brand: Cholula | Color: red | ")
    assert text.count("b") == 900 and text.count("d") >= 900
    assert "Color:" not in cross_encoder_product_text(_row())
