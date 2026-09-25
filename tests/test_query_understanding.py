from product_discovery.query_understanding import analyze_query


def test_normalizes_unicode_punctuation_and_hyphenated_model_tokens():
    analysis = analyze_query("  MÁKITA—DTD-172!!! Impact\tDrill  ")

    assert analysis.normalized_query == "makita dtd172 impact drill"
    assert analysis.retrieval_query == "makita dtd172 impact drill"
    assert analysis.model_tokens == ("dtd172",)


def test_preserves_spaced_brand_and_model_identifier():
    analysis = analyze_query("Makita DTD172 impact drill")

    assert analysis.brands == ("makita",)
    assert analysis.model_tokens == ("dtd172",)
    assert analysis.numeric_tokens == ("dtd172",)


def test_compact_airmax_matches_the_spaced_product_name():
    analysis = analyze_query("nike airmax black")

    assert analysis.normalized_query == "nike air max black"
    assert analysis.tokens == ("nike", "air", "max", "black")
    assert analysis.colors == ("black",)


def test_hyphenated_words_use_the_same_compact_spelling():
    analysis = analyze_query("post-partum support belt")

    assert analysis.normalized_query == "postpartum support belt"


def test_extracts_negation_without_polluting_retrieval_query():
    analysis = analyze_query("tervis cups without lid")

    assert analysis.normalized_query == "tervis cups without lid"
    assert analysis.retrieval_query == "tervis cups"
    assert analysis.tokens == ("tervis", "cups")
    assert analysis.exclusions == ("lid",)
    assert analysis.contains_negation is True
